// Shape operations: broadcast, reshape, convert, slice, concatenate,
// custom_call, etc.

#import "pjrt_plugin/ops/registry.h"
#include <optional>

namespace jax_mps {

static std::optional<int64_t> TryGetConstScalarInt(mlir::Value value) {
    if (auto cstOp = value.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
        auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(cstOp.getValue());
        if (dense && dense.getNumElements() == 1) {
            auto it = dense.getValues<llvm::APInt>().begin();
            return (*it).getSExtValue();
        }
        return std::nullopt;
    }
    if (auto broadcastOp = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        return TryGetConstScalarInt(broadcastOp.getOperand());
    }
    if (auto reshapeOp = value.getDefiningOp<mlir::stablehlo::ReshapeOp>()) {
        return TryGetConstScalarInt(reshapeOp.getOperand());
    }
    if (auto convertOp = value.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
        return TryGetConstScalarInt(convertOp.getOperand());
    }
    return std::nullopt;
}

static MPSGraphTensor* Handle_broadcast(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    return [g broadcastTensor:input toShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.broadcast", Handle_broadcast);

// broadcast_in_dim needs special handling for dimension mapping
static MPSGraphTensor* Handle_broadcast_in_dim(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op);
    if (!broadcastOp) {
        MPS_LOG_ERROR("Expected BroadcastInDimOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        MPS_LOG_ERROR("broadcast_in_dim input tensor not found\n");
        return nullptr;
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;

    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    NSUInteger outputRank = outputShape.count;

    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    // If broadcast_dims is empty, just broadcast directly
    if (broadcastDims.empty()) {
        return [g broadcastTensor:input toShape:outputShape name:nil];
    }

    // If ranks already match, just broadcast
    if (inputRank == outputRank) {
        return [g broadcastTensor:input toShape:outputShape name:nil];
    }

    // Build intermediate shape: start with all 1s, then fill in from broadcast_dims
    NSMutableArray<NSNumber*>* intermediateShape = [NSMutableArray arrayWithCapacity:outputRank];
    for (NSUInteger i = 0; i < outputRank; i++) {
        [intermediateShape addObject:@1];
    }

    // Map input dimensions to output dimensions according to broadcast_dims
    for (size_t i = 0; i < broadcastDims.size() && i < inputRank; i++) {
        int64_t outDim = broadcastDims[i];
        if (outDim >= 0 && (NSUInteger)outDim < outputRank) {
            intermediateShape[outDim] = inputShape[i];
        }
    }

    // Reshape input to intermediate shape (same rank as output)
    MPSGraphTensor* reshaped = [g reshapeTensor:input withShape:intermediateShape name:nil];

    // Now broadcast to final output shape
    return [g broadcastTensor:reshaped toShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.broadcast_in_dim", Handle_broadcast_in_dim);

static MPSGraphTensor* Handle_reshape(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    return [g reshapeTensor:input withShape:outputShape name:nil];
}
REGISTER_MPS_OP("stablehlo.reshape", Handle_reshape);

static MPSGraphTensor* Handle_transpose(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op);
    if (!transposeOp) {
        MPS_LOG_ERROR("Expected TransposeOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto permutation = transposeOp.getPermutation();
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    for (int64_t d : permutation) {
        [perm addObject:@(d)];
    }

    return [g transposeTensor:input permutation:perm name:nil];
}
REGISTER_MPS_OP("stablehlo.transpose", Handle_transpose);

static MPSGraphTensor* Handle_convert(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        MPS_LOG_ERROR("Invalid dtype for convert operation\n");
        return nullptr;
    }
    return [g castTensor:input toType:dtype name:nil];
}
REGISTER_MPS_OP("stablehlo.convert", Handle_convert);

// Slice - extract a portion of a tensor (static indices)
static MPSGraphTensor* Handle_slice(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op);
    if (!sliceOp) {
        MPS_LOG_ERROR("Expected SliceOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    for (int64_t s : sliceOp.getStartIndices()) {
        [starts addObject:@(s)];
    }
    for (int64_t l : sliceOp.getLimitIndices()) {
        [ends addObject:@(l)];
    }
    for (int64_t s : sliceOp.getStrides()) {
        [strides addObject:@(s)];
    }

    return [g sliceTensor:input starts:starts ends:ends strides:strides name:nil];
}
REGISTER_MPS_OP("stablehlo.slice", Handle_slice);

// Dynamic slice - extract a portion using runtime indices
static MPSGraphTensor* Handle_dynamic_slice(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(op);
    if (!dynSliceOp) {
        MPS_LOG_ERROR("Expected DynamicSliceOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto sliceSizes = dynSliceOp.getSliceSizes();
    NSUInteger rank = sliceSizes.size();

    // Build the output shape from slice sizes
    NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];
    for (int64_t s : sliceSizes) {
        [outputShape addObject:@(s)];
    }

    // Get start indices as tensors (operands 1 through N)
    // and create coordinate tensors offset by the start indices
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        // Get the start index tensor for this dimension (scalar tensor)
        MPSGraphTensor* startIdx = GetInputTensor(values, op, dim + 1);
        if (!startIdx) {
            MPS_LOG_ERROR("dynamic_slice missing start index for dimension %lu\n",
                          (unsigned long)dim);
            return nullptr;
        }

        // Create coordinate tensor for this dimension (0, 1, 2, ..., slice_size-1)
        MPSGraphTensor* coords = [g coordinateAlongAxis:(NSInteger)dim
                                              withShape:outputShape
                                                   name:nil];

        // Cast coordinates to match start index type for addition
        coords = [g castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates (broadcasts the scalar start index)
        MPSGraphTensor* adjustedCoords = [g additionWithPrimaryTensor:coords
                                                      secondaryTensor:startIdx
                                                                 name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    // Shape: [slice_size_0, slice_size_1, ..., rank]
    MPSGraphTensor* indices = [g stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Use gatherND to gather the slice from the input tensor
    // batchDimensions: 0 means no batch dimensions
    return [g gatherNDWithUpdatesTensor:input indicesTensor:indices batchDimensions:0 name:nil];
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", Handle_dynamic_slice);

// Bitcast convert - reinterpret bits as a different type
static MPSGraphTensor* Handle_bitcast_convert(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        MPS_LOG_ERROR("Invalid dtype for bitcast_convert operation\n");
        return nullptr;
    }

    // MPS reinterpretCastTensor doesn't support rank-0 (scalar) tensors.
    // Work around by reshaping to rank-1, casting, then reshaping back.
    NSArray<NSNumber*>* inputShape = input.shape;
    bool isScalar = (inputShape.count == 0);

    if (isScalar) {
        // Reshape scalar to [1]
        input = [g reshapeTensor:input withShape:@[@1] name:nil];
    }

    // Use reinterpretCast which preserves bit patterns
    MPSGraphTensor* result = [g reinterpretCastTensor:input toType:dtype name:nil];

    if (isScalar) {
        // Reshape back to scalar
        result = [g reshapeTensor:result withShape:@[] name:nil];
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.bitcast_convert", Handle_bitcast_convert);

// Concatenate - joins tensors along a dimension
static MPSGraphTensor* Handle_concatenate(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op);
    if (!concatOp) {
        MPS_LOG_ERROR(" Expected ConcatenateOp\n");
        return nullptr;
    }

    // Gather all input tensors
    NSMutableArray<MPSGraphTensor*>* input_tensors = [NSMutableArray array];
    for (mlir::Value operand : op->getOperands()) {
        MPSGraphTensor* tensor = GetTensor(values, operand);
        if (tensor) {
            [input_tensors addObject:tensor];
        }
    }

    if (input_tensors.count == 0) {
        MPS_LOG_ERROR(" Concatenate operation has no valid inputs\n");
        return nullptr;
    }

    // Get the concatenate dimension from the op
    NSInteger dimension = static_cast<NSInteger>(concatOp.getDimension());

    return [g concatTensors:input_tensors dimension:dimension name:nil];
}
REGISTER_MPS_OP("stablehlo.concatenate", Handle_concatenate);

// Sharding is a marker used by JAX for partitioning - just pass through the input
static MPSGraphTensor* Handle_sharding(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    return GetInputTensor(values, op, 0);
}
REGISTER_CUSTOM_CALL_TARGET("Sharding", Handle_sharding);

// Custom call - generic dispatcher using CustomCallRegistry
static MPSGraphTensor* Handle_custom_call(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp) {
        MPS_LOG_ERROR("Expected CustomCallOp\n");
        return nullptr;
    }

    std::string target = customCallOp.getCallTargetName().str();

    auto handler = CustomCallRegistry::Find(target);
    if (handler) {
        return handler(g, op, values);
    }

    MPS_LOG_ERROR("Unknown custom_call target: %s\n", target.c_str());
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.custom_call", Handle_custom_call);

// Pad - add padding around tensor
static MPSGraphTensor* Handle_pad(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
    if (!padOp) {
        MPS_LOG_ERROR("Expected PadOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* paddingValue = GetInputTensor(values, op, 1);
    if (!input || !paddingValue)
        return nullptr;

    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();

    // Check if interior padding is all zeros (simple edge padding case)
    bool hasInteriorPadding = false;
    for (int64_t p : interiorPadding) {
        if (p != 0) {
            hasInteriorPadding = true;
            break;
        }
    }

    if (hasInteriorPadding) {
        MPS_LOG_ERROR("Interior padding not yet supported\n");
        return nullptr;
    }

    // Get output shape and create a tensor filled with padding value
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    MPSGraphTensor* padded = [g broadcastTensor:paddingValue toShape:outputShape name:nil];

    // Calculate starts and ends for sliceUpdate (where to place the input)
    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    NSArray<NSNumber*>* inputShape = input.shape;
    for (NSUInteger i = 0; i < edgePaddingLow.size(); i++) {
        int64_t start = edgePaddingLow[i];
        int64_t inputDim = [inputShape[i] longLongValue];
        [starts addObject:@(start)];
        [ends addObject:@(start + inputDim)];
        [strides addObject:@1];
    }

    // Use sliceUpdateDataTensor to insert input into the padded tensor
    return [g sliceUpdateDataTensor:padded
                       updateTensor:input
                             starts:starts
                               ends:ends
                            strides:strides
                               name:nil];
}
REGISTER_MPS_OP("stablehlo.pad", Handle_pad);

// Dynamic update slice - update a portion of a tensor with new values
static MPSGraphTensor* Handle_dynamic_update_slice(MPSGraph* g, mlir::Operation* op,
                                                   ValueMap& values) {
    auto updateSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(op);
    if (!updateSliceOp) {
        MPS_LOG_ERROR("Expected DynamicUpdateSliceOp\n");
        return nullptr;
    }

    MPSGraphTensor* operand = GetInputTensor(values, op, 0);
    MPSGraphTensor* update = GetInputTensor(values, op, 1);
    if (!operand || !update)
        return nullptr;

    NSArray<NSNumber*>* updateShape = update.shape;
    NSUInteger rank = updateShape.count;

    // Get start indices (operands 2 through N)
    NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        MPSGraphTensor* startIdx = GetInputTensor(values, op, i + 2);
        if (!startIdx) {
            MPS_LOG_ERROR("dynamic_update_slice missing start index for dimension %lu\n",
                          (unsigned long)i);
            return nullptr;
        }
        [startIndices addObject:startIdx];
    }

    // Build starts array by reading the scalar start indices
    // For sliceUpdateDataTensor, we need static starts/ends/strides
    // But the start indices are dynamic tensors, so we need to use scatter instead

    // Create coordinate tensors for the update region
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        MPSGraphTensor* startIdx = startIndices[dim];

        // Create coordinate tensor for this dimension (0, 1, 2, ..., update_size-1)
        MPSGraphTensor* coords = [g coordinateAlongAxis:(NSInteger)dim
                                              withShape:updateShape
                                                   name:nil];

        // Cast coordinates to match start index type
        coords = [g castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates
        MPSGraphTensor* adjustedCoords = [g additionWithPrimaryTensor:coords
                                                      secondaryTensor:startIdx
                                                                 name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    MPSGraphTensor* indices = [g stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Cast indices to int32 if needed
    indices = EnsureInt32(g, indices);

    // Use scatterND to update the operand at the specified indices
    return [g scatterNDWithDataTensor:operand
                        updatesTensor:update
                        indicesTensor:indices
                      batchDimensions:0
                                 mode:MPSGraphScatterModeSet
                                 name:nil];
}
REGISTER_MPS_OP("stablehlo.dynamic_update_slice", Handle_dynamic_update_slice);

// Gather - generalized indexing operation
// Handles embedding lookups and other gather patterns
static MPSGraphTensor* Handle_gather(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op);
    if (!gatherOp) {
        MPS_LOG_ERROR("Expected GatherOp\n");
        return nullptr;
    }

    MPSGraphTensor* operand = GetInputTensor(values, op, 0);
    MPSGraphTensor* startIndices = GetInputTensor(values, op, 1);
    if (!operand || !startIndices)
        return nullptr;

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    auto sliceSizes = gatherOp.getSliceSizes();

    NSArray<NSNumber*>* indicesShape = startIndices.shape;
    NSUInteger indicesRank = indicesShape.count;

    // Single-axis gather pattern used by take/take_along_axis variants.
    if (indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && startIndexMap.size() == 1 &&
        collapsedSliceDims.size() == 1 && collapsedSliceDims[0] == startIndexMap[0]) {
        int64_t gatherAxis = startIndexMap[0];

        // [batch..., 1] -> [batch...]
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* squeezedIndices = [g reshapeTensor:startIndices withShape:squeezedShape name:nil];
        squeezedIndices = EnsureInt32(g, squeezedIndices);

        // Infer batch dimensions from shared leading dimensions before gatherAxis.
        NSUInteger batchDims = 0;
        NSArray<NSNumber*>* operandShape = operand.shape;
        NSArray<NSNumber*>* squeezedIndicesShape = squeezedIndices.shape;
        while (batchDims < (NSUInteger)gatherAxis && batchDims < operandShape.count &&
               batchDims < squeezedIndicesShape.count &&
               [operandShape[batchDims] integerValue] == [squeezedIndicesShape[batchDims] integerValue]) {
            batchDims++;
        }

        return [g gatherWithUpdatesTensor:operand
                            indicesTensor:squeezedIndices
                                     axis:(NSUInteger)gatherAxis
                          batchDimensions:batchDims
                                     name:nil];
    }

    // Selector introduces a new axis:
    // operand [B, S, V], indices [B, T, 2] (batch, vocab) -> output [B, T, S].
    if (operand.shape.count == 3 && indicesRank == 3 && indexVectorDim == 2 &&
        startIndexMap.size() == 2 && collapsedSliceDims.size() == 2 && offsetDims.size() == 1 &&
        sliceSizes.size() == 3) {
        bool collapsed0 = false, collapsed2 = false;
        for (int64_t d : collapsedSliceDims) {
            if (d == 0)
                collapsed0 = true;
            else if (d == 2)
                collapsed2 = true;
        }
        int64_t compBatch = -1;
        int64_t compVocab = -1;
        for (size_t j = 0; j < startIndexMap.size(); ++j) {
            if (startIndexMap[j] == 0)
                compBatch = (int64_t)j;
            else if (startIndexMap[j] == 2)
                compVocab = (int64_t)j;
        }
        if (collapsed0 && collapsed2 && compBatch >= 0 && compVocab >= 0 && sliceSizes[0] == 1 &&
            sliceSizes[2] == 1) {
            MPSGraphTensor* vocab2D =
                [g sliceTensor:startIndices dimension:2 start:compVocab length:1 name:nil];
            MPSGraphTensor* vocabIdx =
                [g reshapeTensor:vocab2D withShape:@[ indicesShape[0], indicesShape[1] ] name:nil];
            vocabIdx = EnsureInt32(g, vocabIdx);

            MPSGraphTensor* gathered = [g gatherWithUpdatesTensor:operand
                                                     indicesTensor:vocabIdx
                                                              axis:2
                                                   batchDimensions:1
                                                              name:nil];
            gathered = [g transposeTensor:gathered permutation:@[ @0, @2, @1 ] name:nil];

            NSArray<NSNumber*>* outputShape = GetOutputShape(op);
            if (outputShape) {
                gathered = [g reshapeTensor:gathered withShape:outputShape name:nil];
            }
            return gathered;
        }
    }

    // Non-contiguous batched selector pattern:
    // operand [B, X, Z, Y], indices [B, 3] (batch, x, y) -> output [B, Z].
    if (indexVectorDim == 1 && indicesRank == 2 && operand.shape.count == 4 &&
        startIndexMap.size() == 3 && offsetDims.size() == 1 && offsetDims[0] == 1 &&
        collapsedSliceDims.size() == 3 && sliceSizes.size() == 4) {
        auto findIndexComponent = [&](int64_t operandDim) -> int64_t {
            for (size_t j = 0; j < startIndexMap.size(); ++j) {
                if (startIndexMap[j] == operandDim)
                    return (int64_t)j;
            }
            return -1;
        };

        bool collapsed0 = false, collapsed1 = false, collapsed3 = false;
        for (int64_t d : collapsedSliceDims) {
            if (d == 0)
                collapsed0 = true;
            else if (d == 1)
                collapsed1 = true;
            else if (d == 3)
                collapsed3 = true;
        }

        int64_t compBatch = findIndexComponent(0);
        int64_t compX = findIndexComponent(1);
        int64_t compY = findIndexComponent(3);
        if (collapsed0 && collapsed1 && collapsed3 && compBatch >= 0 && compX >= 0 && compY >= 0 &&
            sliceSizes[0] == 1 && sliceSizes[1] == 1 && sliceSizes[3] == 1) {
            NSArray<NSNumber*>* operandShape = operand.shape;
            int64_t batch = [operandShape[0] longLongValue];
            int64_t xSize = [operandShape[1] longLongValue];
            int64_t zSize = [operandShape[2] longLongValue];
            int64_t ySize = [operandShape[3] longLongValue];

            MPSDataType idxType = startIndices.dataType;
            MPSGraphTensor* zeroIdx = [g constantWithScalar:0 shape:@[ @1 ] dataType:idxType];
            MPSGraphTensor* oneIdx = [g constantWithScalar:1 shape:@[ @1 ] dataType:idxType];
            MPSGraphTensor* zSizeIdx = [g constantWithScalar:zSize shape:@[ @1 ] dataType:idxType];
            MPSGraphTensor* sizeVec = [g concatTensors:@[ oneIdx, zSizeIdx, oneIdx ] dimension:0 name:nil];

            NSMutableArray<MPSGraphTensor*>* batchRows = [NSMutableArray array];
            for (int64_t b = 0; b < batch; ++b) {
                MPSGraphTensor* batchSlice4 = [g sliceTensor:operand
                                                   dimension:0
                                                       start:b
                                                      length:1
                                                        name:nil];
                MPSGraphTensor* batchSlice =
                    [g reshapeTensor:batchSlice4 withShape:@[ @(xSize), @(zSize), @(ySize) ] name:nil];

                MPSGraphTensor* row2D = [g sliceTensor:startIndices
                                             dimension:0
                                                 start:b
                                                length:1
                                                  name:nil];
                MPSGraphTensor* x2D = [g sliceTensor:row2D dimension:1 start:compX length:1 name:nil];
                MPSGraphTensor* y2D = [g sliceTensor:row2D dimension:1 start:compY length:1 name:nil];
                MPSGraphTensor* x1D = EnsureInt32(g, [g reshapeTensor:x2D withShape:@[ @1 ] name:nil]);
                MPSGraphTensor* y1D = EnsureInt32(g, [g reshapeTensor:y2D withShape:@[ @1 ] name:nil]);
                MPSGraphTensor* startVec = [g concatTensors:@[ x1D, zeroIdx, y1D ] dimension:0 name:nil];

                MPSGraphTensor* slice = [g sliceTensor:batchSlice
                                           startTensor:startVec
                                            sizeTensor:sizeVec
                                           squeezeMask:0
                                                  name:nil];
                MPSGraphTensor* row = [g reshapeTensor:slice withShape:@[ @1, @(zSize) ] name:nil];
                [batchRows addObject:row];
            }

            if (batchRows.count > 0) {
                MPSGraphTensor* result = [g concatTensors:batchRows dimension:0 name:nil];
                NSArray<NSNumber*>* outputShape = GetOutputShape(op);
                if (outputShape) {
                    result = [g reshapeTensor:result withShape:outputShape name:nil];
                }
                return result;
            }
        }
    }

    // General gatherND lowering for patterns where index vectors are carried in
    // the last indices dimension, which covers advanced indexing used by Haliax.
    if (indexVectorDim == (int64_t)indicesRank - 1) {
        MPSGraphTensor* ndIndices = EnsureInt32(g, startIndices);
        return [g gatherNDWithUpdatesTensor:operand indicesTensor:ndIndices batchDimensions:0 name:nil];
    }

    // For now, log unsupported patterns
    MPS_LOG_ERROR("Unsupported gather pattern - offset_dims size: %lu, collapsed_slice_dims "
                  "size: %lu, start_index_map size: %lu, index_vector_dim: %lld\n",
                  (unsigned long)offsetDims.size(), (unsigned long)collapsedSliceDims.size(),
                  (unsigned long)startIndexMap.size(), indexVectorDim);
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.gather", Handle_gather);

// Scatter - update tensor at specified indices
// This handles the common pattern used by gather gradients
static MPSGraphTensor* Handle_scatter(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op);
    if (!scatterOp) {
        MPS_LOG_ERROR("Expected ScatterOp\n");
        return nullptr;
    }

    // Get inputs (may be variadic, but we handle single input case)
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    MPSGraphTensor* scatterIndices = GetInputTensor(values, op, 1);
    MPSGraphTensor* updates = GetInputTensor(values, op, 2);
    if (!input || !scatterIndices || !updates)
        return nullptr;

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    NSArray<NSNumber*>* indicesShape = scatterIndices.shape;
    NSUInteger indicesRank = indicesShape.count;
    // Handle dynamic-update-slice-like 1D scatter:
    // input: [N], indices: [1], updates: [M], update_window_dims=[0]
    // This pattern appears in jnp.unique internals and is not representable
    // via scatterWithDataTensor directly.
    if (input.shape.count == 1 && updates.shape.count == 1 && indicesRank == 1 &&
        [indicesShape[0] integerValue] == 1 && scatterDimsToOperandDims.size() == 1 &&
        scatterDimsToOperandDims[0] == 0 && updateWindowDims.size() == 1 &&
        updateWindowDims[0] == 0 && insertedWindowDims.empty()) {
        int64_t inputLen = [input.shape[0] longLongValue];
        int64_t updateLen = [updates.shape[0] longLongValue];

        if (@available(macOS 15.2, *)) {
            MPSGraphTensor* start = EnsureInt32(g, scatterIndices);
            MPSGraphTensor* zero = [g constantWithScalar:0 shape:@[@1] dataType:start.dataType];
            MPSGraphTensor* updateLenTensor =
                [g constantWithScalar:updateLen shape:@[@1] dataType:start.dataType];
            MPSGraphTensor* inputLenTensor =
                [g constantWithScalar:inputLen shape:@[@1] dataType:start.dataType];

            MPSGraphTensor* prefix =
                [g sliceTensor:input startTensor:zero sizeTensor:start squeezeMask:0 name:nil];
            MPSGraphTensor* suffixStart =
                [g additionWithPrimaryTensor:start secondaryTensor:updateLenTensor name:nil];
            MPSGraphTensor* suffixSize =
                [g subtractionWithPrimaryTensor:inputLenTensor secondaryTensor:suffixStart name:nil];
            MPSGraphTensor* suffix = [g sliceTensor:input
                                        startTensor:suffixStart
                                         sizeTensor:suffixSize
                                        squeezeMask:0
                                               name:nil];
            return [g concatTensors:@[ prefix, updates, suffix ] dimension:0 name:nil];
        } else {
            auto maybeStart = TryGetConstScalarInt(op->getOperand(1));
            if (maybeStart) {
                int64_t start = *maybeStart;
                if (start >= 0 && start <= inputLen && updateLen >= 0 && start + updateLen <= inputLen) {
                    NSMutableArray<MPSGraphTensor*>* pieces = [NSMutableArray array];
                    if (start > 0) {
                        MPSGraphTensor* prefix =
                            [g sliceTensor:input dimension:0 start:0 length:start name:nil];
                        [pieces addObject:prefix];
                    }
                    [pieces addObject:updates];
                    int64_t suffixStart = start + updateLen;
                    int64_t suffixLen = inputLen - suffixStart;
                    if (suffixLen > 0) {
                        MPSGraphTensor* suffix = [g sliceTensor:input
                                                       dimension:0
                                                           start:suffixStart
                                                          length:suffixLen
                                                            name:nil];
                        [pieces addObject:suffix];
                    }
                    if (pieces.count == 1) {
                        return pieces[0];
                    }
                    return [g concatTensors:pieces dimension:0 name:nil];
                }
            }
        }
    }

    // Pointwise scatter with explicit full-rank coordinates.
    // Examples:
    // - input[B,S,V], indices[B,S,3], updates[B,S]
    // - input[B,V],   indices[B,2],   updates[B]
    if (updateWindowDims.empty() && indexVectorDim == (int64_t)indicesRank - 1 &&
        insertedWindowDims.size() == input.shape.count &&
        scatterDimsToOperandDims.size() == input.shape.count &&
        [indicesShape[indicesRank - 1] integerValue] == (NSInteger)input.shape.count &&
        updates.shape.count + 1 == indicesRank) {
        int64_t flatCount = 1;
        for (NSUInteger i = 0; i + 1 < indicesRank; ++i) {
            flatCount *= [indicesShape[i] longLongValue];
        }
        MPSGraphTensor* flatIndices = [g reshapeTensor:scatterIndices
                                             withShape:@[ @(flatCount), @((NSInteger)input.shape.count) ]
                                                  name:nil];
        flatIndices = EnsureInt32(g, flatIndices);
        MPSGraphTensor* flatUpdates =
            [g reshapeTensor:updates withShape:@[ @(flatCount) ] name:nil];

        MPSGraphScatterMode mode = MPSGraphScatterModeSet;
        auto& updateRegion = scatterOp.getUpdateComputation();
        if (!updateRegion.empty()) {
            auto& block = updateRegion.front();
            for (auto& innerOp : block) {
                if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                    mode = MPSGraphScatterModeAdd;
                    break;
                } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                    mode = MPSGraphScatterModeSub;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                    mode = MPSGraphScatterModeMul;
                    break;
                } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                    mode = MPSGraphScatterModeDiv;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                    mode = MPSGraphScatterModeMax;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                    mode = MPSGraphScatterModeMin;
                    break;
                }
            }
        }

        MPSGraphTensor* scatterInput = input;
        MPSGraphTensor* scatterUpdates = flatUpdates;
        bool castBackToBool = (input.dataType == MPSDataTypeBool);
        if (castBackToBool) {
            scatterInput = [g castTensor:input toType:MPSDataTypeInt32 name:nil];
            scatterUpdates = [g castTensor:flatUpdates toType:MPSDataTypeInt32 name:nil];
        }

        MPSGraphTensor* scattered = [g scatterNDWithDataTensor:scatterInput
                                                  updatesTensor:scatterUpdates
                                                  indicesTensor:flatIndices
                                               batchDimensions:0
                                                           mode:mode
                                                           name:nil];
        if (castBackToBool) {
            scattered = [g castTensor:scattered toType:MPSDataTypeBool name:nil];
        }
        return scattered;
    }

    // Handle common embedding gradient pattern (reverse of gather):
    // input: [num_embeddings, embedding_dim] - zeros initially
    // indices: [batch..., 1] where last dim is index vector
    // updates: [batch..., embedding_dim] - gradients to scatter
    // Result: accumulate updates into input at specified indices

    // Check for the common pattern where:
    // - index_vector_dim is the last dimension of indices
    // - indices has size 1 in that dimension
    // - we're scattering along a single dimension
    if (indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && scatterDimsToOperandDims.size() == 1 &&
        (insertedWindowDims.empty() ||
         (insertedWindowDims.size() == 1 && insertedWindowDims[0] == scatterDimsToOperandDims[0]))) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        // Squeeze the index vector dimension from indices
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        // If squeezing produces a scalar, keep as [1] so MPS has a valid rank for the axis
        if (squeezedShape.count == 0)
            [squeezedShape addObject:@1];

        MPSGraphTensor* squeezedIndices = [g reshapeTensor:scatterIndices
                                                 withShape:squeezedShape
                                                      name:nil];

        // Cast indices to int32 if needed
        squeezedIndices = EnsureInt32(g, squeezedIndices);

        // Determine the scatter mode based on the update computation.
        // Default to Set (plain assignment); arithmetic ops override below.
        MPSGraphScatterMode mode = MPSGraphScatterModeSet;

        auto& updateRegion = scatterOp.getUpdateComputation();
        if (!updateRegion.empty()) {
            auto& block = updateRegion.front();
            for (auto& innerOp : block) {
                if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                    mode = MPSGraphScatterModeAdd;
                    break;
                } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                    mode = MPSGraphScatterModeSub;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                    mode = MPSGraphScatterModeMul;
                    break;
                } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                    mode = MPSGraphScatterModeDiv;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                    mode = MPSGraphScatterModeMax;
                    break;
                } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                    mode = MPSGraphScatterModeMin;
                    break;
                }
            }
        }

        // Batched single-axis updates into a rank-2 tensor (e.g. vmapped dynamic_update_slice)
        // are not representable via scatterWithDataTensor directly because each batch element has
        // a different index along the scatter axis. Lower these as scatterND with explicit
        // [batch, index] coordinates.
        bool batchedScalarUpdate =
            (updates.shape.count == 1 && [updates.shape[0] integerValue] == [indicesShape[0] integerValue]) ||
            (updates.shape.count == 2 && [updates.shape[0] integerValue] == [indicesShape[0] integerValue] &&
             [updates.shape[1] integerValue] == 1);

        if (insertedWindowDims.empty() && input.shape.count == 2 && indicesRank == 2 &&
            [indicesShape[indicesRank - 1] integerValue] == 1 && batchedScalarUpdate) {
            NSNumber* batch = indicesShape[0];
            NSArray<NSNumber*>* prefixShape = @[ batch ];

            MPSGraphTensor* squeezedIndices =
                [g reshapeTensor:scatterIndices withShape:prefixShape name:nil];
            squeezedIndices = EnsureInt32(g, squeezedIndices);

            MPSGraphTensor* scalarUpdates = updates;
            if (updates.shape.count == 2 && [updates.shape[1] integerValue] == 1) {
                scalarUpdates = [g reshapeTensor:updates withShape:prefixShape name:nil];
            }

            MPSGraphTensor* batchCoords =
                [g coordinateAlongAxis:0 withShape:prefixShape name:nil];
            batchCoords = EnsureInt32(g, batchCoords);

            NSMutableArray<MPSGraphTensor*>* coordTensors = [NSMutableArray arrayWithCapacity:2];
            if (scatterAxis == 0) {
                [coordTensors addObject:squeezedIndices];
                [coordTensors addObject:batchCoords];
            } else if (scatterAxis == 1) {
                [coordTensors addObject:batchCoords];
                [coordTensors addObject:squeezedIndices];
            } else {
                MPS_LOG_ERROR("Unsupported scatter axis for rank-2 batched scatter: %lld\n",
                              scatterAxis);
                return nullptr;
            }

            MPSGraphTensor* ndIndices = [g stackTensors:coordTensors axis:1 name:nil];
            MPSGraphTensor* scatterInput = input;
            MPSGraphTensor* scatterUpdates = scalarUpdates;
            bool castBackToBool = (input.dataType == MPSDataTypeBool);
            if (castBackToBool) {
                scatterInput = [g castTensor:input toType:MPSDataTypeInt32 name:nil];
                scatterUpdates = [g castTensor:scalarUpdates toType:MPSDataTypeInt32 name:nil];
            }

            MPSGraphTensor* scattered = [g scatterNDWithDataTensor:scatterInput
                                                      updatesTensor:scatterUpdates
                                                      indicesTensor:ndIndices
                                                   batchDimensions:0
                                                               mode:mode
                                                               name:nil];
            if (castBackToBool) {
                scattered = [g castTensor:scattered toType:MPSDataTypeBool name:nil];
            }
            return scattered;
        }

        // Ensure updates is at least rank 1 (MPS doesn't support scalar updates)
        if (updates.shape.count == 0)
            updates = [g reshapeTensor:updates withShape:@[@1] name:nil];

        // Use scatterWithDataTensor to scatter updates into input
        MPSGraphTensor* scatterInput = input;
        MPSGraphTensor* scatterUpdates = updates;
        bool castBackToBool = (input.dataType == MPSDataTypeBool);
        if (castBackToBool) {
            scatterInput = [g castTensor:input toType:MPSDataTypeInt32 name:nil];
            scatterUpdates = [g castTensor:updates toType:MPSDataTypeInt32 name:nil];
        }

        MPSGraphTensor* scattered = [g scatterWithDataTensor:scatterInput
                                               updatesTensor:scatterUpdates
                                               indicesTensor:squeezedIndices
                                                        axis:(NSUInteger)scatterAxis
                                                        mode:mode
                                                        name:nil];
        if (castBackToBool) {
            scattered = [g castTensor:scattered toType:MPSDataTypeBool name:nil];
        }
        return scattered;
    }

    MPS_LOG_ERROR("Unsupported scatter pattern - update_window_dims size: %lu, "
                  "inserted_window_dims size: %lu, scatter_dims_to_operand_dims size: %lu, "
                  "index_vector_dim: %lld\n",
                  (unsigned long)updateWindowDims.size(), (unsigned long)insertedWindowDims.size(),
                  (unsigned long)scatterDimsToOperandDims.size(), indexVectorDim);
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.scatter", Handle_scatter);

// Reverse - reverse elements along specified dimensions
static MPSGraphTensor* Handle_reverse(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(op);
    if (!reverseOp) {
        MPS_LOG_ERROR("Expected ReverseOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input)
        return nullptr;

    auto dimensions = reverseOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    return [g reverseTensor:input axes:axes name:nil];
}
REGISTER_MPS_OP("stablehlo.reverse", Handle_reverse);

}  // namespace jax_mps
