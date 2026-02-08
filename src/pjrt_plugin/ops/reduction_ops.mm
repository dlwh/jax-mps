// Reduction operations: reduce (sum, product, max, min, and, or)

#import "pjrt_plugin/ops/registry.h"
#include <algorithm>

namespace jax_mps {

// Helper to identify the reduction operation type from the region body
// Returns the operation name if it's a simple binary reduction, empty string otherwise
static std::string GetReductionOpType(mlir::Region& body) {
    if (body.empty())
        return "";

    mlir::Block& block = body.front();

    // The reduction body should have exactly one operation (plus terminator)
    // that is the reduction function
    for (mlir::Operation& op : block) {
        std::string opName = op.getName().getStringRef().str();

        // Skip the terminator (stablehlo.return)
        if (opName == "stablehlo.return")
            continue;

        // Return the first binary operation we find
        return opName;
    }

    return "";
}

// Reduce operation - identifies reduction type and maps to MPS reduction
static MPSGraphTensor* Handle_reduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        MPS_LOG_ERROR(" Expected ReduceOp\n");
        return nullptr;
    }

    // Get the input tensor (first operand)
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        MPS_LOG_ERROR(" reduce input tensor not found\n");
        return nullptr;
    }

    // Get reduction dimensions
    auto dimensions = reduceOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    // Identify the reduction operation from the body
    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    auto reduceOnce = [&](MPSGraphTensor* in, NSArray<NSNumber*>* reduceAxes) -> MPSGraphTensor* {
        if (reductionType == "stablehlo.add")
            return [g reductionSumWithTensor:in axes:reduceAxes name:nil];
        if (reductionType == "stablehlo.multiply")
            return [g reductionProductWithTensor:in axes:reduceAxes name:nil];
        if (reductionType == "stablehlo.maximum")
            return [g reductionMaximumWithTensor:in axes:reduceAxes name:nil];
        if (reductionType == "stablehlo.minimum")
            return [g reductionMinimumWithTensor:in axes:reduceAxes name:nil];
        if (reductionType == "stablehlo.and")
            return [g reductionAndWithTensor:in axes:reduceAxes name:nil];
        if (reductionType == "stablehlo.or")
            return [g reductionOrWithTensor:in axes:reduceAxes name:nil];
        return nullptr;
    };

    // MPS reduction kernels expect reduction axes to be within the "minor 4"
    // dimensions. To handle higher-rank tensors robustly, reduce one axis at a
    // time from highest to lowest axis, collapsing rank between steps.
    std::vector<int64_t> reduceDims;
    reduceDims.reserve(dimensions.size());
    for (int64_t dim : dimensions)
        reduceDims.push_back(dim);
    std::sort(reduceDims.begin(), reduceDims.end(), std::greater<int64_t>());

    MPSGraphTensor* result = input;
    for (int64_t axis : reduceDims) {
        NSArray<NSNumber*>* singleAxis = @[ @(axis) ];
        result = reduceOnce(result, singleAxis);
        if (!result) {
            MPS_LOG_ERROR(" Unsupported reduction type: %s\n", reductionType.c_str());
            return nullptr;
        }

        NSArray<NSNumber*>* currentShape = result.shape;
        if (!currentShape || axis < 0 || axis >= (int64_t)currentShape.count) {
            MPS_LOG_ERROR(" Invalid intermediate shape during reduction\n");
            return nullptr;
        }

        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < currentShape.count; i++) {
            if ((int64_t)i != axis)
                [squeezedShape addObject:currentShape[i]];
        }
        result = [g reshapeTensor:result withShape:squeezedShape name:nil];
    }

    // MPS Graph reduction keeps dimensions (with size 1), but StableHLO reduce removes them
    // Reshape to the expected output shape from the MLIR operation
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    if (outputShape && result) {
        result = [g reshapeTensor:result withShape:outputShape name:nil];
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.reduce", Handle_reduce);

static MPSGraphTensor* Handle_reduce_window(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reduceWindowOp = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(op);
    if (!reduceWindowOp) {
        MPS_LOG_ERROR(" Expected ReduceWindowOp\n");
        return nullptr;
    }

    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        MPS_LOG_ERROR(" reduce_window input tensor not found\n");
        return nullptr;
    }

    std::string reductionType = GetReductionOpType(reduceWindowOp.getBody());
    auto windowDims = reduceWindowOp.getWindowDimensions();
    auto maybeWindowStrides = reduceWindowOp.getWindowStrides();
    auto maybeBaseDilations = reduceWindowOp.getBaseDilations();
    auto maybeWindowDilations = reduceWindowOp.getWindowDilations();
    auto maybePadding = reduceWindowOp.getPadding();
    llvm::ArrayRef<int64_t> windowStrides =
        maybeWindowStrides ? *maybeWindowStrides : llvm::ArrayRef<int64_t>();
    llvm::ArrayRef<int64_t> baseDilations =
        maybeBaseDilations ? *maybeBaseDilations : llvm::ArrayRef<int64_t>();
    llvm::ArrayRef<int64_t> windowDilations =
        maybeWindowDilations ? *maybeWindowDilations : llvm::ArrayRef<int64_t>();

    NSArray<NSNumber*>* inputShape = input.shape;
    if (!inputShape) {
        MPS_LOG_ERROR(" reduce_window input shape missing\n");
        return nullptr;
    }

    const int64_t rank = (int64_t)inputShape.count;
    if ((int64_t)windowDims.size() != rank) {
        MPS_LOG_ERROR(" reduce_window rank/attribute mismatch (rank=%lld, window_dims=%lld)\n",
                      rank, (long long)windowDims.size());
        return nullptr;
    }
    mlir::DenseIntElementsAttr padding;
    if (maybePadding) {
        padding = *maybePadding;
        if (padding.getType().getRank() != 2 || padding.getType().getShape()[0] != rank ||
            padding.getType().getShape()[1] != 2) {
            MPS_LOG_ERROR(" reduce_window padding shape mismatch\n");
            return nullptr;
        }
    }

    if (!windowStrides.empty() && (int64_t)windowStrides.size() != rank) {
        MPS_LOG_ERROR(" reduce_window strides rank mismatch\n");
        return nullptr;
    }
    if (!baseDilations.empty() && (int64_t)baseDilations.size() != rank) {
        MPS_LOG_ERROR(" reduce_window base_dilations rank mismatch\n");
        return nullptr;
    }
    if (!windowDilations.empty() && (int64_t)windowDilations.size() != rank) {
        MPS_LOG_ERROR(" reduce_window window_dilations rank mismatch\n");
        return nullptr;
    }

    // Pooling-style reduce_window for add/max/min. Lower each active axis independently
    // by reshaping to a 1D pooling problem. This supports rank-agnostic rectangular windows.
    if (reductionType == "stablehlo.add" || reductionType == "stablehlo.maximum" ||
        reductionType == "stablehlo.minimum") {
        for (int64_t i = 0; i < rank; ++i) {
            int64_t baseDil = baseDilations.empty() ? 1 : baseDilations[(size_t)i];
            if (baseDil != 1) {
                MPS_LOG_ERROR(" reduce_window pooling with base dilation is unsupported\n");
                return nullptr;
            }
        }

        auto productOfDims = [](NSArray<NSNumber*>* shape, const std::vector<int64_t>& dims) -> int64_t {
            int64_t p = 1;
            for (int64_t d : dims) {
                p *= [shape[(NSUInteger)d] longLongValue];
            }
            return p;
        };

        MPSGraphTensor* pooled = input;
        for (int64_t axis = 0; axis < rank; ++axis) {
            int64_t wd = windowDims[(size_t)axis];
            int64_t stride = windowStrides.empty() ? 1 : windowStrides[(size_t)axis];
            int64_t winDil = windowDilations.empty() ? 1 : windowDilations[(size_t)axis];
            int64_t padLow = maybePadding ? padding.getValues<int64_t>()[{(uint64_t)axis, (uint64_t)0}] : 0;
            int64_t padHigh = maybePadding ? padding.getValues<int64_t>()[{(uint64_t)axis, (uint64_t)1}] : 0;
            bool isIdentityAxis =
                wd == 1 && stride == 1 && winDil == 1 && padLow == 0 && padHigh == 0;
            if (isIdentityAxis) {
                continue;
            }

            NSArray<NSNumber*>* curShape = pooled.shape;
            if (!curShape || curShape.count != (NSUInteger)rank) {
                MPS_LOG_ERROR(" reduce_window pooling shape/rank mismatch during lowering\n");
                return nullptr;
            }

            std::vector<int64_t> otherDims;
            otherDims.reserve((size_t)rank - 1);
            for (int64_t d = 0; d < rank; ++d) {
                if (d != axis)
                    otherDims.push_back(d);
            }

            NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
            for (int64_t d : otherDims)
                [perm addObject:@(d)];
            [perm addObject:@(axis)];

            MPSGraphTensor* transposed = pooled;
            bool isIdentityPerm = true;
            for (NSUInteger i = 0; i < perm.count; ++i) {
                if ([perm[i] integerValue] != (NSInteger)i) {
                    isIdentityPerm = false;
                    break;
                }
            }
            if (!isIdentityPerm) {
                transposed = [g transposeTensor:pooled permutation:perm name:nil];
            }

            int64_t batch = productOfDims(curShape, otherDims);
            int64_t axisLen = [curShape[(NSUInteger)axis] longLongValue];
            MPSGraphTensor* pooledInput =
                [g reshapeTensor:transposed withShape:@[ @(batch), @1, @(axisLen), @1 ] name:nil];

            MPSGraphTensor* poolSource = pooledInput;
            if (reductionType == "stablehlo.minimum") {
                poolSource = [g negativeWithTensor:pooledInput name:nil];
            }

            MPSDataType origType = poolSource.dataType;
            bool castForAvg = false;
            if (reductionType == "stablehlo.add" &&
                !(origType == MPSDataTypeFloat16 || origType == MPSDataTypeFloat32 ||
                  origType == MPSDataTypeBFloat16)) {
                poolSource = [g castTensor:poolSource toType:MPSDataTypeFloat32 name:nil];
                castForAvg = true;
            }

            MPSGraphPooling2DOpDescriptor* poolDesc = [MPSGraphPooling2DOpDescriptor
                descriptorWithKernelWidth:(NSUInteger)wd
                              kernelHeight:1
                                 strideInX:(NSUInteger)stride
                                 strideInY:1
                           dilationRateInX:(NSUInteger)winDil
                           dilationRateInY:1
                               paddingLeft:(NSUInteger)padLow
                              paddingRight:(NSUInteger)padHigh
                                paddingTop:0
                             paddingBottom:0
                              paddingStyle:MPSGraphPaddingStyleExplicit
                                dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
            MPSGraphTensor* axisPooled = nullptr;
            if (reductionType == "stablehlo.add") {
                poolDesc.includeZeroPadToAverage = YES;
                axisPooled = [g avgPooling2DWithSourceTensor:poolSource descriptor:poolDesc name:nil];
                MPSGraphTensor* scale =
                    [g constantWithScalar:wd shape:@[] dataType:axisPooled.dataType];
                axisPooled = [g multiplicationWithPrimaryTensor:axisPooled secondaryTensor:scale name:nil];
                if (castForAvg) {
                    axisPooled = [g castTensor:axisPooled toType:origType name:nil];
                }
            } else {
                axisPooled = [g maxPooling2DWithSourceTensor:poolSource descriptor:poolDesc name:nil];
                if (reductionType == "stablehlo.minimum") {
                    axisPooled = [g negativeWithTensor:axisPooled name:nil];
                }
            }
            if (!axisPooled) {
                MPS_LOG_ERROR(" reduce_window pooling lowering failed on axis %lld\n", axis);
                return nullptr;
            }

            NSArray<NSNumber*>* axisShape = axisPooled.shape;
            if (!axisShape || axisShape.count != 4) {
                MPS_LOG_ERROR(" reduce_window pooled axis shape mismatch\n");
                return nullptr;
            }
            int64_t outLen = [axisShape[2] longLongValue];

            NSMutableArray<NSNumber*>* backShape = [NSMutableArray array];
            for (int64_t d : otherDims) {
                [backShape addObject:curShape[(NSUInteger)d]];
            }
            [backShape addObject:@(outLen)];
            MPSGraphTensor* restored = [g reshapeTensor:axisPooled withShape:backShape name:nil];

            NSMutableArray<NSNumber*>* invPerm = [NSMutableArray array];
            for (int64_t i = 0; i < rank; ++i)
                [invPerm addObject:@0];
            for (NSUInteger i = 0; i < perm.count; ++i) {
                NSInteger p = [perm[i] integerValue];
                invPerm[(NSUInteger)p] = @(i);
            }

            if (!isIdentityPerm) {
                restored = [g transposeTensor:restored permutation:invPerm name:nil];
            }
            pooled = restored;
        }

        NSArray<NSNumber*>* outputShape = GetOutputShape(op);
        if (outputShape && pooled) {
            pooled = [g reshapeTensor:pooled withShape:outputShape name:nil];
        }
        return pooled;
    }

    // Support the canonical cumulative lowering:
    // reduce_window with one active axis and
    // - stride=1
    // - base/window dilation=1
    // - padding low = window_dim - 1, high = 0 on active axis
    // - window_dim=1 and zero padding on all other axes
    int64_t cumulativeAxis = -1;
    for (int64_t i = 0; i < rank; ++i) {
        int64_t wd = windowDims[(size_t)i];
        int64_t stride = windowStrides.empty() ? 1 : windowStrides[(size_t)i];
        int64_t baseDil = baseDilations.empty() ? 1 : baseDilations[(size_t)i];
        int64_t winDil = windowDilations.empty() ? 1 : windowDilations[(size_t)i];
        int64_t padLow = maybePadding ? padding.getValues<int64_t>()[{(uint64_t)i, (uint64_t)0}] : 0;
        int64_t padHigh = maybePadding ? padding.getValues<int64_t>()[{(uint64_t)i, (uint64_t)1}] : 0;
        int64_t dimSize = [inputShape[(NSUInteger)i] longLongValue];

        if (stride != 1 || baseDil != 1 || winDil != 1) {
            cumulativeAxis = -1;
            break;
        }

        if (wd == 1 && padLow == 0 && padHigh == 0) {
            continue;
        }

        if (wd == dimSize && padLow == wd - 1 && padHigh == 0 && cumulativeAxis < 0) {
            cumulativeAxis = i;
            continue;
        }

        cumulativeAxis = -1;
        break;
    }

    if (cumulativeAxis < 0) {
        MPS_LOG_ERROR(" Unsupported reduce_window configuration\n");
        return nullptr;
    }

    MPSGraphTensor* result = nullptr;
    if (reductionType == "stablehlo.add") {
        result = [g cumulativeSumWithTensor:input
                                       axis:(NSInteger)cumulativeAxis
                                  exclusive:NO
                                    reverse:NO
                                       name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        result = [g cumulativeProductWithTensor:input
                                           axis:(NSInteger)cumulativeAxis
                                      exclusive:NO
                                        reverse:NO
                                           name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        result = [g cumulativeMaximumWithTensor:input
                                           axis:(NSInteger)cumulativeAxis
                                      exclusive:NO
                                        reverse:NO
                                           name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        result = [g cumulativeMinimumWithTensor:input
                                           axis:(NSInteger)cumulativeAxis
                                      exclusive:NO
                                        reverse:NO
                                           name:nil];
    } else {
        MPS_LOG_ERROR(" Unsupported reduce_window reduction type: %s\n", reductionType.c_str());
        return nullptr;
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    if (outputShape && result) {
        result = [g reshapeTensor:result withShape:outputShape name:nil];
    }

    return result;
}
REGISTER_MPS_OP("stablehlo.reduce_window", Handle_reduce_window);

enum class SelectScatterKind { kUnknown, kMax, kMin };

static bool TryGetI64ListAttr(mlir::Operation* op, llvm::StringRef name,
                              std::vector<int64_t>& out) {
    if (auto dense = op->getAttrOfType<mlir::DenseI64ArrayAttr>(name)) {
        out.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
        return true;
    }
    if (auto arr = op->getAttrOfType<mlir::ArrayAttr>(name)) {
        out.clear();
        out.reserve(arr.size());
        for (mlir::Attribute attr : arr) {
            auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
            if (!intAttr) {
                return false;
            }
            out.push_back(intAttr.getInt());
        }
        return true;
    }
    return false;
}

static SelectScatterKind GetSelectScatterKind(mlir::Region& selectRegion) {
    if (selectRegion.empty()) {
        return SelectScatterKind::kUnknown;
    }
    for (mlir::Operation& nestedOp : selectRegion.front()) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }
        auto dir = compareOp.getComparisonDirection();
        if (dir == mlir::stablehlo::ComparisonDirection::GE ||
            dir == mlir::stablehlo::ComparisonDirection::GT) {
            return SelectScatterKind::kMax;
        }
        if (dir == mlir::stablehlo::ComparisonDirection::LE ||
            dir == mlir::stablehlo::ComparisonDirection::LT) {
            return SelectScatterKind::kMin;
        }
    }
    return SelectScatterKind::kUnknown;
}

static MPSGraphTensor* Handle_select_and_scatter(MPSGraph* g, mlir::Operation* op,
                                                 ValueMap& values) {
    auto sasOp = mlir::dyn_cast<mlir::stablehlo::SelectAndScatterOp>(op);
    if (!sasOp) {
        MPS_LOG_ERROR(" Expected SelectAndScatterOp\n");
        return nullptr;
    }

    MPSGraphTensor* operand = GetInputTensor(values, op, 0);
    MPSGraphTensor* source = GetInputTensor(values, op, 1);
    if (!operand || !source) {
        MPS_LOG_ERROR(" select_and_scatter operand/source tensor not found\n");
        return nullptr;
    }

    std::string scatterType = GetReductionOpType(sasOp.getScatter());
    if (scatterType != "stablehlo.add") {
        MPS_LOG_ERROR(" select_and_scatter only supports add scatter, got %s\n", scatterType.c_str());
        return nullptr;
    }
    SelectScatterKind kind = GetSelectScatterKind(sasOp.getSelect());
    if (kind == SelectScatterKind::kUnknown) {
        MPS_LOG_ERROR(" select_and_scatter only supports compare-based max/min select\n");
        return nullptr;
    }

    std::vector<int64_t> windowDimsStorage;
    std::vector<int64_t> windowStridesStorage;
    mlir::DenseIntElementsAttr padding = op->getAttrOfType<mlir::DenseIntElementsAttr>("padding");

    if (!TryGetI64ListAttr(op, "window_dimensions", windowDimsStorage)) {
        if (auto dimsAttr = sasOp.getWindowDimensionsAttr()) {
            windowDimsStorage.assign(dimsAttr.asArrayRef().begin(), dimsAttr.asArrayRef().end());
        }
    }
    if (!TryGetI64ListAttr(op, "window_strides", windowStridesStorage)) {
        if (auto stridesAttr = sasOp.getWindowStridesAttr()) {
            windowStridesStorage.assign(stridesAttr.asArrayRef().begin(), stridesAttr.asArrayRef().end());
        }
    }
    if (!padding) {
        if (auto paddingAttr = sasOp.getPaddingAttr()) {
            padding = paddingAttr;
        }
    }
    if (windowDimsStorage.empty()) {
        MPS_LOG_ERROR(" select_and_scatter requires window_dimensions\n");
        return nullptr;
    }

    NSArray<NSNumber*>* operandShape = operand.shape;
    NSArray<NSNumber*>* sourceShape = source.shape;
    if (!operandShape) {
        MPS_LOG_ERROR(" select_and_scatter operand shape missing\n");
        return nullptr;
    }
    if (!sourceShape) {
        MPS_LOG_ERROR(" select_and_scatter source shape missing\n");
        return nullptr;
    }
    const int64_t rank = (int64_t)operandShape.count;
    if ((int64_t)sourceShape.count != rank) {
        MPS_LOG_ERROR(" select_and_scatter source/operand rank mismatch\n");
        return nullptr;
    }
    if (windowStridesStorage.empty()) {
        windowStridesStorage.assign((size_t)rank, 1);
    }
    llvm::ArrayRef<int64_t> windowDims(windowDimsStorage);
    llvm::ArrayRef<int64_t> windowStrides(windowStridesStorage);
    if ((int64_t)windowDims.size() != rank || (int64_t)windowStrides.size() != rank) {
        MPS_LOG_ERROR(" select_and_scatter rank/attribute mismatch\n");
        return nullptr;
    }
    if (padding &&
        (padding.getType().getRank() != 2 || padding.getType().getShape()[0] != rank ||
         padding.getType().getShape()[1] != 2)) {
        MPS_LOG_ERROR(" select_and_scatter padding rank/shape mismatch\n");
        return nullptr;
    }

    auto padAt = [&](int64_t axis, int64_t loOrHi) -> int64_t {
        if (!padding) {
            return 0;
        }
        return padding.getValues<int64_t>()[{(uint64_t)axis, (uint64_t)loOrHi}];
    };

    std::vector<int64_t> activeAxes;
    std::vector<int64_t> inactiveAxes;
    activeAxes.reserve((size_t)rank);
    inactiveAxes.reserve((size_t)rank);
    for (int64_t i = 0; i < rank; ++i) {
        int64_t wd = windowDims[(size_t)i];
        int64_t ws = windowStrides[(size_t)i];
        int64_t padLow = padAt(i, 0);
        int64_t padHigh = padAt(i, 1);
        if (wd < 1 || ws < 1 || padLow < 0 || padHigh < 0) {
            MPS_LOG_ERROR(" select_and_scatter has invalid pooling attributes\n");
            return nullptr;
        }
        bool identityAxis = wd == 1 && ws == 1 && padLow == 0 && padHigh == 0;
        if (identityAxis) {
            inactiveAxes.push_back(i);
        } else {
            activeAxes.push_back(i);
        }
    }
    if (activeAxes.empty() || activeAxes.size() > 3) {
        MPS_LOG_ERROR(" select_and_scatter supports at most 3 active window axes\n");
        return nullptr;
    }
    if (inactiveAxes.empty()) {
        MPS_LOG_ERROR(" select_and_scatter requires at least one identity axis for batching\n");
        return nullptr;
    }

    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    for (int64_t axis : inactiveAxes)
        [perm addObject:@(axis)];
    for (int64_t axis : activeAxes)
        [perm addObject:@(axis)];

    bool isIdentityPerm = true;
    for (NSUInteger i = 0; i < perm.count; ++i) {
        if ([perm[i] integerValue] != (NSInteger)i) {
            isIdentityPerm = false;
            break;
        }
    }

    MPSGraphTensor* permutedOperand = operand;
    MPSGraphTensor* permutedSource = source;
    if (!isIdentityPerm) {
        permutedOperand = [g transposeTensor:operand permutation:perm name:nil];
        permutedSource = [g transposeTensor:source permutation:perm name:nil];
    }

    int64_t batch = 1;
    for (int64_t axis : inactiveAxes) {
        batch *= [operandShape[(NSUInteger)axis] longLongValue];
    }
    NSMutableArray<NSNumber*>* operand4DShape = [NSMutableArray arrayWithObject:@(batch)];
    NSMutableArray<NSNumber*>* source4DShape = [NSMutableArray arrayWithObject:@(batch)];
    NSMutableArray<NSNumber*>* kernelSizes = [NSMutableArray arrayWithObject:@1];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithObject:@1];
    NSMutableArray<NSNumber*>* dilations = [NSMutableArray arrayWithObject:@1];
    NSMutableArray<NSNumber*>* paddingValues = [NSMutableArray arrayWithObjects:@0, @0, nil];
    for (int64_t axis : activeAxes) {
        int64_t inDim = [operandShape[(NSUInteger)axis] longLongValue];
        int64_t outDim = [sourceShape[(NSUInteger)axis] longLongValue];
        [operand4DShape addObject:@(inDim)];
        [source4DShape addObject:@(outDim)];
        [kernelSizes addObject:@(windowDims[(size_t)axis])];
        [strides addObject:@(windowStrides[(size_t)axis])];
        [dilations addObject:@1];
        [paddingValues addObject:@(padAt(axis, 0))];
        [paddingValues addObject:@(padAt(axis, 1))];
    }
    while (operand4DShape.count < 4) {
        [operand4DShape addObject:@1];
        [source4DShape addObject:@1];
        [kernelSizes addObject:@1];
        [strides addObject:@1];
        [dilations addObject:@1];
        [paddingValues addObject:@0];
        [paddingValues addObject:@0];
    }

    MPSGraphTensor* operand4D = [g reshapeTensor:permutedOperand withShape:operand4DShape name:nil];
    MPSGraphTensor* source4D = [g reshapeTensor:permutedSource withShape:source4DShape name:nil];

    MPSGraphPooling4DOpDescriptor* poolDesc =
        [MPSGraphPooling4DOpDescriptor descriptorWithKernelSizes:kernelSizes
                                                         strides:strides
                                                   dilationRates:dilations
                                                   paddingValues:paddingValues
                                                    paddingStyle:MPSGraphPaddingStyleExplicit];
    if (!poolDesc) {
        MPS_LOG_ERROR(" select_and_scatter failed to create pooling descriptor\n");
        return nullptr;
    }

    MPSGraphTensor* srcForSelect = operand4D;
    if (kind == SelectScatterKind::kMin) {
        srcForSelect = [g negativeWithTensor:operand4D name:nil];
    }

    MPSGraphTensor* result = [g maxPooling4DGradientWithGradientTensor:source4D
                                                           sourceTensor:srcForSelect
                                                             descriptor:poolDesc
                                                                   name:nil];
    if (!result) {
        MPS_LOG_ERROR(" select_and_scatter pooling gradient lowering failed\n");
        return nullptr;
    }

    MPSDataType outType = GetResultMpsType(op);
    if (outType != MPSDataTypeInvalid && result.dataType != outType) {
        result = [g castTensor:result toType:outType name:nil];
    }

    NSMutableArray<NSNumber*>* permutedResultShape = [NSMutableArray array];
    for (int64_t axis : inactiveAxes)
        [permutedResultShape addObject:operandShape[(NSUInteger)axis]];
    for (int64_t axis : activeAxes)
        [permutedResultShape addObject:operandShape[(NSUInteger)axis]];
    result = [g reshapeTensor:result withShape:permutedResultShape name:nil];

    if (!isIdentityPerm) {
        NSMutableArray<NSNumber*>* invPerm = [NSMutableArray array];
        for (int64_t i = 0; i < rank; ++i)
            [invPerm addObject:@0];
        for (NSUInteger i = 0; i < perm.count; ++i) {
            NSInteger p = [perm[i] integerValue];
            invPerm[(NSUInteger)p] = @(i);
        }
        result = [g transposeTensor:result permutation:invPerm name:nil];
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    if (outputShape && result) {
        result = [g reshapeTensor:result withShape:outputShape name:nil];
    }
    return result;
}
REGISTER_MPS_OP("stablehlo.select_and_scatter", Handle_select_and_scatter);

// stablehlo.return is a terminator used inside regions (e.g., reduce body)
// It's handled implicitly by parent operations, not executed directly
static MPSGraphTensor* Handle_return(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    // This should never be called directly - it's handled by the parent operation
    // But we register it so it's not flagged as unsupported during module verification
    MPS_LOG_WARN("stablehlo.return should not be called directly\n");
    return nullptr;
}
REGISTER_MPS_OP("stablehlo.return", Handle_return);

}  // namespace jax_mps
