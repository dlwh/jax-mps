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

    // Pooling-style reduce_window for 1D/2D windows (max/min/sum), lowered via 2D pooling.
    if ((rank == 1 || rank == 2) &&
        (reductionType == "stablehlo.maximum" || reductionType == "stablehlo.minimum")) {
        int64_t kH = 1, kW = 1;
        int64_t sH = 1, sW = 1;
        int64_t dH = 1, dW = 1;
        int64_t pTop = 0, pBottom = 0, pLeft = 0, pRight = 0;

        for (int64_t i = 0; i < rank; ++i) {
            int64_t wd = windowDims[(size_t)i];
            int64_t stride = windowStrides.empty() ? 1 : windowStrides[(size_t)i];
            int64_t baseDil = baseDilations.empty() ? 1 : baseDilations[(size_t)i];
            int64_t winDil = windowDilations.empty() ? 1 : windowDilations[(size_t)i];
            int64_t padLow = maybePadding ? padding.getValues<int64_t>()[{(uint64_t)i, (uint64_t)0}] : 0;
            int64_t padHigh = maybePadding ? padding.getValues<int64_t>()[{(uint64_t)i, (uint64_t)1}] : 0;

            if (baseDil != 1) {
                MPS_LOG_ERROR(" reduce_window pooling with base dilation is unsupported\n");
                return nullptr;
            }

            if (rank == 1) {
                kW = wd;
                sW = stride;
                dW = winDil;
                pLeft = padLow;
                pRight = padHigh;
            } else if (i == 0) {
                kH = wd;
                sH = stride;
                dH = winDil;
                pTop = padLow;
                pBottom = padHigh;
            } else {
                kW = wd;
                sW = stride;
                dW = winDil;
                pLeft = padLow;
                pRight = padHigh;
            }
        }

        MPSGraphTensor* poolInput = input;
        if (rank == 1) {
            int64_t inputLen = [inputShape[0] longLongValue];
            poolInput = [g reshapeTensor:input withShape:@[ @1, @1, @(inputLen), @1 ] name:nil];
        } else {
            int64_t h = [inputShape[0] longLongValue];
            int64_t w = [inputShape[1] longLongValue];
            poolInput = [g reshapeTensor:input withShape:@[ @1, @(h), @(w), @1 ] name:nil];
        }

        MPSGraphTensor* poolSource = poolInput;
        if (reductionType == "stablehlo.minimum") {
            poolSource = [g negativeWithTensor:poolInput name:nil];
        }

        MPSGraphPooling2DOpDescriptor* poolDesc = [MPSGraphPooling2DOpDescriptor
            descriptorWithKernelWidth:(NSUInteger)kW
                          kernelHeight:(NSUInteger)kH
                             strideInX:(NSUInteger)sW
                             strideInY:(NSUInteger)sH
                       dilationRateInX:(NSUInteger)dW
                       dilationRateInY:(NSUInteger)dH
                           paddingLeft:(NSUInteger)pLeft
                          paddingRight:(NSUInteger)pRight
                            paddingTop:(NSUInteger)pTop
                         paddingBottom:(NSUInteger)pBottom
                          paddingStyle:MPSGraphPaddingStyleExplicit
                            dataLayout:MPSGraphTensorNamedDataLayoutNHWC];

        MPSGraphTensor* pooled = [g maxPooling2DWithSourceTensor:poolSource descriptor:poolDesc name:nil];
        if (reductionType == "stablehlo.minimum") {
            pooled = [g negativeWithTensor:pooled name:nil];
        }

        NSArray<NSNumber*>* outputShape = GetOutputShape(op);
        if (outputShape && pooled) {
            pooled = [g reshapeTensor:pooled withShape:outputShape name:nil];
        }
        return pooled;
    }

    // Batched 2D pooling: rank-3 input [B, H, W] with identity batch window.
    if (rank == 3 && (reductionType == "stablehlo.maximum" || reductionType == "stablehlo.minimum")) {
        int64_t w0 = windowDims[0];
        int64_t s0 = windowStrides.empty() ? 1 : windowStrides[0];
        int64_t b0 = baseDilations.empty() ? 1 : baseDilations[0];
        int64_t d0 = windowDilations.empty() ? 1 : windowDilations[0];
        int64_t pl0 = maybePadding ? padding.getValues<int64_t>()[{0, 0}] : 0;
        int64_t ph0 = maybePadding ? padding.getValues<int64_t>()[{0, 1}] : 0;
        if (w0 == 1 && s0 == 1 && b0 == 1 && d0 == 1 && pl0 == 0 && ph0 == 0) {
            int64_t kH = windowDims[1], kW = windowDims[2];
            int64_t sH = windowStrides.empty() ? 1 : windowStrides[1];
            int64_t sW = windowStrides.empty() ? 1 : windowStrides[2];
            int64_t dH = windowDilations.empty() ? 1 : windowDilations[1];
            int64_t dW = windowDilations.empty() ? 1 : windowDilations[2];
            int64_t bH = baseDilations.empty() ? 1 : baseDilations[1];
            int64_t bW = baseDilations.empty() ? 1 : baseDilations[2];
            int64_t pTop = maybePadding ? padding.getValues<int64_t>()[{1, 0}] : 0;
            int64_t pBottom = maybePadding ? padding.getValues<int64_t>()[{1, 1}] : 0;
            int64_t pLeft = maybePadding ? padding.getValues<int64_t>()[{2, 0}] : 0;
            int64_t pRight = maybePadding ? padding.getValues<int64_t>()[{2, 1}] : 0;
            if (bH != 1 || bW != 1) {
                MPS_LOG_ERROR(" reduce_window pooling with base dilation is unsupported\n");
                return nullptr;
            }

            int64_t B = [inputShape[0] longLongValue];
            int64_t H = [inputShape[1] longLongValue];
            int64_t W = [inputShape[2] longLongValue];
            MPSGraphTensor* poolInput =
                [g reshapeTensor:input withShape:@[ @(B), @(H), @(W), @1 ] name:nil];
            MPSGraphTensor* poolSource = poolInput;
            if (reductionType == "stablehlo.minimum") {
                poolSource = [g negativeWithTensor:poolInput name:nil];
            }

            MPSGraphPooling2DOpDescriptor* poolDesc = [MPSGraphPooling2DOpDescriptor
                descriptorWithKernelWidth:(NSUInteger)kW
                              kernelHeight:(NSUInteger)kH
                                 strideInX:(NSUInteger)sW
                                 strideInY:(NSUInteger)sH
                           dilationRateInX:(NSUInteger)dW
                           dilationRateInY:(NSUInteger)dH
                               paddingLeft:(NSUInteger)pLeft
                              paddingRight:(NSUInteger)pRight
                                paddingTop:(NSUInteger)pTop
                             paddingBottom:(NSUInteger)pBottom
                              paddingStyle:MPSGraphPaddingStyleExplicit
                                dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
            MPSGraphTensor* pooled = [g maxPooling2DWithSourceTensor:poolSource descriptor:poolDesc name:nil];
            if (reductionType == "stablehlo.minimum") {
                pooled = [g negativeWithTensor:pooled name:nil];
            }

            NSArray<NSNumber*>* outputShape = GetOutputShape(op);
            if (outputShape && pooled) {
                pooled = [g reshapeTensor:pooled withShape:outputShape name:nil];
            }
            return pooled;
        }
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
