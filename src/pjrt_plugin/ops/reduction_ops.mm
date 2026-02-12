// Reduction operations: reduce (sum, product, max, min, and, or, argmax, argmin)

#include <algorithm>

#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_mps {

namespace {

// Helper to identify the reduction operation type from the region body
// Returns the operation name if it's a simple binary reduction, empty string otherwise
std::string GetReductionOpType(mlir::Region& body) {
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

// For detecting argmax/argmin patterns in multi-result reduce
enum class ArgReduceKind { kUnknown, kMax, kMin };

bool IsBlockArg(mlir::Value value, mlir::Block& block, unsigned index) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == index;
}

ArgReduceKind detectArgReduceKind(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return ArgReduceKind::kUnknown;
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() < 4) {
        return ArgReduceKind::kUnknown;
    }

    for (mlir::Operation& nestedOp : body) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }

        bool forwardValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 0) && IsBlockArg(compareOp.getRhs(), body, 2);
        bool reversedValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 2) && IsBlockArg(compareOp.getRhs(), body, 0);
        if (!forwardValueCompare && !reversedValueCompare) {
            continue;
        }

        auto dir = compareOp.getComparisonDirection();
        bool lhsWins = false;
        if (dir == mlir::stablehlo::ComparisonDirection::GT ||
            dir == mlir::stablehlo::ComparisonDirection::GE) {
            lhsWins = true;
        } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                   dir == mlir::stablehlo::ComparisonDirection::LE) {
            lhsWins = false;
        } else {
            continue;
        }

        // If the compare operands are swapped, the max/min interpretation flips.
        if (reversedValueCompare) {
            lhsWins = !lhsWins;
        }
        return lhsWins ? ArgReduceKind::kMax : ArgReduceKind::kMin;
    }
    return ArgReduceKind::kUnknown;
}

}  // namespace

// Single-result reduce: sum, product, max, min, and, or
static ProcessResult HandleSingleResultReduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }

    // Get the input tensor (first operand)
    MPSGraphTensor* input = GetInputTensor(values, op, 0);
    if (!input) {
        return ProcessResult::Error("reduce: input tensor not found");
    }

    // Canonicalize to a 2D reduction so MPS reduction kernels always see minor axes.
    auto dimensions = reduceOp.getDimensions();
    const NSInteger rank = (NSInteger)input.shape.count;
    std::vector<int64_t> reducedDims(dimensions.begin(), dimensions.end());
    std::vector<int64_t> nonReducedDims;
    nonReducedDims.reserve((size_t)rank);
    for (NSInteger i = 0; i < rank; ++i) {
        if (std::find(reducedDims.begin(), reducedDims.end(), (int64_t)i) == reducedDims.end()) {
            nonReducedDims.push_back((int64_t)i);
        }
    }

    std::vector<int64_t> permutation;
    permutation.reserve((size_t)rank);
    permutation.insert(permutation.end(), nonReducedDims.begin(), nonReducedDims.end());
    permutation.insert(permutation.end(), reducedDims.begin(), reducedDims.end());

    auto isIdentityPermutation = [](const std::vector<int64_t>& perm) {
        for (size_t i = 0; i < perm.size(); ++i) {
            if ((int64_t)i != perm[i]) {
                return false;
            }
        }
        return true;
    };

    auto productOfDims = [](NSArray<NSNumber*>* shape,
                            const std::vector<int64_t>& dims) -> int64_t {
        int64_t prod = 1;
        for (int64_t d : dims) {
            prod *= [shape[(NSUInteger)d] longLongValue];
        }
        return prod;
    };

    MPSGraphTensor* canonicalInput = input;
    if (!isIdentityPermutation(permutation)) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:permutation.size()];
        for (int64_t d : permutation) {
            [perm addObject:@(d)];
        }
        canonicalInput = [g transposeTensor:canonicalInput permutation:perm name:nil];
    }

    int64_t nonReducedSize = productOfDims(input.shape, nonReducedDims);
    int64_t reducedSize = productOfDims(input.shape, reducedDims);
    canonicalInput = [g reshapeTensor:canonicalInput
                            withShape:@[@(nonReducedSize), @(reducedSize)]
                                 name:nil];

    // Identify the reduction operation from the body
    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    MPSGraphTensor* result = nullptr;
    if (reductionType == "stablehlo.add") {
        result = [g reductionSumWithTensor:canonicalInput axis:1 name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        result = [g reductionProductWithTensor:canonicalInput axis:1 name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        result = [g reductionMaximumWithTensor:canonicalInput axis:1 name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        result = [g reductionMinimumWithTensor:canonicalInput axis:1 name:nil];
    } else if (reductionType == "stablehlo.and") {
        result = [g reductionAndWithTensor:canonicalInput axis:1 name:nil];
    } else if (reductionType == "stablehlo.or") {
        result = [g reductionOrWithTensor:canonicalInput axis:1 name:nil];
    } else {
        return ProcessResult::Error("reduce: unsupported reduction type: " + reductionType);
    }

    // MPS Graph reduction keeps dimensions (with size 1), but StableHLO reduce removes them
    // Reshape to the expected output shape from the MLIR operation
    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    if (outputShape && result) {
        result = [g reshapeTensor:result withShape:outputShape name:nil];
    }

    return Result(values, op, result, "reduce");
}

// Multi-result reduce: argmax/argmin patterns
static ProcessResult HandleMultiResultReduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }
    if (op->getNumResults() != 2 || op->getNumOperands() < 2) {
        return ProcessResult::Error("reduce: unsupported multi-result shape");
    }

    MPSGraphTensor* valueInput = GetInputTensor(values, op, 0);
    if (!valueInput) {
        return ProcessResult::Error("reduce: value input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1) {
        return ProcessResult::Error("reduce: only single-axis multi-result reduce is supported");
    }
    NSInteger axis = (NSInteger)dimensions[0];

    ArgReduceKind kind = detectArgReduceKind(reduceOp);
    if (kind == ArgReduceKind::kUnknown) {
        return ProcessResult::Error("reduce: unsupported multi-result reduce body");
    }

    const NSInteger rank = (NSInteger)valueInput.shape.count;
    std::vector<int64_t> nonReducedDims;
    nonReducedDims.reserve((size_t)rank);
    for (NSInteger i = 0; i < rank; ++i) {
        if (i != axis) {
            nonReducedDims.push_back((int64_t)i);
        }
    }

    std::vector<int64_t> permutation = nonReducedDims;
    permutation.push_back((int64_t)axis);

    auto isIdentityPermutation = [](const std::vector<int64_t>& perm) {
        for (size_t i = 0; i < perm.size(); ++i) {
            if ((int64_t)i != perm[i]) {
                return false;
            }
        }
        return true;
    };
    auto productOfDims = [](NSArray<NSNumber*>* shape,
                            const std::vector<int64_t>& dims) -> int64_t {
        int64_t prod = 1;
        for (int64_t d : dims) {
            prod *= [shape[(NSUInteger)d] longLongValue];
        }
        return prod;
    };

    MPSGraphTensor* canonicalInput = valueInput;
    if (!isIdentityPermutation(permutation)) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:permutation.size()];
        for (int64_t d : permutation) {
            [perm addObject:@(d)];
        }
        canonicalInput = [g transposeTensor:canonicalInput permutation:perm name:nil];
    }

    int64_t nonReducedSize = productOfDims(valueInput.shape, nonReducedDims);
    int64_t reducedSize = [valueInput.shape[(NSUInteger)axis] longLongValue];
    canonicalInput = [g reshapeTensor:canonicalInput
                            withShape:@[@(nonReducedSize), @(reducedSize)]
                                 name:nil];

    MPSGraphTensor* valueOut = nullptr;
    MPSGraphTensor* indexOut = nullptr;
    if (kind == ArgReduceKind::kMax) {
        valueOut = [g reductionMaximumWithTensor:canonicalInput axis:1 name:nil];
        indexOut = [g reductionArgMaximumWithTensor:canonicalInput axis:1 name:nil];
    } else {
        valueOut = [g reductionMinimumWithTensor:canonicalInput axis:1 name:nil];
        indexOut = [g reductionArgMinimumWithTensor:canonicalInput axis:1 name:nil];
    }
    if (!valueOut || !indexOut) {
        return ProcessResult::Error("reduce: failed to lower multi-result reduce");
    }

    MPSDataType valueType = GetResultMpsType(op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [g castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [g castTensor:indexOut toType:indexType name:nil];
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(op, 0);
    if (valueShape && valueOut) {
        valueOut = [g reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(op, 1);
    if (indexShape && indexOut) {
        indexOut = [g reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    values[op->getResult(0).getAsOpaquePointer()] = valueOut;
    values[op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

// Unified reduce handler - dispatches based on result count
static ProcessResult HandleReduce(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    if (op->getNumResults() > 1) {
        return HandleMultiResultReduce(g, op, values);
    }
    return HandleSingleResultReduce(g, op, values);
}
REGISTER_MPS_OP("stablehlo.reduce", HandleReduce);

// stablehlo.return is a terminator used inside regions (e.g., reduce body)
// It's handled implicitly by parent operations, not executed directly
static ProcessResult HandleReturn(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    // This should never be called directly - it's handled by the parent operation
    // But we register it so it's not flagged as unsupported during module verification
    MPS_LOG_WARN("stablehlo.return should not be called directly\n");
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.return", HandleReturn);

}  // namespace jax_mps
