// Binary operations: add, subtract, multiply, divide, maximum, minimum,
// compare, select, clamp, next_after, dot, dot_general

#include <algorithm>

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

REGISTER_MLIR_BINARY_OP("stablehlo.add", addition, add);
REGISTER_MLIR_BINARY_OP("stablehlo.subtract", subtraction, subtract);
REGISTER_MLIR_BINARY_OP("stablehlo.multiply", multiplication, multiply);
REGISTER_MLIR_BINARY_OP("stablehlo.divide", division, divide);
REGISTER_MLIR_BINARY_OP("stablehlo.maximum", maximum, maximum);
REGISTER_MLIR_BINARY_OP("stablehlo.minimum", minimum, minimum);
REGISTER_MLIR_BINARY_OP("stablehlo.remainder", modulo, remainder);
REGISTER_MLIR_BINARY_OP("stablehlo.power", power, power);
REGISTER_MLIR_BINARY_OP("stablehlo.atan2", atan2, atan2);

// Matrix multiplication (dot)
static ProcessResult HandleDot(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("dot: missing input tensor");
    MPSGraphTensor* result = [g matrixMultiplicationWithPrimaryTensor:lhs
                                                      secondaryTensor:rhs
                                                                 name:nil];
    return Result(values, op, result, "dot");
}
REGISTER_MPS_OP("stablehlo.dot", HandleDot);

// Generalized matrix multiplication (dot_general)
// Handles contracting dimensions and batch dimensions
static ProcessResult HandleDotGeneral(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        return ProcessResult::Error("dot_general: expected DotGeneralOp");
    }

    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("dot_general: missing input tensor");

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    NSArray<NSNumber*>* lhsShape = lhs.shape;
    NSArray<NSNumber*>* rhsShape = rhs.shape;
    if (!lhsShape || !rhsShape) {
        return ProcessResult::Error("dot_general: missing input shapes");
    }

    auto containsDim = [](const auto& dims, int64_t dim) {
        return std::find(dims.begin(), dims.end(), dim) != dims.end();
    };
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

    std::vector<int64_t> lhsFreeDims;
    lhsFreeDims.reserve(lhsShape.count);
    for (int64_t i = 0; i < (int64_t)lhsShape.count; ++i) {
        if (!containsDim(lhsBatchDims, i) && !containsDim(lhsContractingDims, i)) {
            lhsFreeDims.push_back(i);
        }
    }

    std::vector<int64_t> rhsFreeDims;
    rhsFreeDims.reserve(rhsShape.count);
    for (int64_t i = 0; i < (int64_t)rhsShape.count; ++i) {
        if (!containsDim(rhsBatchDims, i) && !containsDim(rhsContractingDims, i)) {
            rhsFreeDims.push_back(i);
        }
    }

    std::vector<int64_t> lhsPerm;
    lhsPerm.reserve(lhsShape.count);
    lhsPerm.insert(lhsPerm.end(), lhsBatchDims.begin(), lhsBatchDims.end());
    lhsPerm.insert(lhsPerm.end(), lhsFreeDims.begin(), lhsFreeDims.end());
    lhsPerm.insert(lhsPerm.end(), lhsContractingDims.begin(), lhsContractingDims.end());

    std::vector<int64_t> rhsPerm;
    rhsPerm.reserve(rhsShape.count);
    rhsPerm.insert(rhsPerm.end(), rhsBatchDims.begin(), rhsBatchDims.end());
    rhsPerm.insert(rhsPerm.end(), rhsContractingDims.begin(), rhsContractingDims.end());
    rhsPerm.insert(rhsPerm.end(), rhsFreeDims.begin(), rhsFreeDims.end());

    if (lhsPerm.size() != lhsShape.count || rhsPerm.size() != rhsShape.count) {
        return ProcessResult::Error("dot_general: invalid permutation dimensions");
    }

    MPSGraphTensor* lhsPermuted = lhs;
    if (!isIdentityPermutation(lhsPerm)) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:lhsPerm.size()];
        for (int64_t d : lhsPerm) {
            [perm addObject:@(d)];
        }
        lhsPermuted = [g transposeTensor:lhs permutation:perm name:nil];
    }

    MPSGraphTensor* rhsPermuted = rhs;
    if (!isIdentityPermutation(rhsPerm)) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:rhsPerm.size()];
        for (int64_t d : rhsPerm) {
            [perm addObject:@(d)];
        }
        rhsPermuted = [g transposeTensor:rhs permutation:perm name:nil];
    }

    int64_t lhsM = productOfDims(lhsShape, lhsFreeDims);
    int64_t rhsN = productOfDims(rhsShape, rhsFreeDims);
    int64_t lhsK = productOfDims(lhsShape, lhsContractingDims);
    int64_t rhsK = productOfDims(rhsShape, rhsContractingDims);
    if (lhsK != rhsK) {
        return ProcessResult::Error("dot_general: contracting dimensions mismatch");
    }

    NSMutableArray<NSNumber*>* batchShape = [NSMutableArray arrayWithCapacity:lhsBatchDims.size()];
    for (int64_t d : lhsBatchDims) {
        [batchShape addObject:lhsShape[(NSUInteger)d]];
    }

    NSMutableArray<NSNumber*>* lhsMatmulShape = [NSMutableArray arrayWithArray:batchShape];
    [lhsMatmulShape addObject:@(lhsM)];
    [lhsMatmulShape addObject:@(lhsK)];

    NSMutableArray<NSNumber*>* rhsMatmulShape = [NSMutableArray arrayWithArray:batchShape];
    [rhsMatmulShape addObject:@(rhsK)];
    [rhsMatmulShape addObject:@(rhsN)];

    MPSGraphTensor* lhsForMatmul = [g reshapeTensor:lhsPermuted withShape:lhsMatmulShape name:nil];
    MPSGraphTensor* rhsForMatmul = [g reshapeTensor:rhsPermuted withShape:rhsMatmulShape name:nil];
    MPSGraphTensor* matmul = [g matrixMultiplicationWithPrimaryTensor:lhsForMatmul
                                                      secondaryTensor:rhsForMatmul
                                                                 name:nil];
    if (!matmul) {
        return ProcessResult::Error("dot_general: matrix multiplication lowering failed");
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(op);
    if (outputShape) {
        matmul = [g reshapeTensor:matmul withShape:outputShape name:nil];
    }

    return Result(values, op, matmul, "dot_general");
}
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

// Compare operation
static ProcessResult HandleCompare(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op);
    if (!compareOp) {
        return ProcessResult::Error("compare: expected CompareOp");
    }

    MPSGraphTensor* lhs = GetInputTensor(values, op, 0);
    MPSGraphTensor* rhs = GetInputTensor(values, op, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("compare: missing input tensor");

    auto direction = compareOp.getComparisonDirection();
    using Dir = mlir::stablehlo::ComparisonDirection;

    MPSGraphTensor* result = nil;
    switch (direction) {
        case Dir::LT:
            result = [g lessThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::LE:
            result = [g lessThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::GT:
            result = [g greaterThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::GE:
            result = [g greaterThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::EQ:
            result = [g equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        case Dir::NE:
            result = [g notEqualWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
            break;
        default:
            return ProcessResult::Error("compare: unknown compare direction");
    }

    return Result(values, op, result, "compare");
}
REGISTER_MPS_OP("stablehlo.compare", HandleCompare);

// Select operation (conditional selection: pred ? true_val : false_val)
static ProcessResult HandleSelect(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* pred = GetInputTensor(values, op, 0);
    MPSGraphTensor* onTrue = GetInputTensor(values, op, 1);
    MPSGraphTensor* onFalse = GetInputTensor(values, op, 2);
    if (!pred || !onTrue || !onFalse)
        return ProcessResult::Error("select: missing input tensor");

    MPSGraphTensor* result = [g selectWithPredicateTensor:pred
                                      truePredicateTensor:onTrue
                                     falsePredicateTensor:onFalse
                                                     name:nil];
    return Result(values, op, result, "select");
}
REGISTER_MPS_OP("stablehlo.select", HandleSelect);

// Clamp operation: clamp(min, x, max)
static ProcessResult HandleClamp(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* minVal = GetInputTensor(values, op, 0);
    MPSGraphTensor* operand = GetInputTensor(values, op, 1);
    MPSGraphTensor* maxVal = GetInputTensor(values, op, 2);
    if (!minVal || !operand || !maxVal)
        return ProcessResult::Error("clamp: missing input tensor");

    MPSGraphTensor* result = [g clampWithTensor:operand
                                 minValueTensor:minVal
                                 maxValueTensor:maxVal
                                           name:nil];
    return Result(values, op, result, "clamp");
}
REGISTER_MPS_OP("stablehlo.clamp", HandleClamp);

// next_after(x, y) - returns the next representable floating point value from x towards y
// Implementation follows IEEE 754 nextafter semantics:
// 1. If x == y, return y
// 2. If x or y is NaN, return NaN
// 3. If x == 0, return smallest subnormal with sign of y
// 4. Otherwise, treat x as integer bits and increment/decrement based on direction
static ProcessResult HandleNextAfter(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    MPSGraphTensor* x = GetInputTensor(values, op, 0);
    MPSGraphTensor* y = GetInputTensor(values, op, 1);
    if (!x || !y)
        return ProcessResult::Error("next_after: missing input tensor");

    MPSDataType dtype = x.dataType;

    // Handle scalar tensors - MPS reinterpretCast doesn't support rank-0
    NSArray<NSNumber*>* xShape = x.shape;
    bool isScalar = (xShape.count == 0);
    if (isScalar) {
        x = [g reshapeTensor:x withShape:@[@1] name:nil];
        y = [g reshapeTensor:y withShape:@[@1] name:nil];
    }

    // Constants
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:dtype];
    MPSGraphTensor* one_int = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* neg_one_int = [g constantWithScalar:-1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_positive_int = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_negative_int = [g constantWithScalar:0x80000001 dataType:MPSDataTypeInt32];

    // Bitcast x to int32 (reinterpret bits)
    MPSGraphTensor* x_as_int = [g reinterpretCastTensor:x toType:MPSDataTypeInt32 name:nil];

    // Check if x == y
    MPSGraphTensor* x_eq_y = [g equalWithPrimaryTensor:x secondaryTensor:y name:nil];

    // Check if x is zero
    MPSGraphTensor* x_is_zero = [g equalWithPrimaryTensor:x secondaryTensor:zero name:nil];

    // Check if y > 0 (to determine direction when x == 0)
    MPSGraphTensor* y_gt_zero = [g greaterThanWithPrimaryTensor:y secondaryTensor:zero name:nil];

    // When x == 0, return smallest positive or negative subnormal
    MPSGraphTensor* zero_result_int = [g selectWithPredicateTensor:y_gt_zero
                                               truePredicateTensor:min_positive_int
                                              falsePredicateTensor:min_negative_int
                                                              name:nil];
    MPSGraphTensor* zero_result = [g reinterpretCastTensor:zero_result_int toType:dtype name:nil];

    // For non-zero x, determine direction and increment/decrement
    // If x > 0 and y > x, or x < 0 and y > x: increment (add 1 to int representation)
    // If x > 0 and y < x, or x < 0 and y < x: decrement (subtract 1 from int representation)
    MPSGraphTensor* y_gt_x = [g greaterThanWithPrimaryTensor:y secondaryTensor:x name:nil];

    // x > 0
    MPSGraphTensor* x_gt_zero = [g greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];

    // Determine if we should increment the int representation
    // Increment when: (x > 0 && y > x) || (x < 0 && y < x)
    // Which simplifies to: (x > 0) == (y > x)
    MPSGraphTensor* should_increment = [g equalWithPrimaryTensor:x_gt_zero
                                                 secondaryTensor:y_gt_x
                                                            name:nil];

    // Compute the delta (+1 or -1)
    MPSGraphTensor* delta = [g selectWithPredicateTensor:should_increment
                                     truePredicateTensor:one_int
                                    falsePredicateTensor:neg_one_int
                                                    name:nil];

    // Add delta to x_as_int
    MPSGraphTensor* result_int = [g additionWithPrimaryTensor:x_as_int
                                              secondaryTensor:delta
                                                         name:nil];

    // Bitcast back to float
    MPSGraphTensor* non_zero_result = [g reinterpretCastTensor:result_int toType:dtype name:nil];

    // Select between zero and non-zero cases
    MPSGraphTensor* non_equal_result = [g selectWithPredicateTensor:x_is_zero
                                                truePredicateTensor:zero_result
                                               falsePredicateTensor:non_zero_result
                                                               name:nil];

    // If x == y, return y; otherwise return the computed result
    MPSGraphTensor* result = [g selectWithPredicateTensor:x_eq_y
                                      truePredicateTensor:y
                                     falsePredicateTensor:non_equal_result
                                                     name:nil];

    // Reshape back to scalar if needed
    if (isScalar) {
        result = [g reshapeTensor:result withShape:@[] name:nil];
    }

    return Result(values, op, result, "next_after");
}
REGISTER_MPS_OP("chlo.next_after", HandleNextAfter);

}  // namespace jax_mps
