// Tensor creation operations: constant, iota

#include <cstring>
#include <vector>

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Constant creation - creates a constant tensor from MLIR constant op
static ProcessResult HandleConstant(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
    if (!constantOp) {
        return ProcessResult::Error("constant: expected ConstantOp");
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("constant: invalid dtype");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    auto value = constantOp.getValue();

    // Check for empty tensor (any dimension is 0)
    // MPSGraph doesn't support empty tensors, so create a minimal [1] tensor instead
    // The scatter handler will detect empty indices based on MLIR types and handle appropriately
    bool isEmpty = false;
    for (NSNumber* dim in shape) {
        if ([dim integerValue] == 0) {
            isEmpty = true;
            break;
        }
    }
    if (isEmpty) {
        // Create a minimal tensor with shape [1] and a dummy value
        // This is safe because operations that use this tensor will detect
        // empty dimensions from the MLIR types and not actually use the tensor values
        MPSGraphTensor* result = [g constantWithScalar:0 shape:@[@1] dataType:dtype];
        SetOutputTensor(values, op, result);
        return ProcessResult{};
    }

    MPSGraphTensor* result = nil;
    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {
        // Check if it's a splat (single value broadcast to all elements)
        if (denseAttr.isSplat()) {
            auto elemType = denseAttr.getElementType();

            // Complex splat: extract real and imaginary parts separately.
            if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elemType)) {
                auto complexVal = denseAttr.getSplatValue<std::complex<float>>();
                double realPart = complexVal.real();
                double imagPart = complexVal.imag();
                if (shape.count == 0) {
                    result = [g constantWithRealPart:realPart
                                       imaginaryPart:imagPart
                                            dataType:dtype];
                } else {
                    result = [g constantWithRealPart:realPart
                                       imaginaryPart:imagPart
                                               shape:shape
                                            dataType:dtype];
                }
                return Result(values, op, result, "constant");
            }

            // Use raw-element replication for splats instead of constantWithScalar.
            // This preserves exact bit patterns for BF16 and avoids scalar conversion pitfalls.
            auto rawData = denseAttr.getRawData();
            size_t elemSize = rawData.size();
            if (elemSize == 0) {
                return ProcessResult::Error("constant: invalid splat element size");
            }

            bool isScalarShape = (shape.count == 0);
            NSArray<NSNumber*>* storageShape = isScalarShape ? @[@1] : shape;

            size_t numel = 1;
            for (NSNumber* dim in storageShape) {
                numel *= (size_t)[dim unsignedLongLongValue];
            }

            std::vector<uint8_t> expanded(elemSize * numel);
            for (size_t i = 0; i < numel; ++i) {
                memcpy(expanded.data() + i * elemSize, rawData.data(), elemSize);
            }
            NSData* data = [NSData dataWithBytes:expanded.data() length:expanded.size()];
            result = [g constantWithData:data shape:storageShape dataType:dtype];
            if (isScalarShape) {
                result = [g reshapeTensor:result withShape:@[] name:nil];
            }
        } else {
            // Non-splat dense constant - use raw data
            auto rawData = denseAttr.getRawData();
            NSData* data = [NSData dataWithBytes:rawData.data() length:rawData.size()];
            result = [g constantWithData:data shape:shape dataType:dtype];
        }
    }

    if (!result)
        return ProcessResult::Error("constant: unsupported value type");
    return Result(values, op, result, "constant");
}
REGISTER_MPS_OP("stablehlo.constant", HandleConstant);

// Iota - create an array of indices
static ProcessResult HandleIota(MPSGraph* g, mlir::Operation* op, ValueMap& values) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp) {
        return ProcessResult::Error("iota: expected IotaOp");
    }

    MPSDataType dtype = GetResultMpsType(op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("iota: invalid dtype");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(op);
    int64_t iotaDim = static_cast<int64_t>(iotaOp.getIotaDimension());

    // Create a coordinate tensor along the iota dimension
    MPSGraphTensor* result = [g coordinateAlongAxis:(NSInteger)iotaDim withShape:shape name:nil];

    // Cast to the target type if needed
    if (result.dataType != dtype) {
        result = [g castTensor:result toType:dtype name:nil];
    }

    return Result(values, op, result, "iota");
}
REGISTER_MPS_OP("stablehlo.iota", HandleIota);

}  // namespace jax_mps
