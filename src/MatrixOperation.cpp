#include "MatrixOperation.h"

torch::Tensor MatrixOperation::generateRandomMatrix(int size) {
    return torch::rand({size, size});
}

torch::Tensor MatrixOperation::squareMatrix(const torch::Tensor& matrix) {
    return matrix.square();
}

bool MatrixOperation::checkIfSquaredMatrix(const torch::Tensor& original, const torch::Tensor& squared) {
    // check if the shapes of the two tensors are identical
    if (original.sizes() != squared.sizes()) {
        return false;
    }
    
    // calculate the square of the original matrix
    auto expectedSquared = original.square();

    // check if the expected squared matrix is close to the squared matrix
    // within a tolerance to account for floating-point arithmetic errors.
    bool isClose = torch::allclose(expectedSquared, squared, 1e-6, 1e-4);
    return isClose;
}

void MatrixOperation::printMatrix(const torch::Tensor& matrix) {
    std::cout << matrix << std::endl;
}