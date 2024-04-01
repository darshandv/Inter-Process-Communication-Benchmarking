#ifndef MATRIXOPERATION_H
#define MATRIXOPERATION_H

#include <torch/torch.h>

class MatrixOperation {
public:
    static torch::Tensor generateRandomMatrix(int size);
    static torch::Tensor squareMatrix(const torch::Tensor& matrix);
    static bool checkIfSquaredMatrix(const torch::Tensor& original, const torch::Tensor& squared);

    static void printMatrix(const torch::Tensor& matrix);
};

#endif