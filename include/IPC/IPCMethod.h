#ifndef IPCMETHOD_H
#define IPCMETHOD_H

#include <string>

#include "debug.h"

// For C++ standard library containers
#define CPP_TENSOR_DTYPE float

// For PyTorch tensors
#define MATRIX_DTYPE torch::kFloat32

class IPCMethod {
public:
    virtual ~IPCMethod() {}
    virtual void sendAndReceive(int matrixSize) = 0;
    virtual std::string methodName() const = 0;
};

#endif
