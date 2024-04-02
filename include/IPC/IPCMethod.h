#ifndef IPCMETHOD_H
#define IPCMETHOD_H

#include <string>
#include <torch/torch.h>

#include "debug.h"

// for C++ standard library containers
#define CPP_TENSOR_DTYPE float

// for PyTorch tensors
#define MATRIX_DTYPE torch::kFloat32

class IPCMethod {
public:
    virtual ~IPCMethod() {}
    virtual void initSubprocess() = 0; // to setup communication channel and fork
    virtual void exitSubprocess() = 0; // to close communication channel and exit
    virtual void sendAndReceive(int matrixSize) = 0;
    virtual torch::Tensor sendAndReceiveV2(const torch::Tensor& matrix) = 0;
    virtual std::string methodName() const = 0;

};

#endif
