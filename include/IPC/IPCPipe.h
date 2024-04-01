#ifndef IPCPIPE_H
#define IPCPIPE_H

#include "IPCMethod.h"
#include <torch/torch.h>

class IPCPipe : public IPCMethod {
public:
    void sendAndReceive(int matrixSize) override;
    std::string methodName() const override { return "Pipe"; }

private:
    void writeMatrixToPipe(int fd, const torch::Tensor &matrix);
    void readMatrixFromPipe(int fd, torch::Tensor &matrix, int matrixSize);
};

#endif