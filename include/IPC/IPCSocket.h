#ifndef IPCSOCKET_H
#define IPCSOCKET_H 

#include "IPCMethod.h"
#include <string>
#include <vector>
#include <torch/torch.h>

class IPCSocket: public IPCMethod {
    public:
        void sendAndReceive(int matrixSize) override;
        std::string methodName() const override { return "Socket"; }
    private:
        std::vector<char> serializeTensor(const torch::Tensor &tensor);
        torch::Tensor deserializeTensor(const std::vector<char> &buffer, const std::vector<int64_t> &size);
        ssize_t read_full(int fd, char *buf, size_t count);
        ssize_t write_full(int fd, const char *buf, size_t count);

};

#endif // IPCSOCKET_H