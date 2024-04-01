#ifndef IPCPIPE_H
#define IPCPIPE_H

#include "IPCMethod.h"

class IPCPipe : public IPCMethod {
public:
    IPCPipe();
    ~IPCPipe() override;
    void sendAndReceive(int matrixSize) override;
    std::string methodName() const override { return "Pipe"; }

    void initSubprocess() override;
    void exitSubprocess() override;
    torch::Tensor sendAndReceiveV2(const torch::Tensor& matrix) override;
    void setMatrixSize(int matrixSize);
    
private:
    int matrixSize = 4; // use same varaible to communicate matrix size (can use separate pipe for this too)
    int dataPipe[2][2]; // pipe for matrix data: [0] is read end, [1] is write end
    int controlPipe[2]; // control pipe: [0] is read end, [1] is write end
    pid_t childPid = -1;  // PID of the child process
    
    std::string readFromControlPipe();
    void writeToControlPipe(const char* msg, int matrixSize = -1);

    void writeMatrixToPipe(int fd, const torch::Tensor &matrix);
    void readMatrixFromPipe(int fd, torch::Tensor &matrix, int matrixSize);
};

#endif