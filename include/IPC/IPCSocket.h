#ifndef IPCSOCKET_H
#define IPCSOCKET_H 

#include "IPCMethod.h"
#include <string>
#include <vector>

class IPCSocket: public IPCMethod {
    public:
        IPCSocket();
        ~IPCSocket() override;
        void initSubprocess() override; // setup communication channel and fork
        void exitSubprocess() override; // close communication channel and exit
        void sendAndReceive(int matrixSize) override; // Placeholder for backward compatibility
        torch::Tensor sendAndReceiveV2(const torch::Tensor& matrix) override; // actual implementation for tensor transmission
        std::string methodName() const override { return "Socket"; }
    private:
        int serverFd = -1; // server socket file descriptor
        int clientFd = -1; // client socket file descriptor
        pid_t childPid = -1; // PID of the child process
        int customPort = 8080; // port number for socket communication

        // utility methods for socket operations
        int createSocket();
        void connectToServer(int sock, const char* serverAddress, int port);
        void setupServer(int& server_fd, int port, struct sockaddr_in& address);
        void closeSockets();
        void sendTensor(int socketFd, const torch::Tensor& tensor);
        torch::Tensor receiveTensor(int socketFd, int matrixSize=-1);

        std::vector<char> serializeTensor(const torch::Tensor &tensor);
        torch::Tensor deserializeTensor(const std::vector<char> &buffer, const std::vector<int64_t> &size);
        ssize_t read_full(int fd, char *buf, size_t count);
        ssize_t write_full(int fd, const char *buf, size_t count);

};

#endif // IPCSOCKET_H