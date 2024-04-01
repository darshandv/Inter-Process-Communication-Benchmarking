#include "IPCSocket.h"
#include "MatrixOperation.h"
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>


void IPCSocket::sendAndReceive(int matrixSize) {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    const int port = 8080;

    // creating socket fd
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // attaching socket to port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // bind the socket to port 8080
    if (bind(server_fd, (struct sockaddr *)&address,sizeof(address))< 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid==-1){
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (pid==0) { // child process
        // connect to the server
        int sock = 0;
        struct sockaddr_in serv_addr;
        if ((sock=socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("\n Error in socket creation \n");
            return;
        }
        DEBUG_PRINT(1, "Socket: Child created socket\n");
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);

        // convert address from text to binary
        if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0){
            printf("\nInvalid address/ Address not supported \n");
            return;
        }

        DEBUG_PRINT(1, "Socket: Child converted Address using inet_pton. Trying to connect...\n");
        while (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
            usleep(1000); // sleep for 1ms and then retry
        }

        // receive the matrix from the server
        std::vector<char> recv_buf(matrixSize * matrixSize * sizeof(CPP_TENSOR_DTYPE));
        // read(sock, recv_buf.data(), recv_buf.size());
        ssize_t bytes_read = read_full(sock, recv_buf.data(), recv_buf.size());
        DEBUG_PRINT(1, "Socket:Child Read "<<bytes_read<<" bytes\n");
        
        if (bytes_read < 0) {
            perror("read_full failed");
            exit(EXIT_FAILURE);
        }

        torch::Tensor matrix = deserializeTensor(recv_buf, {matrixSize, matrixSize});
        DEBUG_PRINT(1, "Socket: Child completed reading buffer and deserializing\n");
        // MatrixOperation::printMatrix(matrix);
        torch::Tensor squared_matrix = MatrixOperation::squareMatrix(matrix);

        DEBUG_PRINT(1, "Socket: Child deserialized and squared matrix\n");
        // MatrixOperation::printMatrix(squared_matrix);
        // send the processed matrix back to server
        auto send_buf = serializeTensor(squared_matrix);
        // write(sock, send_buf.data(), send_buf.size());
        ssize_t bytes_written = write_full(sock, send_buf.data(), send_buf.size());
        DEBUG_PRINT(1, "Socket: Child wrote "<<bytes_written<<" bytes to the socket\n");
        if (bytes_written < 0) {
            perror("write_full failed");
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT(1, "Socket: Child Serialized and and wrote to socket\n");
        exit(0);
    } else { //parent process
        DEBUG_PRINT(1, "Socket: Parent accepting connections\n");
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0){
            perror("accept");
            exit(EXIT_FAILURE);
        }

        // send matrix to child
        torch::Tensor matrix = MatrixOperation::generateRandomMatrix(matrixSize);
        auto send_buf = serializeTensor(matrix);
        // write(new_socket, send_buf.data(), send_buf.size());
        ssize_t bytes_written = write_full(new_socket, send_buf.data(), send_buf.size());
        if (bytes_written < 0) {
            perror("write_full failed");
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT(1, "Socket: Parent wrote "<<bytes_written<<" bytes to the socket\n");
        // MatrixOperation::printMatrix(matrix);

        std::vector<char>recv_buf(matrixSize * matrixSize * sizeof(CPP_TENSOR_DTYPE));
        // read(new_socket, recv_buf.data(), recv_buf.size());
        ssize_t bytes_read = read_full(new_socket, recv_buf.data(), recv_buf.size());
        if (bytes_read < 0) {
            perror("read_full failed");
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT(1, "Socket: Parent read "<<bytes_read<<" from the socket\n");

        // Deserialize the received buffer into a tensor
        auto result = deserializeTensor(recv_buf, {matrixSize, matrixSize});
        DEBUG_PRINT(1, "Socket: Parent deserialized the matrix\n");
        // MatrixOperation::printMatrix(result);

        // Check if the received squared matrix matches the reference squared matrix
        bool isSquaredCorrectly = MatrixOperation::checkIfSquaredMatrix(matrix, result);
        if (isSquaredCorrectly) {
            std::cout << "Socket: The matrix was squared correctly." << std::endl;
        } else {
            std::cout << "Socket: The matrix was not squared correctly." << std::endl;
        }


        close(new_socket);
        close(server_fd);

        // wait for child
        wait(nullptr);
    }
}

// function to serialize the tensor
std::vector<char> IPCSocket::serializeTensor(const torch::Tensor &tensor){
    auto d_ptr = tensor.data_ptr<CPP_TENSOR_DTYPE>();
    auto num_bytes = tensor.numel() * sizeof(CPP_TENSOR_DTYPE);
    std::vector<char> buffer(num_bytes);
    std::memcpy(buffer.data(), d_ptr, num_bytes);
    return buffer;
}

// function to deserialize the tensor
torch::Tensor IPCSocket::deserializeTensor(const std::vector<char> &buffer, const std::vector<int64_t> &size){
    torch::Tensor tensor = torch::from_blob((void*)buffer.data(), at::IntArrayRef(size), MATRIX_DTYPE).clone();
    return tensor;
}

// attempt to read exactly 'count' bytes from 'fd' into 'buf'.
// returns the number of bytes read, or -1 on error.
ssize_t IPCSocket::read_full(int fd, char *buf, size_t count) {
    size_t total_read = 0;
    while (total_read < count) {
        ssize_t res = read(fd, buf + total_read, count - total_read);
        if (res < 0) {
            if (errno == EINTR) continue; // if interrupted by signal, try again
            return -1; // return error on actual read error
        }
        if (res == 0) break; // break on EOF
        total_read += res;
    }
    return total_read;
}

// Attempt to write exactly 'count' bytes from 'buf' to 'fd'.
// Returns the number of bytes written, or -1 on error.
ssize_t IPCSocket::write_full(int fd, const char *buf, size_t count) {
    size_t total_written = 0;
    while (total_written < count) {
        ssize_t res = write(fd, buf + total_written, count - total_written);
        if (res < 0) {
            if (errno == EINTR) continue; // if interrupted by signal, try again
            return -1; // return error on actual write error
        }
        total_written += res;
    }
    return total_written;
}

