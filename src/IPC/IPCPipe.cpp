#include "IPCPipe.h"
#include "MatrixOperation.h"
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstring> // For memcpy
#include <algorithm> 

#define DEBUG 1

void IPCPipe::sendAndReceive(int matrixSize) {
    int pipefd[2][2]; // 0 for read, 1 for write
    pid_t pid;
    torch::Tensor matrix, result;

    // create two pipes
    if (pipe(pipefd[0]) == -1 || pipe(pipefd[1]) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "Pipes: 2 pipes created\n");

    pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (pid == 0) { // child process
        close(pipefd[0][1]); // close unused write end of first pipe
        close(pipefd[1][0]); // close unused read end of second pipe

        // read matrix from the pipe
        readMatrixFromPipe(pipefd[0][0], matrix, matrixSize);
        DEBUG_PRINT(1, "Pipes: Child read matrix from the pipe\n");
        // MatrixOperation::printMatrix(matrix);

        // process the matrix
        result = MatrixOperation::squareMatrix(matrix);

        // write the processed matrix back to the pipe
        writeMatrixToPipe(pipefd[1][1], result);
        DEBUG_PRINT(1, "Pipes: Child wrote matrix to the pipe\n");
        // MatrixOperation::printMatrix(result);

        close(pipefd[0][0]);
        close(pipefd[1][1]);
        exit(0);
    } else { // parent process
        close(pipefd[0][0]); // close unused read end of first pipe
        close(pipefd[1][1]); // close unused write end of second pipe

        // generate a random matrix
        matrix = MatrixOperation::generateRandomMatrix(matrixSize);
        
        // write the matrix to the first pipe
        writeMatrixToPipe(pipefd[0][1], matrix);
        DEBUG_PRINT(1, "Pipes: Parent wrote matrix to the pipe\n");
        // MatrixOperation::printMatrix(matrix);

        // read the processed matrix from the second pipe
        readMatrixFromPipe(pipefd[1][0], result, matrixSize);
        DEBUG_PRINT(1, "Pipes: Parent read matrix from the pipe\n");
        // MatrixOperation::printMatrix(result);

        // check if matrix is squared properly
        bool isSquaredCorrectly = MatrixOperation::checkIfSquaredMatrix(matrix, result);
        if (isSquaredCorrectly) {
            std::cout << "Pipes: The matrix was squared correctly." << std::endl;
        } else {
            std::cout << "Pipes: The matrix was not squared correctly." << std::endl;
        }

        close(pipefd[0][1]);
        close(pipefd[1][0]);
        // wait for child process to finish processing
        wait(nullptr);
    }
}

void IPCPipe::writeMatrixToPipe(int fd, const torch::Tensor &matrix) {

    auto data = matrix.data_ptr<CPP_TENSOR_DTYPE>();
    auto totalBytes = matrix.numel() * sizeof(CPP_TENSOR_DTYPE);
    auto bytesWritten = 0;
    while (bytesWritten < totalBytes) {
        size_t elementsToWrite = std::min(static_cast<size_t>(PIPE_BUF / sizeof(CPP_TENSOR_DTYPE)),
                                           static_cast<size_t>(totalBytes - bytesWritten) / sizeof(CPP_TENSOR_DTYPE));
        size_t written = write(fd, data + (bytesWritten / sizeof(CPP_TENSOR_DTYPE)), elementsToWrite * sizeof(CPP_TENSOR_DTYPE));
        if (written == -1) {
            perror("write");
            exit(EXIT_FAILURE);
        }
        bytesWritten += written;
    }
}

void IPCPipe::readMatrixFromPipe(int fd, torch::Tensor &matrix, int matrixSize) {

    const auto totalSize = matrixSize * matrixSize * sizeof(CPP_TENSOR_DTYPE);
    std::vector<CPP_TENSOR_DTYPE> buffer(matrixSize * matrixSize);
    size_t bytesReadTotal = 0;

    // continue reading until all data has been received
    while (bytesReadTotal < totalSize) {
        auto remaining = totalSize - bytesReadTotal;
        ssize_t bytesRead = read(fd, &buffer.data()[bytesReadTotal / sizeof(CPP_TENSOR_DTYPE)], remaining);
        if (bytesRead < 0) {
            perror("read");
            exit(EXIT_FAILURE);
        } else if (bytesRead == 0) {
            // end of file reached or pipe closed, might need special handling
            break; // may indicate that the writing side has closed the pipe, so break
        } else {
            bytesReadTotal += bytesRead;
        }
    }

    // Ensure the entire matrix data was read

    
    if (bytesReadTotal != totalSize) {
        std::cerr << "Error: Did not read the entire matrix from the pipe." << std::endl;
        exit(EXIT_FAILURE);
    }

    matrix = torch::from_blob(buffer.data(), {matrixSize, matrixSize}).clone();
}
