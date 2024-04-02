#include "IPCPipe.h"
#include "MatrixOperation.h"
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <cstring> // For memcpy
#include <algorithm> 
#include <fcntl.h>


IPCPipe::IPCPipe() {
    // create two pipes
    if (pipe(dataPipe[0]) == -1 || pipe(dataPipe[1]) == -1 || pipe(controlPipe) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }
    
}

IPCPipe::~IPCPipe() {
    close(dataPipe[0][0]); close(dataPipe[0][1]);
    close(dataPipe[1][0]); close(dataPipe[1][1]);
    close(controlPipe[0]); close(controlPipe[1]);
    waitpid(childPid, nullptr, 0);
}


void IPCPipe::initSubprocess() {
    childPid = fork();
    if (childPid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (childPid == 0) { // child process
        // close(controlPipe[1]); // close unused write end of control pipe
        torch::Tensor matrix, result;
        char cmd[10];
        while (true) {
            ssize_t bytesRead = read(controlPipe[0], cmd, sizeof(cmd));
            if (bytesRead > 0) {
                cmd[bytesRead] = '\0';
                if (strncmp(cmd, "Process", 7) == 0) {
                    DEBUG_PRINT(2, "Pipes: Child entered processing\n");

                    // read matrix from the pipe
                    readMatrixFromPipe(dataPipe[0][0], matrix, matrixSize);
                    DEBUG_PRINT(1, "Pipes: Child read matrix from the pipe\n");
                    // MatrixOperation::printMatrix(matrix);

                    // process the matrix
                    result = MatrixOperation::squareMatrix(matrix);

                    // write the processed matrix back to the pipe
                    writeMatrixToPipe(dataPipe[1][1], result);
                    DEBUG_PRINT(1, "Pipes: Child wrote matrix to the pipe\n");
                    // MatrixOperation::printMatrix(result);

                    
                    // after processing, send "Wait" command to itself to be in wait state
                    writeToControlPipe("Wait");
                } else if (strncmp(cmd, "Exit", 4) == 0) {
                    break; // exit the loop and clean up
                }
        
            }
            
        }
        exit(0);
    } else { // Parent process
        
    }
}

void IPCPipe::writeToControlPipe(const char* msg, int matrixSize) {
    // send a "Size" message if matrixSize is provided
    if (matrixSize >= 0) {
        const char* sizeMsg = "Size";
        write(controlPipe[1], sizeMsg, strlen(sizeMsg) + 1);    // send "Size" indicator
        write(controlPipe[1], &matrixSize, sizeof(matrixSize)); // send the actual matrixSize
    }
    // send the command message
    else if (write(controlPipe[1], msg, strlen(msg) + 1) == -1) {
        perror("write");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "Pipes: Parent wrote to control pipe: " << msg << std::endl);
}


std::string IPCPipe::readFromControlPipe() {
    char buffer[256];
    ssize_t bytesRead = read(controlPipe[0], buffer, sizeof(buffer));
    if (bytesRead == -1) {
        perror("read");
        exit(EXIT_FAILURE);
    }
    return std::string(buffer);
}

// set matrix size
void IPCPipe::setMatrixSize(int matrixSize) {
    // not using this in v2 anymore
    // signal child process to set matrix size
    // writeToControlPipe("Size", matrixSize);
    this->matrixSize = matrixSize;
}

torch::Tensor IPCPipe::sendAndReceiveV2(const torch::Tensor& matrix) {
    torch::Tensor result;

    // signal child process to start processing
    writeToControlPipe("Process");

    // write the matrix to the first pipe
    writeMatrixToPipe(dataPipe[0][1], matrix);
    DEBUG_PRINT(1, "Pipes: Parent wrote matrix to the pipe\n");
    // MatrixOperation::printMatrix(matrix);

    // read the processed matrix from the second pipe
    readMatrixFromPipe(dataPipe[1][0], result, matrixSize);
    DEBUG_PRINT(1, "Pipes: Parent read matrix from the pipe\n");
    // MatrixOperation::printMatrix(result);

    return result;
}

void IPCPipe::exitSubprocess() {
    writeToControlPipe("Exit");
    close(controlPipe[1]);
    waitpid(childPid, nullptr, 0);
}


void IPCPipe::writeMatrixToPipe(int fd, const torch::Tensor &matrix) {

    auto data = matrix.data_ptr<CPP_TENSOR_DTYPE>();
    auto totalBytes = matrix.numel() * sizeof(CPP_TENSOR_DTYPE);
    int matrixSize = matrix.size(0); // assuming square matrix
    write(fd, &matrixSize, sizeof(matrixSize));
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

void IPCPipe::readMatrixFromPipe(int fd, torch::Tensor &matrix, int matrixSizes) {
    //print matrix size
    int matrixSize;
    read(fd, &matrixSize, sizeof(matrixSize));
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

    // ensure the entire matrix data was read
    if (bytesReadTotal != totalSize) {
        std::cerr << "Error: Did not read the entire matrix from the pipe." << std::endl;
        exit(EXIT_FAILURE);
    }

    matrix = torch::from_blob(buffer.data(), {matrixSize, matrixSize}).clone();
}

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
    } else if (pid == 0) {   // child process
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

