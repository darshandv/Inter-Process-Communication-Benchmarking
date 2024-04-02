#include "MatrixOperation.h"
#include "IPCPipe.h"
#include "IPCSharedMemory.h"
#include "IPCSocket.h"
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <random>

int main() {
    std::vector<std::unique_ptr<IPCMethod>> ipcMethods;
    ipcMethods.push_back(std::make_unique<IPCPipe>());
    ipcMethods.push_back(std::make_unique<IPCSharedMemory>());
    ipcMethods.push_back(std::make_unique<IPCSocket>());

    // initialize subprocesses for each IPC method
    for (auto& method : ipcMethods) {
        method->initSubprocess();
    }

    // // v1 code of main
    // for (auto& method : ipcMethods) {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     std::cout<<"\n\nMethod: "<<method->methodName()<<std::endl;
    //     method->sendAndReceive(matrixSize);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> diff = end - start;
    //     std::cout << method->methodName() << " took " << diff.count() << " seconds." << std::endl;
    // }

    // v2 code of main
    
    // random distribution setup - using uniform int distribution
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, ipcMethods.size() - 1);

    // random generator for matrix size
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1024); // Distribution range 1 to 1024

    const int numberOfMatrices = 10; // number of matrices to process
    for (int i = 0; i < numberOfMatrices; ++i) {
        int matrixSize = dis(gen); // generate a random matrix size
        // generate a random matrix
        auto matrix = MatrixOperation::generateRandomMatrix(matrixSize);

        // select an IPC method at random
        int methodIndex = distribution(generator);
        auto& selectedMethod = ipcMethods[methodIndex];

        auto start = std::chrono::high_resolution_clock::now();
        
        // send the matrix to the selected subprocess and receive the squared matrix
        // in v2, its parents responsibility to send size to child
        // selectedMethod->setMatrixSize(matrixSize);
        auto squaredMatrix = selectedMethod->sendAndReceiveV2(matrix);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // output the time taken
        std::cout << "\n\nMethod " << selectedMethod->methodName() << " processed a matrix in " 
                  << elapsed.count() << " seconds." << std::endl;

        std::cout << "Matrix shape: " << matrixSize << "x"<< matrixSize << std::endl;
        // calculate the rate is matrix size in kilo bytes by time taken in seconds
        double rate = (matrixSize * matrixSize * sizeof(CPP_TENSOR_DTYPE)) / (elapsed.count() * 1024 * 1024);
        std::cout << "Rate: " << rate << " MB/sec" << std::endl;
        
        // check if the squared matrix is correct
        bool isSquaredCorrectly = MatrixOperation::checkIfSquaredMatrix(matrix, squaredMatrix);
        if (isSquaredCorrectly) {
            std::cout << "The matrix is squared correctly." << std::endl;
        } else {
            std::cout << "The matrix was not squared correctly." << std::endl;
        }
    }

    for (auto& method : ipcMethods) {
        // exit subprocesses for each IPC method
        method->exitSubprocess();
    }

    return 0;
}
