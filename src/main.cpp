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
    const int matrixSize = 1024; // Example matrix size
    std::vector<std::unique_ptr<IPCMethod>> ipcMethods;
    // ipcMethods.push_back(std::make_unique<IPCPipe>());
    ipcMethods.push_back(std::make_unique<IPCSharedMemory>());
    // ipcMethods.push_back(std::make_unique<IPCSocket>());

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
    
    // random distribution setup
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, ipcMethods.size() - 1);

    const int numberOfMatrices = 1; // number of matrices to process
    for (int i = 0; i < numberOfMatrices; ++i) {
        // generate a random matrix
        auto matrix = MatrixOperation::generateRandomMatrix(matrixSize);

        // select an IPC method at random
        int methodIndex = distribution(generator);
        auto& selectedMethod = ipcMethods[methodIndex];

        auto start = std::chrono::high_resolution_clock::now();
        
        // send the matrix to the selected subprocess and receive the squared matrix
        // selectedMethod->setMatrixSize(matrixSize);
        auto squaredMatrix = selectedMethod->sendAndReceiveV2(matrix);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Output the time taken
        std::cout << "Method " << selectedMethod->methodName() << " processed a matrix in " 
                  << elapsed.count() << " seconds." << std::endl;
        
        // Check if the squared matrix is correct
        bool isSquaredCorrectly = MatrixOperation::checkIfSquaredMatrix(matrix, squaredMatrix);
        if (isSquaredCorrectly) {
            std::cout << "The matrix was squared correctly." << std::endl;
        } else {
            std::cout << "The matrix was not squared correctly." << std::endl;
        }
    }

    for (auto& method : ipcMethods) {
        method->exitSubprocess();
    }

    return 0;
}
