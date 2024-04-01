#include "MatrixOperation.h"
#include "IPCPipe.h"
#include "IPCSharedMemory.h"
#include "IPCSocket.h"
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>

int main() {
    const int matrixSize = 4; // Example matrix size
    std::vector<std::unique_ptr<IPCMethod>> ipcMethods;
    // ipcMethods.push_back(std::make_unique<IPCPipe>());
    // ipcMethods.push_back(std::make_unique<IPCSharedMemory>());
    ipcMethods.push_back(std::make_unique<IPCSocket>());

    for (auto& method : ipcMethods) {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout<<"\n\nMethod: "<<method->methodName()<<std::endl;
        method->sendAndReceive(matrixSize);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << method->methodName() << " took " << diff.count() << " seconds." << std::endl;
    }

    return 0;
}
