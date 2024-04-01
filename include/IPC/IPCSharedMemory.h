#ifndef IPCSHAREDMEMORY_H
#define IPCSHAREDMEMORY_H

#include "IPCMethod.h"
#include <string>

class IPCSharedMemory : public IPCMethod {
public:
    void sendAndReceive(int matrixSize) override;
    std::string methodName() const override {return "SharedMemory";};
};

#endif // IPCSHAREDMEMORY_H
