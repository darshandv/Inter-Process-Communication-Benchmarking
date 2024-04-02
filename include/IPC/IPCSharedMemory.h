#ifndef IPCSHAREDMEMORY_H
#define IPCSHAREDMEMORY_H

#include "IPCMethod.h"
#include <string>
#include <semaphore.h>

class IPCSharedMemory : public IPCMethod {
public:
    IPCSharedMemory();
    ~IPCSharedMemory() override;
    void sendAndReceive(int matrixSize) override;
    std::string methodName() const override {return "SharedMemory";};

    void initSubprocess() override;
    torch::Tensor sendAndReceiveV2(const torch::Tensor& matrix) override;
    void exitSubprocess() override;

private:
    int shmFd = -1;                                   // file descriptor for the shared memory object
    void* shmAddr = nullptr;                          // pointer to the shared memory object
    off_t shmSize = 128*128*sizeof(CPP_TENSOR_DTYPE); // size of the shared memory segment
    const char* shmName = "/dv_ipc_shared_mem";       // name of the shared memory object
    pid_t childPid = -1;

    // Semaphores for synchronization
    sem_t* sem_parent_to_child;                       // Semaphore for parent-to-child signaling
    sem_t* sem_child_to_parent;                       // Semaphore for child-to-parent signaling
    sem_t* sem_exit;                                  // semaphore for signaling exit

    torch::Tensor writeMatrixInBatchesAndReadBack(const torch::Tensor& matrix);
    bool processMatrixInBatches();

};

#endif // IPCSHAREDMEMORY_H
