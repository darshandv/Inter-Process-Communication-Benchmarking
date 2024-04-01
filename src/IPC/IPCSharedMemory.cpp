#include "IPCSharedMemory.h"
#include "MatrixOperation.h"
#include <iostream>
#include <errno.h> // Include errno.h for errno
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring> // For memcpy
#include <semaphore.h>

sem_t *sem_parent_to_child, *sem_child_to_parent;

#define DEBUG 0

void IPCSharedMemory::sendAndReceive(int matrixSize){
    const char* memname = "/dv_mk1_shared_memory";
    shm_unlink(memname);
    int memFd;
    void *shared_mem;
    size_t memSize = matrixSize * matrixSize * sizeof(CPP_TENSOR_DTYPE);

    // child process id
    pid_t pid;
    
    // create shared memory object
    memFd = shm_open(memname, O_CREAT | O_RDWR, 0666);
    if (memFd == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }

    // configure the size of the shared memory object
    if (ftruncate(memFd, memSize) == -1) {
        int errsv = errno; // Capture errno immediately after the failed call
        std::cerr << "ftruncate failed: " << strerror(errsv) << " (Error code: " << errsv << ")\n";
        exit(EXIT_FAILURE);
    }

    sem_parent_to_child = sem_open("/sem_parent_to_child", O_CREAT, 0666, 0);
    sem_child_to_parent = sem_open("/sem_child_to_parent", O_CREAT, 0666, 0);

    // forking
    pid = fork();
    if (pid == -1){
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (pid==0) {
        // map shared memory in child's address space
        
        shared_mem = mmap(NULL, memSize, PROT_READ | PROT_WRITE, MAP_SHARED, memFd, 0);
        if (shared_mem == MAP_FAILED) {
            perror("mmap");
            exit(EXIT_FAILURE);
        }
        sem_wait(sem_parent_to_child); // wait for parent to write
        torch::Tensor matrix = torch::from_blob(shared_mem, {matrixSize, matrixSize}, MATRIX_DTYPE).clone();
        DEBUG_PRINT(1, "SharedMem: Child process received matrix\n");
        // MatrixOperation::printMatrix(matrix);

        torch::Tensor result = MatrixOperation::squareMatrix(matrix);
        DEBUG_PRINT(1, "SharedMem: Child process squared matrix\n");
        // MatrixOperation::printMatrix(result);

        // copy squred matric back to shared memory
        memcpy(shared_mem, result.data_ptr<CPP_TENSOR_DTYPE>(), memSize);
        sem_post(sem_child_to_parent); // signal parent that processing is done

        // unmap and close
        if (munmap(shared_mem, memSize) == -1) {
            perror("munmap");
            exit(EXIT_FAILURE);
        }
        exit(0);
    } else { // Parent process
        // map shared memroy in address space of parent
        shared_mem = mmap(NULL, memSize, PROT_READ | PROT_WRITE, MAP_SHARED, memFd, 0);
        if (shared_mem == MAP_FAILED) {
            perror("mmap");
            exit(EXIT_FAILURE);
        }

        // generate random matrix and copy to shared memory
        torch::Tensor matrix = MatrixOperation::generateRandomMatrix(matrixSize);
        
        memcpy(shared_mem, matrix.data_ptr<CPP_TENSOR_DTYPE>(), memSize);
        sem_post(sem_parent_to_child); // signal child to read

        DEBUG_PRINT(1, "SharedMem: Parent process generated matrix and copied to shared memory.\n");
        // MatrixOperation::printMatrix(matrix);

        sem_wait(sem_child_to_parent); // wait for child to process
        // read back the result
        torch::Tensor result = torch::from_blob(shared_mem, {matrixSize, matrixSize}, MATRIX_DTYPE);
        
        DEBUG_PRINT(1, "SharedMem: Parent process received squared matrix.\n");
        // MatrixOperation::printMatrix(result);
        
        // check if its squared properly
        bool isSquaredCorrectly = MatrixOperation::checkIfSquaredMatrix(matrix, result);
        if (isSquaredCorrectly) {
            std::cout << "SharedMem: The matrix was squared correctly." << std::endl;
        } else {
            std::cout << "SharedMem: The matrix was not squared correctly." << std::endl;
        }
        // cleanup
        if (munmap(shared_mem, memSize) == -1) {
            perror("munmap");
            exit(EXIT_FAILURE);
        }
        shm_unlink(memname);
        // wait for child process to finish
        wait(nullptr);

        // cleanup
        sem_close(sem_parent_to_child);
        sem_close(sem_child_to_parent);
        sem_unlink("/sem_parent_to_child");
        sem_unlink("/sem_child_to_parent");
        exit(0);
    }
}