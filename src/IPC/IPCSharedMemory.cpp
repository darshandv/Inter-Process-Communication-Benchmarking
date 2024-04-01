#include "IPCSharedMemory.h"
#include "MatrixOperation.h"
#include <iostream>
#include <errno.h> // include errno.h for errno
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring> // for memcpy
#include <signal.h> // for kill

// constructor
IPCSharedMemory::IPCSharedMemory() {
    // unlink old and open new semaphores
    sem_unlink("/sem_parent_to_child");
    sem_unlink("/sem_child_to_parent");
    sem_unlink("/sem_exit");

    sem_parent_to_child = sem_open("/sem_parent_to_child", O_CREAT, 0666, 0);
    if (sem_parent_to_child == SEM_FAILED) {
        perror("Error opening semaphore for parent to child");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "SharedMem: Parent to child semaphore opened\n");

    sem_child_to_parent = sem_open("/sem_child_to_parent", O_CREAT, 0666, 0);
    if (sem_child_to_parent == SEM_FAILED) {
        perror("Error opening semaphore for child to parent");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "SharedMem: Child to parent semaphore opened\n");

    sem_exit = sem_open("/sem_exit", O_CREAT, 0666, 0);
    if (sem_exit == SEM_FAILED) {
        perror("Error opening semaphore for exit");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "SharedMem: Exit semaphore opened\n");

    // create shared memory
    shmFd = shm_open(shmName, O_CREAT | O_RDWR, 0666);
    if (shmFd == -1) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "SharedMem: Shared memory created\n");

    DEBUG_PRINT(2, "SharedMem: Parent - shmSize: " << shmSize<<std::endl);
    if (ftruncate(shmFd, shmSize) == -1) {
        perror("ftruncate error");
        std::cerr << "ftruncate failed with errno " << errno << std::endl;
        // exit(EXIT_FAILURE);
    }
    DEBUG_PRINT(1, "SharedMem: Shared memory truncated\n");
}

// destructor
IPCSharedMemory::~IPCSharedMemory() {
    // Cleanup
    if (shmFd != -1) {
        close(shmFd);
        shm_unlink(shmName);
    }
    sem_close(sem_parent_to_child);
    sem_close(sem_child_to_parent);
    sem_unlink("/sem_parent_to_child");
    sem_unlink("/sem_child_to_parent");
    DEBUG_PRINT(1, "SharedMem: Cleaned up shared memory and semaphores in ~IPCSharedMemory\n");
}

void IPCSharedMemory::initSubprocess() {
    childPid = fork();
    if (childPid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    } else if (childPid == 0) { // Child
        DEBUG_PRINT(1, "SharedMem: Child process created. Doing mmap\n");
        DEBUG_PRINT(2, "SharedMem: Child - shmFd: " << shmFd<<std::endl);
        shmAddr = mmap(NULL, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
        if (shmAddr == MAP_FAILED) {
            perror("mmap");
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINT(1, "SharedMem: Child process mapped shared memory\n");
        
        bool shouldChildExit = false;
        while (shouldChildExit == false) {
            shouldChildExit = processMatrixInBatches();

            // sem_wait(sem_parent_to_child);
            // // non-blocking check if the exit semaphore was posted
            // if (sem_trywait(sem_exit) == 0) {
            //     std::cout << "SharedMem: Child process exiting...\n";
            //     break; // exit the loop and thus the process
            // }
            // // Process the matrix here, use shmAddr
            // torch::Tensor matrix = torch::from_blob(shmAddr, {4, 4}, MATRIX_DTYPE).clone();

            // DEBUG_PRINT(1, "SharedMem: Child process received matrix\n");
            // MatrixOperation::printMatrix(matrix);

            // torch::Tensor result = MatrixOperation::squareMatrix(matrix);
            // DEBUG_PRINT(1, "SharedMem: Child process squared matrix\n");
            // MatrixOperation::printMatrix(result);

            // // copy squred matric back to shared memory
            // memcpy(shmAddr, result.data_ptr<CPP_TENSOR_DTYPE>(), result.numel() * sizeof(CPP_TENSOR_DTYPE));
            // // signal back to parent
            // sem_post(sem_child_to_parent);

        }

        // Cleanup
        munmap(shmAddr, shmSize);
        exit(0);
    }
    // Parent continues without waiting here
}

torch::Tensor IPCSharedMemory::sendAndReceiveV2(const torch::Tensor& matrix) {
    // size calculation for shared memory
    DEBUG_PRINT(1, "SharedMem: Parent process sending matrix to child process\n");


    shmAddr = mmap(NULL, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    if (shmAddr == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    // if (ftruncate(shmFd, shmSize) == -1) {
    //     perror("ftruncate");
    //     exit(EXIT_FAILURE);
    // }

    // assuming matrix is serialized to contiguous memory here
    // copy matrix data to shared memory
    // std::memcpy(shmAddr, matrix.data_ptr<CPP_TENSOR_DTYPE>(), shmSize);
    

    // // signal child process that data is ready
    // sem_post(sem_parent_to_child);

    // // Wait for the child to process the data
    // sem_wait(sem_child_to_parent);

    // // Assume result will be written back to the same shared memory location
    // // Deserialize the result from shared memory back into a torch::Tensor
    // auto result = torch::from_blob(shmAddr, {matrix.size(0), matrix.size(1)}, torch::kFloat32).clone();
    
    DEBUG_PRINT(1, "SharedMem: Parent process has generated the matrix\n");
    // MatrixOperation::printMatrix(matrix);
    auto result = writeMatrixInBatchesAndReadBack(matrix);
    DEBUG_PRINT(1, "SharedMem: Parent process received squared matrix\n");
    // MatrixOperation::printMatrix(result);

    munmap(shmAddr, shmSize);
    return result;
}


// write matrix in batches
torch::Tensor IPCSharedMemory::writeMatrixInBatchesAndReadBack(const torch::Tensor& matrix) {
    int totalElements = matrix.numel();
    // write totalElements at the beginning of shared memory
    std::memcpy(shmAddr, &totalElements, sizeof(int));

    // immediately signal the child that totalElements is available
    sem_post(sem_parent_to_child);

    int batchSizeInBytes = shmSize - sizeof(int); // adjusting for totalElements
    int batchSize = batchSizeInBytes / sizeof(CPP_TENSOR_DTYPE); // elements per batch
    auto ptr = matrix.data_ptr<CPP_TENSOR_DTYPE>();
    char* batchPtr = static_cast<char*>(shmAddr) + sizeof(int); // offset by size of int

    // Wait for child to acknowledge reading totalElements
    sem_wait(sem_child_to_parent);

    torch::Tensor result = torch::empty({matrix.size(0), matrix.size(1)}, matrix.options());
    auto resultPtr = result.data_ptr<CPP_TENSOR_DTYPE>();

    for (int i = 0; i < totalElements;) {
        int currentBatchSize = std::min(batchSize, totalElements - i);
        int currentBatchBytes = currentBatchSize * sizeof(CPP_TENSOR_DTYPE);

        // copy current batch to shared memory after the totalElements
        std::memcpy(batchPtr, ptr + i, currentBatchBytes);

        // signal child process that batch is ready
        sem_post(sem_parent_to_child);

        // wait for the child to signal back
        sem_wait(sem_child_to_parent);

        // Read the squared matrix batch back from shared memory
        std::memcpy(resultPtr + i, batchPtr, currentBatchSize * sizeof(CPP_TENSOR_DTYPE));

        i += currentBatchSize; // update for the next iteration
    }

    return result;
}

bool IPCSharedMemory::processMatrixInBatches() {
    // wait for the parent signal that totalElements is ready
    sem_wait(sem_parent_to_child);
    // read totalElements from the beginning of shared memory
    if (sem_trywait(sem_exit) == 0) {
            std::cout << "SharedMem: Child process exiting...\n";
            return true; // exit the loop and thus the process
        }
    int totalElements;
    std::memcpy(&totalElements, shmAddr, sizeof(int));
    // Signal back to parent that totalElements has been read
    sem_post(sem_child_to_parent);
    int batchSizeInBytes = shmSize - sizeof(int); // adjusting for totalElements
    int batchSize = batchSizeInBytes / sizeof(CPP_TENSOR_DTYPE);

    char* batchPtr = static_cast<char*>(shmAddr) + sizeof(int); // offset by size of int

    for (int i = 0; i < totalElements;) {
        sem_wait(sem_parent_to_child); // wait for parent to signal batch is ready
        if (sem_trywait(sem_exit) == 0) {
            std::cout << "SharedMem: Child process exiting...\n";
            return true; // exit the loop and thus the process
        }

        int currentBatchSize = std::min(batchSize, totalElements - i);

        // process the current batch here - since we need a Tensor for processing,
        // we create one from the current batch in shared memory.
        torch::Tensor batch = torch::from_blob(batchPtr, {currentBatchSize}, torch::kFloat32).clone();

        // Example processing: Squaring the matrix
        torch::Tensor squaredBatch = batch.square();

        // If you need to write the processed batch back:
        std::memcpy(batchPtr, squaredBatch.data_ptr<CPP_TENSOR_DTYPE>(), currentBatchSize * sizeof(CPP_TENSOR_DTYPE));

        i += currentBatchSize; // Update for the next iteration

        sem_post(sem_child_to_parent); // Signal back to parent

        // Check if the exit semaphore was posted after processing a batch
        if (sem_trywait(sem_exit) == 0) {
            std::cout << "SharedMem: Child process exiting after processing batch...\n";
            return true; // Exit the loop and thus the process
        }
    }
    return false;
}

void IPCSharedMemory::exitSubprocess() {
    DEBUG_PRINT(1, "SharedMem: Parent process exiting...\n");
    // signal child process to exit
    sem_post(sem_exit);
    sem_post(sem_parent_to_child);
    
    DEBUG_PRINT(1, "SharedMem: Parent process signaled child to exit\n");

    // Wait for the child process to terminate
    if (childPid > 0) {
        waitpid(childPid, nullptr, 0);
    }
    DEBUG_PRINT(1, "SharedMem: Parent process waited for child to exit\n");
    // Cleanup shared memory and semaphore resources
    munmap(shmAddr, shmSize); // Unmap shared memory
    shm_unlink(shmName); // Unlink shared memory object
    sem_close(sem_parent_to_child); // Close semaphore
    sem_close(sem_child_to_parent); // Close semaphore
    sem_close(sem_exit); // Close semaphore
    sem_unlink("/sem_parent_to_child"); // Unlink semaphore
    sem_unlink("/sem_child_to_parent"); // Unlink semaphore
    sem_unlink("/sem_exit"); // Unlink semaphore

    // // terminate child process
    // if (childPid > 0) {
    //     ::kill(childPid, SIGTERM); // Send termination signal using the global scope resolution operator
    //     waitpid(childPid, nullptr, 0); // Wait for child process to terminate
    // }
}


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
        // exit(0);
    }
}

