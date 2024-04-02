# IPC Benchmark Project

The IPC Benchmark Project demonstrates efficient inter-process communication (IPC) using various mechanisms such as pipes, shared memory, and sockets in C++. It benchmarks the performance of these IPC methods in the context of matrix operations, particularly focusing on the squaring of matrices.

## Features

- **IPC Mechanisms**: Implements pipes, shared memory, and sockets for IPC.
- **Matrix Operations**: Generates random matrices and performs squaring operations.
- **Benchmarking**: Compares the performance of different IPC methods in terms of processing rate (in MBps).
- **LibTorch Integration**: Utilizes LibTorch for matrix operations to leverage hardware acceleration.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- C++ Compiler: GCC or Clang supporting C++17.
- [CMake](https://cmake.org/download/): Version 3.5 or higher.
- [LibTorch](https://pytorch.org/get-started/locally/): Download and extract the LibTorch library.

Certainly! Here's an addition to the `README.md` to include instructions on setting up the MPI environment variable, which is essential for the project's operation. This assumes you're referring to the setup involving `DYLD_LIBRARY_PATH` or similar environment variables that were discussed earlier. I've added this under the "Installation" section:

## Installation

To install the IPC Benchmark Project, follow these steps:

1. Clone the repository:
   ```sh
   git clone <yet to be done> IPCBenchmarkProject
   cd IPCBenchmarkProject
   ```

2. Set the `LIBTORCH_PATH` environment variable to the path of your LibTorch installation:
   ```sh
   export LIBTORCH_PATH=/path/to/libtorch
   ```

3. If you're using OpenMP (`libomp`) with LibTorch on macOS, ensure that your dynamic linker can find `libomp.dylib`. Set the `DYLD_LIBRARY_PATH` environment variable. Please set it according to your installation paths. Here is an example:
   ```sh
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
    export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
   ```
   Note: Adjust the path `/usr/local/opt/libomp/lib` as necessary based on where `libomp.dylib` is located on your system. You can find this location by running `brew info libomp` if you've installed `libomp` via Homebrew.

4. Build the project using CMake:
   ```sh
   mkdir build && cd build
   cmake ..
   make
   ```

## Usage

After building the project, you can run the benchmark by executing:

```sh
./IPCBenchmarkProject
```

The program will output the results of the benchmarking, comparing the performance of IPC mechanisms.

This snippet assumes that `libomp` is required for your project, which is a common dependency when using LibTorch, especially if it's configured to use OpenMP for parallelism. The `DYLD_LIBRARY_PATH` environment variable is specifically relevant to macOS users. If your project or its dependencies do not use OpenMP, or if you're targeting a different operating system, you may need to adjust these instructions accordingly.

The program will output the results of the benchmarking, comparing the performance of IPC mechanisms.

## Contributing

Contributions to the IPC Benchmark Project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.



## Acknowledgments

- Thank you to the [PyTorch](https://pytorch.org/) team for providing the LibTorch library.
- This project was inspired by the need to understand and benchmark IPC mechanisms in modern C++ applications.

## Contact

If you have any questions or feedback, please contact [Darshan D Vishwanath] at my email dvishwan@usc.edu

