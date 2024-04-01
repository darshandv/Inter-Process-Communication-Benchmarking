cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(IPCBenchmarkProject)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Optionally, allow the user to specify the path to libtorch
set(LIBTORCH_PATH "$ENV{LIBTORCH_PATH}" CACHE PATH "Path to libtorch")
if(NOT LIBTORCH_PATH)
    set(LIBTORCH_PATH "${CMAKE_SOURCE_DIR}/libtorch")
endif()

# Add the current directory to find Torch package
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
find_package(Torch REQUIRED)

# Add include directory for header files
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/include/IPC")
# Note: ${TORCH_INCLUDE_DIRS} is automatically included through target_link_libraries

# Glob source files from src directory
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")

# Specify the executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Link against libtorch
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
