# Specify the minimum version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(FEM VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the Gmsh package
# find_package(Gmsh REQUIRED)
set(GMSH_INCLUDE_DIR "/home/qinsx/FEM/gmsh/install/include")
set(GMSH_LIBRARY "/home/qinsx/FEM/gmsh/install/lib/libgmsh.so")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# # Add the include directory
include_directories(${GMSH_INCLUDE_DIR})

# Add the executable target
add_executable(${PROJECT_NAME} main.cpp)

# Link the Gmsh library
target_link_libraries(${PROJECT_NAME} ${GMSH_LIBRARY} Eigen3::Eigen)
# target_link_libraries(${PROJECT_NAME} Gmsh::gmsh Eigen3::Eigen)