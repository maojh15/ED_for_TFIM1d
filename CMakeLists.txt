cmake_minimum_required(VERSION 3.10)
project(TFIM1dExactDiag)
add_executable(main main.cpp)
add_library(TFIM1d TFIM1d.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(thirdPartLibInclude "D:/MyProject/thirdPartLibCpp/include")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(thirdPartLibInclude "/home/maojh/thirdPartLibCpp/include")
    target_link_libraries(TFIM1d pthread)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(MPI REQUIRED)

target_include_directories(TFIM1d PUBLIC ${thirdPartLibInclude})
target_link_libraries(TFIM1d arpack++ arpack lapack superlu ${MPI_LIBRARIES})
target_link_libraries(main ${MPI_LIBRARIES})
target_include_directories(main PUBLIC  ${Python3_INCLUDE_DIRS})
target_link_libraries(main TFIM1d  ${Python3_LIBRARIES})

install(TARGETS main
        DESTINATION ${CMAKE_SOURCE_DIR}/bin)
