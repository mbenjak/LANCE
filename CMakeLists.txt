cmake_minimum_required(VERSION 3.10)

# Project definition
project(LANCE VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Platform-specific settings
if(WIN32)
    message(STATUS "[ERROR] LANCE decoder not yet implemented for Windows...")
    return()
elseif(APPLE)
    message(STATUS "Architecture: Apple")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "Apple ARM64 detected")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Winline")
        set(USE_AVX2 OFF)
    else()
        message(STATUS "Apple x86_64 detected")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Winline")
        set(USE_AVX2 ON)
    endif()
elseif(UNIX)
    message(STATUS "Architecture: Linux")
    # Base flags without optimization
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Winline -DCCDEC_EXE -DCCDECAPI_AVX2_OPTIONAL")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -Wall -Winline -DCCDEC_EXE -DCCDECAPI_CPU")

    # Set optimization based on build type
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")
    # Removed static linking as it may not be available on all systems
    # set(CMAKE_EXE_LINKER_FLAGS "-static")
    set(USE_AVX2 ON)
endif()

# Define source directory
set(CPP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/coolchic/cpp)

# Define source files
set(COMMON_SOURCES
    ${CPP_SOURCE_DIR}/ccdecapi.cpp
    ${CPP_SOURCE_DIR}/cc-bitstream.cpp
    ${CPP_SOURCE_DIR}/cc-contexts.cpp
    ${CPP_SOURCE_DIR}/frame-memory.cpp
    ${CPP_SOURCE_DIR}/cc-frame-decoder.cpp
    ${CPP_SOURCE_DIR}/arm_cpu.cpp
    ${CPP_SOURCE_DIR}/sparm_cpu.cpp
    ${CPP_SOURCE_DIR}/syn_cpu.cpp
    ${CPP_SOURCE_DIR}/resampler.cpp
    ${CPP_SOURCE_DIR}/BitStream.cpp
    ${CPP_SOURCE_DIR}/TDecBinCoderCABAC.cpp
    ${CPP_SOURCE_DIR}/Contexts.cpp
)

set(AVX2_SOURCES
    ${CPP_SOURCE_DIR}/arm_avx2.cpp
    ${CPP_SOURCE_DIR}/ups_cpu.cpp
    ${CPP_SOURCE_DIR}/ups_avx2.cpp
    ${CPP_SOURCE_DIR}/syn_avx2.cpp
)

# Create executable
add_executable(ccdec ${COMMON_SOURCES})

# Add AVX2 sources if supported
if(USE_AVX2)
    target_sources(ccdec PRIVATE ${AVX2_SOURCES})
    
    # Set AVX2 compile flags for specific files
    set_source_files_properties(${CPP_SOURCE_DIR}/arm_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
    set_source_files_properties(${CPP_SOURCE_DIR}/ups_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
    set_source_files_properties(${CPP_SOURCE_DIR}/syn_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2")
endif()

# Include directories
target_include_directories(ccdec PRIVATE ${CPP_SOURCE_DIR})

# Set output directory
set_target_properties(ccdec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration summary
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "AVX2 Support: ${USE_AVX2}")
message(STATUS "Source Directory: ${CPP_SOURCE_DIR}")
message(STATUS "Output Directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=============================")

# Installation
install(TARGETS ccdec RUNTIME DESTINATION bin)
