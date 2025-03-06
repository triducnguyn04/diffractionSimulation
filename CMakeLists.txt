cmake_minimum_required(VERSION 3.10)
project(diffractionSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch gemmi
include(FetchContent)
FetchContent_Declare(
    gemmi
    GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
    GIT_TAG        v0.6.6
)
FetchContent_GetProperties(gemmi)
if(NOT gemmi_POPULATED)
    FetchContent_Populate(gemmi)
    add_subdirectory(${gemmi_SOURCE_DIR} ${gemmi_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Find OpenMP (use Homebrew's libomp on macOS)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    # Manual fallback for Homebrew libomp on macOS
    set(OPENMP_INCLUDE_DIR "/opt/homebrew/opt/libomp/include")
    set(OPENMP_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    if(EXISTS "${OPENMP_INCLUDE_DIR}/omp.h" AND EXISTS "${OPENMP_LIBRARY}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "libomp")
        set(OpenMP_libomp_LIBRARY "${OPENMP_LIBRARY}")
        set(OpenMP_CXX_FOUND TRUE)
    else()
        message(WARNING "OpenMP not found. Parallelization will be disabled.")
    endif()
endif()

# Zlib (manually specified)
find_library(ZLIB_LIBRARY
    NAMES z zlib libz
    PATHS /opt/homebrew/Cellar/zlib/1.3.1/lib
    NO_DEFAULT_PATH
)
if(NOT ZLIB_LIBRARY)
    message(FATAL_ERROR "Zlib not found. Install it with 'brew install zlib'.")
endif()

# BLAS
find_package(BLAS REQUIRED)
if(NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS not found. Install it with 'brew install openblas'.")
endif()
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "BLAS include dirs: ${BLAS_INCLUDE_DIRS}")

# GSL
find_package(GSL REQUIRED)

# FFTW (single-precision)
find_library(FFTW3F_LIBRARY
    NAMES fftw3f
    PATHS /opt/homebrew/Cellar/fftw/3.3.10_2/lib
    NO_DEFAULT_PATH
)
if(NOT FFTW3F_LIBRARY)
    message(FATAL_ERROR "FFTW (single-precision) not found. Install it with 'brew install fftw'.")
endif()
set(FFTW3F_INCLUDE_DIR "/opt/homebrew/Cellar/fftw/3.3.10_2/include")
message(STATUS "FFTW3F library: ${FFTW3F_LIBRARY}")
message(STATUS "FFTW3F include dir: ${FFTW3F_INCLUDE_DIR}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OPENMP_INCLUDE_DIR}
    ${GSL_INCLUDE_DIRS}
    /opt/homebrew/Cellar/zlib/1.3.1/include
    /opt/homebrew/Cellar/openblas/0.3.29/include
    ${FFTW3F_INCLUDE_DIR}
)

# Add executable
add_executable(diffractionSimulation
    src/fitting.cpp
    src/debyescattering.cpp
    src/braggcalculation.cpp
    src/main.cpp
)

# Check for NEON support and add compiler flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_NATIVE)
if(COMPILER_SUPPORTS_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(diffractionSimulation PRIVATE
        -O3                  # Aggressive optimization
        -march=native        # Tune for native architecture (includes NEON on ARM)
        -mfpu=neon           # Explicitly enable NEON FPU
        -ffast-math          # Relax floating-point strictness for SIMD
    )
    if(OpenMP_CXX_FOUND)
        target_compile_options(diffractionSimulation PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(diffractionSimulation PRIVATE ${OpenMP_CXX_LIBRARIES})
    endif()
    message(STATUS "NEON and SIMD support enabled with -O3 -march=native -mfpu=neon")
else()
    message(WARNING "NEON not supported or not an ARM64 platform. Falling back to generic optimization.")
    target_compile_options(diffractionSimulation PRIVATE -O3)
    if(OpenMP_CXX_FOUND)
        target_compile_options(diffractionSimulation PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(diffractionSimulation PRIVATE ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# Link libraries
target_link_libraries(diffractionSimulation
    PRIVATE
    GSL::gsl
    GSL::gslcblas
    gemmi::gemmi_cpp
    ${ZLIB_LIBRARY}
    ${BLAS_LIBRARIES}
    ${FFTW3F_LIBRARY}
)

# Set output directory
set_target_properties(diffractionSimulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(diffractionSimulation PRIVATE -O3)
endif()

message(STATUS "Building diffractionSimulation in ${CMAKE_BUILD_TYPE} mode")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP include directory: ${OPENMP_INCLUDE_DIR}")
message(STATUS "OpenMP library: ${OPENMP_LIBRARY}")
message(STATUS "ZLIB library: ${ZLIB_LIBRARY}")
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
