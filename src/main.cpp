#include "debyescattering.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <cstring>
#include <optional>

// Simple function to check if CUDA is available (placeholder)
bool isCudaAvailable() {
#ifdef __NVCC__
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount > 0;
#else
    return false;
#endif
}

// Helper function to parse flag values (e.g., -x=3)
std::optional<int> parseDimensionFlag(const char* arg, const char* flag) {
    size_t flagLen = std::strlen(flag);
    if (std::strncmp(arg, flag, flagLen) == 0 && arg[flagLen] == '=') {
        const char* value = arg + flagLen + 1;
        try {
            return std::stoi(value);
        } catch (const std::exception& e) {
            std::cerr << "Invalid value for " << flag << ": " << value << std::endl;
            return std::nullopt;
        }
    }
    return std::nullopt;
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    // Default values
    int nx = 3, ny = 3, nz = 3; // Default box dimensions
    std::string default_filename = "VO2.cif";
    bool useCuda = false;
    bool filenameProvided = false;
    std::string filename;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-cuda") == 0) {
            useCuda = true;
        } else if (auto x = parseDimensionFlag(argv[i], "-x")) {
            nx = *x;
        } else if (auto y = parseDimensionFlag(argv[i], "-y")) {
            ny = *y;
        } else if (auto z = parseDimensionFlag(argv[i], "-z")) {
            nz = *z;
        } else if (argv[i][0] != '-') {
            filename = argv[i];
            filenameProvided = true;
        } else {
            std::cerr << "Unknown or malformed argument: " << argv[i] << std::endl;
            return 1;
        }
    }

    // Validate dimensions
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        std::cerr << "Box dimensions must be positive: nx=" << nx << ", ny=" << ny << ", nz=" << nz << std::endl;
        return 1;
    }

    // Check CUDA availability
    bool cudaAvailable = isCudaAvailable();
    if (useCuda && !cudaAvailable) {
        std::cerr << "Warning: -cuda flag provided, but CUDA is not available. Falling back to CPU mode." << std::endl;
        useCuda = false;
    }

    std::cout << "Running in " << (useCuda ? "CUDA" : "CPU") << " mode" << std::endl;
    std::cout << "Box dimensions: nx=" << nx << ", ny=" << ny << ", nz=" << nz << std::endl;

    DebyeScattering debye;
    float s_min = 1.0f, s_max = 10.0f;
    int n_points = 5000;

    // Load CIF data
    bool loadSuccess = false;
    if (filenameProvided) {
        std::cout << "Loading from provided file: " << filename << std::endl;
        loadSuccess = debye.loadFromCIF(filename, s_min, s_max, n_points);
    } else {
        if (CIFParser::hasLoadedData) {
            std::cout << "Loading from previously stored CIF data" << std::endl;
            aligned_vector<float> x, y, z;
            aligned_vector<std::string> elements;
            gemmi::UnitCell cell;
            loadSuccess = CIFParser::loadStoredData(x, y, z, elements, cell);
            if (loadSuccess) {
                debye.setData(x, y, z, elements, cell, s_min, s_max, n_points);
            }
        } else {
            std::cout << "No filename provided and no stored data, loading default: " << default_filename << std::endl;
            loadSuccess = debye.loadFromCIF(default_filename, s_min, s_max, n_points);
        }
    }

    if (!loadSuccess) {
        std::cerr << "Failed to load CIF data" << std::endl;
        return 1;
    }

    float ds = (s_max - s_min) / (n_points - 1);
    float r_max = 1000.0f;
    int n_replicas = std::max({nx, ny, nz});

    std::vector<float> s_values(n_points), intensities(n_points);
    for (int i = 0; i < n_points; i++) {
        s_values[i] = s_min + i * ds;
    }

    std::cout << "Number of atoms loaded: " << debye.getAtomCount() << std::endl;

    if (useCuda) {
        std::cout << "CUDA mode is enabled, but no CUDA implementation is provided yet." << std::endl;
        debye.calculateIntensity(n_replicas, r_max, s_values, intensities); // Placeholder
    } else {
        debye.calculateIntensity(n_replicas, r_max, s_values, intensities);
    }

    std::ofstream out("diffraction_pattern.txt");
    if (!out) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }
    out << "s,intensity\n";
    for (int i = 0; i < n_points; i++) {
        out << s_values[i] << "," << intensities[i] << "\n";
    }
    out.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Diffraction pattern saved to diffraction_pattern.txt" << std::endl;
    std::cout << "Total runtime: " << duration.count() << " seconds" << std::endl;

    return 0;
}
