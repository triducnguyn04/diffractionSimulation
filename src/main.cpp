#include "debyescattering.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <cstring>
#include <optional>
#include <filesystem>

namespace fs = std::filesystem;

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

// Helper function to parse flag values (e.g., -x=3 -y=3 -z=3)
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

// Helper function to process a single CIF file
void processCifFile(const std::string& filename, int nx, int ny, int nz, bool useCuda, 
                    float s_min, float s_max, int n_points, float r_max) {
    DebyeScattering debye;
    bool loadSuccess = debye.loadFromCIF(filename, s_min, s_max, n_points);
    if (!loadSuccess) {
        std::cerr << "Failed to load CIF data from " << filename << std::endl;
        return;
    }

    float ds = (s_max - s_min) / (n_points - 1);
    int n_replicas = std::max({nx, ny, nz});
    std::vector<float> s_values(n_points), intensities(n_points);
    for (int i = 0; i < n_points; i++) {
        s_values[i] = s_min + i * ds;
    }

    std::cout << "Processing " << filename << " with " << debye.getAtomCount() << " atoms\n";

    if (useCuda) {
        std::cout << "CUDA mode is enabled, but no CUDA implementation is provided yet.\n";
        debye.calculateIntensity(n_replicas, r_max, s_values, intensities); // Placeholder
    } else {
        debye.calculateIntensity(n_replicas, r_max, s_values, intensities);
    }

    // Generate output filename (e.g., "VO2.cif" -> "VO2.txt")
    std::string out_filename = fs::path(filename).stem().string() + ".txt";
    std::ofstream out(out_filename);
    if (!out) {
        std::cerr << "Failed to open output file " << out_filename << std::endl;
        return;
    }
    out << "s,intensity\n";
    for (int i = 0; i < n_points; i++) {
        out << s_values[i] << "," << intensities[i] << "\n";
    }
    out.close();
    std::cout << "Diffraction pattern saved to " << out_filename << std::endl;
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();

    // Default values
    int nx = 3, ny = 3, nz = 3; // Default box dimensions
    bool useCuda = false;
    std::string provided_filename;

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
            provided_filename = argv[i];
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
        std::cerr << "Warning: -cuda flag provided, but CUDA is not available. Falling back to CPU mode.\n";
        useCuda = false;
    }

    std::cout << "Running in " << (useCuda ? "CUDA" : "CPU") << " mode\n";
    std::cout << "Box dimensions: nx=" << nx << ", ny=" << ny << ", nz=" << nz << "\n";

    // Simulation parameters
    float s_min = 1.0f, s_max = 12.0f;
    int n_points = 1000;
    float r_max = 1000.0f;

    // Collect all CIF files in the current directory
    std::vector<std::string> cif_files;
    if (!provided_filename.empty()) {
        cif_files.push_back(provided_filename); 
    } else {
        for (const auto& entry : fs::directory_iterator(".")) {
            if (entry.path().extension() == ".cif") {
                cif_files.push_back(entry.path().string());
            }
        }
        if (cif_files.empty()) {
            std::cerr << "No .cif files found in the current directory\n";
            return 1;
        }
    }

    std::cout << "Found " << cif_files.size() << " CIF file(s) to process\n";

    // Process each CIF file
    for (const auto& filename : cif_files) {
        auto file_start = std::chrono::high_resolution_clock::now();
        processCifFile(filename, nx, ny, nz, useCuda, s_min, s_max, n_points, r_max);
        auto file_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> file_duration = file_end - file_start;
        std::cout << "Processed " << filename << " in " << file_duration.count() << " seconds\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> total_duration = end - start;
    std::cout << "Total runtime for all files: " << total_duration.count() << " seconds\n";

    return 0;
}
