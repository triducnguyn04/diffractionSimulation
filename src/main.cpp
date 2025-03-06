#include "debyescattering.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    DebyeScattering debye;
    float s_min = 1.0f, s_max = 10.0f;
    int n_points = 5000;
    std::string cif_file = "VO2.cif";
    if (!debye.loadFromCIF(cif_file, s_min, s_max, n_points)) {
        std::cerr << "Failed to load CIF file" << std::endl;
        return 1;
    }

    float ds = (s_max - s_min) / (n_points - 1);
    int n_replicas = 3;
    float r_max = 1000.0f;

    std::vector<float> s_values(n_points), intensities(n_points);
    for (int i = 0; i < n_points; i++) {
        s_values[i] = s_min + i * ds;
    }

    std::cout << "Number of atoms loaded from CIF: " << debye.getAtomCount() << std::endl;
    debye.calculateIntensity(n_replicas, r_max, s_values, intensities);

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
    std::chrono::duration<float> duration = end - start; // Changed to float
    std::cout << "Diffraction pattern saved to diffraction_pattern.txt" << std::endl;
    std::cout << "Total runtime: " << duration.count() << " seconds" << std::endl;

    return 0;
}
