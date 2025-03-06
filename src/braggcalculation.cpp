#include "braggcalculation.h"
#include <cmath>
#include <iostream>
#include <complex>

BraggCalculation::BraggCalculation() {
    clear();
}

void BraggCalculation::clear() {
    x.clear();
    y.clear();
    z.clear();
    f.clear();
    elements.clear();
}

double BraggCalculation::getScatteringFactor(const std::string& element, double s) const {
    double s2 = s * s / 4.0; // s/2 for sinθ/λ convention
    if (element == "V") return 23.0 * std::exp(-0.1 * s2); // Vanadium
    if (element == "O") return 8.0 * std::exp(-0.2 * s2);  // Oxygen
    return 1.0; // Default
}

bool BraggCalculation::loadFromCIF(const std::string& filename) {
    try {
        gemmi::cif::Document doc = gemmi::cif::read_file(filename);
        gemmi::cif::Block& block = doc.sole_block(); // Fixed syntax here
        clear();

        auto to_double = [](const std::string* str) -> double {
            if (!str || str->empty()) throw std::runtime_error("Invalid CIF value");
            return std::atof(str->c_str());
        };

        cell.a = to_double(block.find_value("_cell_length_a"));
        cell.b = to_double(block.find_value("_cell_length_b"));
        cell.c = to_double(block.find_value("_cell_length_c"));
        cell.alpha = to_double(block.find_value("_cell_angle_alpha"));
        cell.beta = to_double(block.find_value("_cell_angle_beta"));
        cell.gamma = to_double(block.find_value("_cell_angle_gamma"));

        auto atom_loop = block.find_loop("_atom_site_label");
        if (atom_loop.length() == 0) {
            std::cerr << "No atom site data found in CIF" << std::endl;
            return false;
        }

        auto type_col = block.find_loop("_atom_site_type_symbol");
        auto x_col = block.find_loop("_atom_site_fract_x");
        auto y_col = block.find_loop("_atom_site_fract_y");
        auto z_col = block.find_loop("_atom_site_fract_z");

        if (!type_col || !x_col || !y_col || !z_col) {
            std::cerr << "Missing atom site columns in CIF" << std::endl;
            return false;
        }

        size_t n_atoms = x_col.length();
        for (size_t i = 0; i < n_atoms; ++i) {
            std::string element = type_col[i];
            double frac_x = to_double(&x_col[i]);
            double frac_y = to_double(&y_col[i]);
            double frac_z = to_double(&z_col[i]);

            gemmi::Position pos = cell.orthogonalize(gemmi::Fractional(frac_x, frac_y, frac_z));
            x.push_back(pos.x);
            y.push_back(pos.y);
            z.push_back(pos.z);
            elements.push_back(element);
            f.push_back(getScatteringFactor(element, 0.0));
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading CIF: " << e.what() << std::endl;
        return false;
    }
}

double BraggCalculation::calculateBraggIntensity(int h, int k, int l) const {
    std::complex<double> F(0.0, 0.0);
    int n = x.size();

    // Calculate s = 1/d_hkl using GEMMI's built-in method
    double one_over_d2 = cell.calculate_1_d2(gemmi::Miller{h, k, l});
    double s = std::sqrt(one_over_d2); // s = 1/d_hkl

    for (int i = 0; i < n; i++) {
        double fi = getScatteringFactor(elements[i], s);
        gemmi::Fractional frac = cell.fractionalize(gemmi::Position(x[i], y[i], z[i]));
        double phase = 2 * M_PI * (h * frac.x + k * frac.y + l * frac.z);
        F += fi * std::complex<double>(std::cos(phase), -std::sin(phase));
    }
    return std::norm(F); // Intensity = |F|^2
}
