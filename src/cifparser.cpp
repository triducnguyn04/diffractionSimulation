#include "cifparser.h"
#include <iostream>
#include <stdexcept>

bool CIFParser::loadFromCIF(const std::string& filename, 
                            aligned_vector<float>& x, 
                            aligned_vector<float>& y, 
                            aligned_vector<float>& z, 
                            aligned_vector<std::string>& elements, 
                            gemmi::UnitCell& cell) {
    try {
        gemmi::cif::Document doc = gemmi::cif::read_file(filename);
        gemmi::cif::Block& block = doc.sole_block();

        // Parse unit cell parameters
        cell.a = to_float(block.find_value("_cell_length_a"));
        cell.b = to_float(block.find_value("_cell_length_b"));
        cell.c = to_float(block.find_value("_cell_length_c"));
        cell.alpha = to_float(block.find_value("_cell_angle_alpha"));
        cell.beta = to_float(block.find_value("_cell_angle_beta"));
        cell.gamma = to_float(block.find_value("_cell_angle_gamma"));

        std::cout << "Unit cell: a=" << cell.a << ", b=" << cell.b << ", c=" << cell.c
                  << ", alpha=" << cell.alpha << ", beta=" << cell.beta << ", gamma=" << cell.gamma << "\n";

        // Parse atom site data
        auto type_col = block.find_loop("_atom_site_type_symbol");
        auto x_col = block.find_loop("_atom_site_fract_x");
        auto y_col = block.find_loop("_atom_site_fract_y");
        auto z_col = block.find_loop("_atom_site_fract_z");

        if (!type_col || !x_col || !y_col || !z_col) {
            std::cerr << "Missing atom site columns in CIF" << std::endl;
            return false;
        }

        size_t n_atoms = x_col.length();
        std::cout << "Number of atoms loaded from CIF: " << n_atoms << "\n";

        x.resize(n_atoms);
        y.resize(n_atoms);
        z.resize(n_atoms);
        elements.resize(n_atoms);

        float beta_rad = cell.beta * M_PI / 180.0f;
        float cos_beta = std::cos(beta_rad);
        float sin_beta = std::sin(beta_rad);
        float ax = static_cast<float>(cell.a);
        float by = static_cast<float>(cell.b);
        float cz_cos_beta = static_cast<float>(cell.c * cos_beta);
        float cz_sin_beta = static_cast<float>(cell.c * sin_beta);

        for (size_t i = 0; i < n_atoms; ++i) {
            std::string element = type_col[i];
            float frac_x = to_float(&x_col[i]);
            float frac_y = to_float(&y_col[i]);
            float frac_z = to_float(&z_col[i]);

            std::cout << "Atom " << i << ": " << element << ", frac=(" << frac_x << ", " << frac_y << ", " << frac_z << ")";

            float cart_x = frac_x * ax + frac_z * cz_cos_beta;
            float cart_y = frac_y * by;
            float cart_z = frac_z * cz_sin_beta;

            std::cout << ", cart=(" << cart_x << ", " << cart_y << ", " << cart_z << ")\n";

            x[i] = cart_x;
            y[i] = cart_y;
            z[i] = cart_z;
            elements[i] = element;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading CIF: " << e.what() << std::endl;
        return false;
    }
}

float CIFParser::to_float(const std::string* str) const {
    if (!str || str->empty()) {
        std::cerr << "Warning: Empty or null CIF value\n";
        return 0.0f;
    }
    std::string cleaned = *str;
    size_t pos = cleaned.find_first_of("()");
    if (pos != std::string::npos) cleaned = cleaned.substr(0, pos);
    return std::stof(cleaned);
}
