#include "debyescattering.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <arm_neon.h>

// Ensure 16-byte alignment for NEON
#if defined(__GNUC__) || defined(__clang__)
#define ALIGNED_ALLOC(n) std::vector<float, std::allocator<float>> __attribute__((aligned(16)))
#else
#define ALIGNED_ALLOC(n) std::vector<float, std::allocator<float>>
#endif

DebyeScattering::DebyeScattering() {
    clear();
}

void DebyeScattering::clear() {
    x.clear();
    y.clear();
    z.clear();
    elements.clear();
    s_min = 0.0f;
    s_max = 0.0f;
    ds = 0.0f;
    n_points = 0;
    f_V.clear();
    f_O.clear();
    s_values_.clear();
}

float DebyeScattering::getScatteringFactor(const std::string& element, float s) const {
    float s2 = s * s;
    if (element == "V") {
        return 2.05710f * std::exp(-102.478f * s2) + 
               2.07030f * std::exp(-26.8938f * s2) + 
               7.35110f * std::exp(-0.438500f * s2) + 
               10.2971f * std::exp(-6.86570f * s2) + 
               1.21990f;
    } 
    if (element == "O") {
        return 0.867f * std::exp(-32.9089f * s2) + 
               1.54630f * std::exp(-0.323900f * s2) + 
               2.28680f * std::exp(-5.70110f * s2) + 
               3.04850f * std::exp(-13.2771f * s2) + 
               0.2508f;
    } 
    return 1.0f;
}

bool DebyeScattering::loadFromCIF(const std::string& filename, float s_min_input, 
                                 float s_max_input, int n_points_input) {
    try {
        gemmi::cif::Document doc = gemmi::cif::read_file(filename);
        gemmi::cif::Block& block = doc.sole_block();
        clear();

        auto to_float = [](const std::string* str) -> float {
            if (!str || str->empty()) {
                std::cerr << "Warning: Empty or null CIF value\n";
                return 0.0f;
            }
            std::string cleaned = *str;
            size_t pos = cleaned.find_first_of("()");
            if (pos != std::string::npos) cleaned = cleaned.substr(0, pos);
            return std::stof(cleaned);
        };

        cell.a = to_float(block.find_value("_cell_length_a"));
        cell.b = to_float(block.find_value("_cell_length_b"));
        cell.c = to_float(block.find_value("_cell_length_c"));
        cell.alpha = to_float(block.find_value("_cell_angle_alpha"));
        cell.beta = to_float(block.find_value("_cell_angle_beta"));
        cell.gamma = to_float(block.find_value("_cell_angle_gamma"));

        std::cout << "Unit cell: a=" << cell.a << ", b=" << cell.b << ", c=" << cell.c
                  << ", alpha=" << cell.alpha << ", beta=" << cell.beta << ", gamma=" << cell.gamma << "\n";

        float beta_rad = cell.beta * M_PI / 180.0f;
        float cos_beta = std::cos(beta_rad);
        float sin_beta = std::sin(beta_rad);
        float ax = static_cast<float>(cell.a);  // Cast to float
        float ay = 0.0f;
        float az = 0.0f;
        float bx = 0.0f;
        float by = static_cast<float>(cell.b);  // Cast to float
        float bz = 0.0f;
        float cx = static_cast<float>(cell.c * cos_beta);  // Cast to float
        float cy = 0.0f;
        float cz = static_cast<float>(cell.c * sin_beta);  // Cast to float

        auto type_col = block.find_loop("_atom_site_type_symbol");
        auto x_col = block.find_loop("_atom_site_fract_x");
        auto y_col = block.find_loop("_atom_site_fract_y");
        auto z_col = block.find_loop("_atom_site_fract_z");

        if (!type_col || !x_col || !y_col || !z_col) {
            std::cerr << "Missing atom site columns in CIF" << std::endl;
            return false;
        }

        size_t n_atoms = x_col.length();
        std::cout << "Number of atoms in CIF: " << n_atoms << "\n";

        x.reserve(n_atoms);
        y.reserve(n_atoms);
        z.reserve(n_atoms);
        elements.reserve(n_atoms);

        for (size_t i = 0; i < n_atoms; ++i) {
            std::string element = type_col[i];
            float frac_x = to_float(&x_col[i]);
            float frac_y = to_float(&y_col[i]);
            float frac_z = to_float(&z_col[i]);

            std::cout << "Atom " << i << ": " << element << ", frac=(" << frac_x << ", " << frac_y << ", " << frac_z << ")";

            float cart_x = frac_x * ax + frac_y * bx + frac_z * cx;
            float cart_y = frac_x * ay + frac_y * by + frac_z * cy;
            float cart_z = frac_x * az + frac_y * bz + frac_z * cz;

            std::cout << ", cart=(" << cart_x << ", " << cart_y << ", " << cart_z << ")\n";

            x.push_back(cart_x);
            y.push_back(cart_y);
            z.push_back(cart_z);
            elements.push_back(element);
        }

        s_min = s_min_input;
        s_max = s_max_input;
        n_points = n_points_input;
        ds = (s_max - s_min) / (n_points - 1);
        f_V.resize(n_points);
        f_O.resize(n_points);
        s_values_.resize(n_points);

        std::cout << "Precomputing scattering factors for V and O...\n";
        for (int i = 0; i < n_points; i += 4) {  // Vectorize in steps of 4
            if (i + 3 < n_points) {
                float32x4_t s_vec = {s_min + i * ds, s_min + (i + 1) * ds, 
                                    s_min + (i + 2) * ds, s_min + (i + 3) * ds};
                vst1q_f32(&s_values_[i], s_vec);

                float32x4_t s2_vec = vmulq_f32(s_vec, s_vec);

                // Scattering factors for "V" (vectorized)
                float32x4_t v_term1 = vmulq_n_f32(vdupq_n_f32(2.05710f), std::exp(-102.478f * vgetq_lane_f32(s2_vec, 0)));
                float32x4_t v_term2 = vmulq_n_f32(vdupq_n_f32(2.07030f), std::exp(-26.8938f * vgetq_lane_f32(s2_vec, 1)));
                float32x4_t v_term3 = vmulq_n_f32(vdupq_n_f32(7.35110f), std::exp(-0.438500f * vgetq_lane_f32(s2_vec, 2)));
                float32x4_t v_term4 = vmulq_n_f32(vdupq_n_f32(10.2971f), std::exp(-6.86570f * vgetq_lane_f32(s2_vec, 3)));
                float32x4_t v_const = vdupq_n_f32(1.21990f);
                float32x4_t f_V_vec = vaddq_f32(vaddq_f32(vaddq_f32(v_term1, v_term2), vaddq_f32(v_term3, v_term4)), v_const);
                vst1q_f32(&f_V[i], f_V_vec);

                // Scattering factors for "O" (vectorized)
                float32x4_t o_term1 = vmulq_n_f32(vdupq_n_f32(0.867f), std::exp(-32.9089f * vgetq_lane_f32(s2_vec, 0)));
                float32x4_t o_term2 = vmulq_n_f32(vdupq_n_f32(1.54630f), std::exp(-0.323900f * vgetq_lane_f32(s2_vec, 1)));
                float32x4_t o_term3 = vmulq_n_f32(vdupq_n_f32(2.28680f), std::exp(-5.70110f * vgetq_lane_f32(s2_vec, 2)));
                float32x4_t o_term4 = vmulq_n_f32(vdupq_n_f32(3.04850f), std::exp(-13.2771f * vgetq_lane_f32(s2_vec, 3)));
                float32x4_t o_const = vdupq_n_f32(0.2508f);
                float32x4_t f_O_vec = vaddq_f32(vaddq_f32(vaddq_f32(o_term1, o_term2), vaddq_f32(o_term3, o_term4)), o_const);
                vst1q_f32(&f_O[i], f_O_vec);
            } else {
                // Cleanup for remaining elements
                for (; i < n_points; i++) {
                    float s = s_min + i * ds;
                    f_V[i] = getScatteringFactor("V", s);
                    f_O[i] = getScatteringFactor("O", s);
                    s_values_[i] = s;
                }
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading CIF: " << e.what() << std::endl;
        return false;
    }
}

size_t DebyeScattering::calculateMultiplicity(int nx, int ny, int nz, int n_replicas) const {
    int doubled_replicas = 2 * n_replicas;
    int min_x = std::max(-n_replicas, -n_replicas - nx);
    int max_x = std::min(n_replicas, n_replicas - nx);
    int min_y = std::max(-n_replicas, -n_replicas - ny);
    int max_y = std::min(n_replicas, n_replicas - ny);
    int min_z = std::max(-n_replicas, -n_replicas - nz);
    int max_z = std::min(n_replicas, n_replicas - nz);
    return (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
}

size_t DebyeScattering::getAtomCount() const {
    return x.size();
}

void DebyeScattering::calculateIntensity(int n_replicas, float r_max, 
                                        std::vector<float>& s_values, 
                                        std::vector<float>& intensities) {
    int n = x.size();
    intensities.assign(n_points, 0.0f);
    s_values = s_values_; // Use precomputed s values

    std::cout << "Calculating intensity from central unit cell with " << n 
              << " atoms, n_replicas = " << n_replicas << "\n";

    // Precompute trigonometric values
    float beta_rad = cell.beta * M_PI / 180.0f;
    float cos_beta = std::cos(beta_rad);
    float sin_beta = std::sin(beta_rad);

    // Define replica shift structure
    struct ReplicaShift {
        float shift_x, shift_y, shift_z;
        size_t multiplicity;
    };
    std::vector<ReplicaShift> shifts;
    shifts.reserve((2 * n_replicas + 1) * (2 * n_replicas + 1) * (2 * n_replicas + 1));

    // Precompute replica shifts with explicit casts
    for (int nx = -2*n_replicas; nx <= 2*n_replicas; nx++) {
        for (int ny = -2*n_replicas; ny <= 2*n_replicas; ny++) {
            for (int nz = -2*n_replicas; nz <= 2*n_replicas; nz++) {
                shifts.push_back({
                    static_cast<float>(nx * cell.a + nz * cell.c * cos_beta),
                    static_cast<float>(ny * cell.b),
                    static_cast<float>(nz * cell.c * sin_beta),
                    calculateMultiplicity(nx, ny, nz, n_replicas)
                });
            }
        }
    }

    // Precompute element indices for faster lookup (0 = "V", 1 = "O")
    std::vector<int> elem_indices(n);
    for (int i = 0; i < n; i++) {
        elem_indices[i] = (elements[i] == "V") ? 0 : 1;
    }

    // Single-threaded intensity calculation with vectorization
    ALIGNED_ALLOC(n_points) local_intensities(n_points, 0.0f);

    for (size_t s = 0; s < shifts.size(); s++) {
        float shift_x = shifts[s].shift_x;
        float shift_y = shifts[s].shift_y;
        float shift_z = shifts[s].shift_z;
        float32x4_t multiplicity_vec = vdupq_n_f32(static_cast<float>(shifts[s].multiplicity));

        for (int i = 0; i < n; i++) {
            int ei = elem_indices[i];
            float xi = x[i], yi = y[i], zi = z[i];
            float32x4_t* fi_vec = (ei == 0) ? (float32x4_t*)f_V.data() : (float32x4_t*)f_O.data();

            for (int j = 0; j < n; j++) {
                int ej = elem_indices[j];
                float dx = xi - (x[j] + shift_x);
                float dy = yi - (y[j] + shift_y);
                float dz = zi - (z[j] + shift_z);
                float rij = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (rij <= r_max) {
                    bool is_self = (i == j && shift_x == 0 && shift_y == 0 && shift_z == 0);
                    float32x4_t rij_vec = vdupq_n_f32(rij);

                    int k = 0;
                    for (; k < n_points - 3; k += 4) {  // Vectorized loop
                        float32x4_t s_vec = vld1q_f32(&s_values_[k]);
                        float32x4_t sr_vec = vmulq_f32(s_vec, rij_vec);

                        float32x4_t sinc_vec;
                        if (is_self) {
                            sinc_vec = vdupq_n_f32(1.0f);
                        } else {
                            // Compute real sinc function (sin(sr)/sr) vectorized
                            float sr_vals[4];
                            vst1q_f32(sr_vals, sr_vec);
                            float sinc_data[4];
                            for (int m = 0; m < 4; m++) {
                                float sr = sr_vals[m];
                                sinc_data[m] = (sr == 0.0f) ? 1.0f : std::sin(sr) / sr;
                            }
                            sinc_vec = vld1q_f32(sinc_data);
                        }

                        float32x4_t fj_vec = vld1q_f32(&((ej == 0) ? f_V[k] : f_O[k]));
                        float32x4_t contrib = vmulq_f32(vmulq_f32(fi_vec[k / 4], fj_vec), 
                                                      vmulq_f32(sinc_vec, multiplicity_vec));
                        float32x4_t intensity_vec = vld1q_f32(&local_intensities[k]);
                        vst1q_f32(&local_intensities[k], vaddq_f32(intensity_vec, contrib));
                    }

                    // Cleanup loop for remaining points
                    for (; k < n_points; k++) {
                        float sr = s_values_[k] * rij;
                        float sinc_val = is_self ? 1.0f : (sr == 0.0f ? 1.0f : std::sin(sr) / sr);
                        float fi = (ei == 0) ? f_V[k] : f_O[k];
                        float fj = (ej == 0) ? f_V[k] : f_O[k];
                        local_intensities[k] += fi * fj * sinc_val * shifts[s].multiplicity;
                    }
                }
            }
        }
    }

    // Copy final intensities
    for (int k = 0; k < n_points; k++) {
        intensities[k] = local_intensities[k];
    }
}
