#include "debyescattering.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <arm_neon.h>
#include <memory>

DebyeScattering::DebyeScattering() {
    clear();
    initSincTable();
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
    ax = 0.0f; by = 0.0f; cz_cos_beta = 0.0f; cz_sin_beta = 0.0f;
}

void DebyeScattering::initSincTable() {
    sinc_table.resize(SINC_TABLE_SIZE);
    float step = SINC_MAX / (SINC_TABLE_SIZE - 1);
    for (size_t i = 0; i < SINC_TABLE_SIZE; ++i) {
        float x = i * step;
        sinc_table[i] = (x == 0.0f) ? 1.0f : std::sin(x) / x;
    }
}

inline float DebyeScattering::interpolateSinc(float x) const {
    if (x >= SINC_MAX) return 0.0f;
    float index = x * (SINC_TABLE_SIZE - 1) / SINC_MAX;
    size_t i = static_cast<size_t>(index);
    float frac = index - i;
    if (i + 1 >= SINC_TABLE_SIZE) return sinc_table[SINC_TABLE_SIZE - 1];
    return sinc_table[i] * (1.0f - frac) + sinc_table[i + 1] * frac;
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
    CIFParser parser;
    clear();
    if (!parser.loadFromCIF(filename, x, y, z, elements, cell)) {
        return false;
    }

    float beta_rad = cell.beta * M_PI / 180.0f;
    float cos_beta = std::cos(beta_rad);
    float sin_beta = std::sin(beta_rad);
    ax = static_cast<float>(cell.a);
    by = static_cast<float>(cell.b);
    cz_cos_beta = static_cast<float>(cell.c * cos_beta);
    cz_sin_beta = static_cast<float>(cell.c * sin_beta);

    s_min = s_min_input;
    s_max = s_max_input;
    n_points = n_points_input;
    ds = (s_max - s_min) / (n_points - 1);
    f_V.resize(n_points);
    f_O.resize(n_points);
    s_values_.resize(n_points);

    std::cout << "Precomputing scattering factors for V and O...\n";
    for (int i = 0; i < n_points; i += 4) {
        if (i + 3 < n_points) {
            float32x4_t s_vec = {s_min + i * ds, s_min + (i + 1) * ds, 
                                 s_min + (i + 2) * ds, s_min + (i + 3) * ds};
            vst1q_f32(&s_values_[i], s_vec);
            float32x4_t s2_vec = vmulq_f32(s_vec, s_vec);

            float v_s2[4];
            vst1q_f32(v_s2, s2_vec);
            float v_vals[4];
            for (int m = 0; m < 4; ++m) {
                v_vals[m] = getScatteringFactor("V", std::sqrt(v_s2[m]));
            }
            float32x4_t f_V_vec = vld1q_f32(v_vals);
            vst1q_f32(&f_V[i], f_V_vec);

            float o_vals[4];
            for (int m = 0; m < 4; ++m) {
                o_vals[m] = getScatteringFactor("O", std::sqrt(v_s2[m]));
            }
            float32x4_t f_O_vec = vld1q_f32(o_vals);
            vst1q_f32(&f_O[i], f_O_vec);
        } else {
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

// Rest of the file (calculateMultiplicity, getAtomCount, calculateIntensity) remains unchanged
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
    intensities.resize(n_points, 0.0f);
    s_values.resize(n_points);
    for (int i = 0; i < n_points; ++i) {
        s_values[i] = s_values_[i];
    }

    std::cout << "Calculating intensity from central unit cell with " << n 
              << " atoms, n_replicas = " << n_replicas << "\n";

    struct ReplicaShift {
        float shift_x, shift_y, shift_z;
        size_t multiplicity;
    };
    std::vector<ReplicaShift> shifts;
    shifts.reserve((2 * n_replicas + 1) * (2 * n_replicas + 1) * (2 * n_replicas + 1));

    for (int nx = -2*n_replicas; nx <= 2*n_replicas; nx++) {
        for (int ny = -2*n_replicas; ny <= 2*n_replicas; ny++) {
            for (int nz = -2*n_replicas; nz <= 2*n_replicas; nz++) {
                float shift_x = nx * ax + nz * cz_cos_beta;
                float shift_y = ny * by;
                float shift_z = nz * cz_sin_beta;
                float r = std::sqrt(shift_x * shift_x + shift_y * shift_y + shift_z * shift_z);
                if (r <= r_max) {
                    shifts.push_back({shift_x, shift_y, shift_z, calculateMultiplicity(nx, ny, nz, n_replicas)});
                }
            }
        }
    }

    aligned_vector<int> elem_indices(n);
    for (int i = 0; i < n; i++) {
        elem_indices[i] = (elements[i] == "V") ? 0 : 1;
    }

    aligned_vector<float> local_intensities(n_points, 0.0f);

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:local_intensities[:n_points])
    #endif
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
                bool is_self = (i == j && shift_x == 0 && shift_y == 0 && shift_z == 0);
                float32x4_t rij_vec = vdupq_n_f32(rij);

                int k = 0;
                for (; k < n_points - 3; k += 4) {
                    float32x4_t s_vec = vld1q_f32(&s_values_[k]);
                    float32x4_t sr_vec = vmulq_f32(s_vec, rij_vec);
                    float sr_vals[4];
                    vst1q_f32(sr_vals, sr_vec);
                    float sinc_vals[4];
                    for (int m = 0; m < 4; ++m) {
                        sinc_vals[m] = is_self ? 1.0f : interpolateSinc(sr_vals[m]);
                    }
                    float32x4_t sinc_vec = vld1q_f32(sinc_vals);

                    float32x4_t fj_vec = vld1q_f32(&((ej == 0) ? f_V[k] : f_O[k]));
                    float32x4_t contrib = vmulq_f32(vmulq_f32(fi_vec[k / 4], fj_vec), 
                                                  vmulq_f32(sinc_vec, multiplicity_vec));
                    float32x4_t intensity_vec = vld1q_f32(&local_intensities[k]);
                    vst1q_f32(&local_intensities[k], vaddq_f32(intensity_vec, contrib));
                }

                for (; k < n_points; k++) {
                    float sr = s_values_[k] * rij;
                    float sinc_val = is_self ? 1.0f : interpolateSinc(sr);
                    float fi = (ei == 0) ? f_V[k] : f_O[k];
                    float fj = (ej == 0) ? f_V[k] : f_O[k];
                    local_intensities[k] += fi * fj * sinc_val * shifts[s].multiplicity;
                }
            }
        }
    }

    for (int k = 0; k < n_points; k++) {
        intensities[k] = local_intensities[k];
    }
}
