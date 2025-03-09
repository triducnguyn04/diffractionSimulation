#ifndef DEBYESCATTERING_H
#define DEBYESCATTERING_H

#include <gemmi/unitcell.hpp>
#include <vector>
#include <string>
#include "aligned_memory.h"
#include "cifparser.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error "This code requires ARM NEON support. Please compile on an ARM platform with NEON enabled."
#endif

class DebyeScattering {
public:
    DebyeScattering();
    ~DebyeScattering() = default;

    void clear();
    bool loadFromCIF(const std::string& filename, float s_min, float s_max, int n_points);
    void setData(const aligned_vector<float>& x_in, 
                 const aligned_vector<float>& y_in, 
                 const aligned_vector<float>& z_in, 
                 const aligned_vector<std::string>& elements_in, 
                 const gemmi::UnitCell& cell_in, 
                 float s_min, float s_max, int n_points);
    void calculateIntensity(int n_replicas, float r_max, 
                           std::vector<float>& s_values, 
                           std::vector<float>& intensities);
    size_t getAtomCount() const;

private:
    aligned_vector<float> x, y, z;           // Cartesian coordinates of atoms
    aligned_vector<std::string> elements;    // Element symbols (e.g., "V", "O")
    gemmi::UnitCell cell;                    // Unit cell parameters from CIF

    float s_min, s_max, ds;                  // Scattering vector range and step size
    int n_points;                            // Number of scattering vector points
    aligned_vector<float> f_V, f_O;          // Precomputed scattering factors for V and O
    aligned_vector<float> s_values_;         // Precomputed s-values

    // Precomputed unit cell constants for efficiency
    float ax, by, cz_cos_beta, cz_sin_beta;

    // Sinc table for fast lookup in intensity calculation
    static constexpr size_t SINC_TABLE_SIZE = 100000;
    static constexpr float SINC_MAX = 6000.0f;
    aligned_vector<float> sinc_table;

    // Private helper functions
    void initSincTable();
    inline float interpolateSinc(float x) const;
    float getScatteringFactor(const std::string& element, float s) const;
    size_t calculateMultiplicity(int nx, int ny, int nz, int n_replicas) const;
    void precomputeScatteringFactors();      // New helper to avoid code duplication
};

#endif // DEBYESCATTERING_H
