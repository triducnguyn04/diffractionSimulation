#ifndef DEBYESCATTERING_H
#define DEBYESCATTERING_H

#include <gemmi/cif.hpp>
#include <gemmi/unitcell.hpp>
#include <vector>
#include <string>

#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error "This code requires ARM NEON support. Please compile on an ARM platform with NEON enabled."
#endif

class DebyeScattering {
public:
    DebyeScattering();
    ~DebyeScattering() = default; // Default destructor for completeness

    void clear();
    bool loadFromCIF(const std::string& filename, float s_min, float s_max, int n_points);
    void calculateIntensity(int n_replicas, float r_max, 
                          std::vector<float>& s_values, 
                          std::vector<float>& intensities);
    size_t getAtomCount() const;

private:
    std::vector<float> x, y, z;  // Cartesian coordinates
    std::vector<std::string> elements;
    gemmi::UnitCell cell;

    float s_min, s_max, ds;
    int n_points;
    std::vector<float> f_V, f_O;  // Scattering factors for V and O
    std::vector<float> s_values_; // Precomputed s-values

    static constexpr size_t SINC_TABLE_SIZE = 1000000;
    static constexpr float SINC_MAX = 6000.0f;
    std::vector<float> sinc_table;

    void initSincTable();
    inline float interpolateSinc(float x) const;
    float getScatteringFactor(const std::string& element, float s) const;
    size_t calculateMultiplicity(int nx, int ny, int nz, int n_replicas) const;
};

#endif // DEBYESCATTERING_H
