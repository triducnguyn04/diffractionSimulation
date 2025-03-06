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

// Custom allocator for aligned memory
template<typename T>
class AlignedAllocator {
public:
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    // 16-byte alignment for NEON
    T* allocate(size_t n) {
        void* ptr = aligned_alloc(16, n * sizeof(T)); 
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t) noexcept {
        std::free(ptr);
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

// Alias for aligned vector
template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

class DebyeScattering {
public:
// Default destructor for completeness
    DebyeScattering();
    ~DebyeScattering() = default; 

    void clear();
    bool loadFromCIF(const std::string& filename, float s_min, float s_max, int n_points);
    void calculateIntensity(int n_replicas, float r_max, 
                           std::vector<float>& s_values, 
                           std::vector<float>& intensities);
    size_t getAtomCount() const;

private:
    aligned_vector<float> x, y, z;                                   // Cartesian coordinates of atoms
    aligned_vector<std::string> elements;                            // Element symbols (e.g., "V", "O")
    gemmi::UnitCell cell;                                            // Unit cell parameters from CIF

    float s_min, s_max, ds;                                          // Scattering vector range and step size
    int n_points;                                                    // Number of scattering vector points
    aligned_vector<float> f_V, f_O;                                  // Precomputed scattering factors for V and O
    aligned_vector<float> s_values_;                                 // Precomputed s-values

    // Precomputed unit cell constants for efficiency
    float ax, by, cz_cos_beta, cz_sin_beta;

    // Sinc table for fast lookup in intensity calculation
    static constexpr size_t SINC_TABLE_SIZE = 1000000;                                          // Size of sinc lookup table
    static constexpr float SINC_MAX = 6000.0f;                                                  // Maximum sr value for table
    aligned_vector<float> sinc_table;                                                           // Precomputed sinc values

    // Private helper functions
    void initSincTable();                                                                       // Initialize sinc lookup table
    inline float interpolateSinc(float x) const;                                                // Interpolate sinc(sr) from table
    float getScatteringFactor(const std::string& element, float s) const;                       // Element scattering factor
    size_t calculateMultiplicity(int nx, int ny, int nz, int n_replicas) const;                 // Replica multiplicity
};

#endif // DEBYESCATTERING_H
