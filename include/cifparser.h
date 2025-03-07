#ifndef CIFPARSER_H
#define CIFPARSER_H

#include <gemmi/cif.hpp>
#include <gemmi/unitcell.hpp>
#include <vector>
#include <string>
#include "aligned_memory.h"

class CIFParser {
public:
    CIFParser() = default;
    ~CIFParser() = default;

    bool loadFromCIF(const std::string& filename, 
                     aligned_vector<float>& x, 
                     aligned_vector<float>& y, 
                     aligned_vector<float>& z, 
                     aligned_vector<std::string>& elements, 
                     gemmi::UnitCell& cell);

    // Static method to load from stored data if available
    static bool loadStoredData(aligned_vector<float>& x, 
                               aligned_vector<float>& y, 
                               aligned_vector<float>& z, 
                               aligned_vector<std::string>& elements, 
                               gemmi::UnitCell& cell);

    // Static flag to check if data has been loaded
    static bool hasLoadedData;

private:
    float to_float(const std::string* str) const;

    // Static storage for previously loaded CIF data
    static aligned_vector<float> stored_x, stored_y, stored_z;
    static aligned_vector<std::string> stored_elements;
    static gemmi::UnitCell stored_cell;
};

#endif // CIFPARSER_H
