#ifndef CIFPARSER_H
#define CIFPARSER_H

#include <gemmi/cif.hpp>
#include <gemmi/unitcell.hpp>
#include <vector>
#include <string>
#include "aligned_memory.h" // Include the new header

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

private:
    float to_float(const std::string* str) const;
};

#endif // CIFPARSER_H
