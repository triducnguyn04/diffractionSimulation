#ifndef BRAGGCALCULATION_H
#define BRAGGCALCULATION_H

#include <gemmi/cif.hpp>
#include <gemmi/unitcell.hpp>
#include <vector>
#include <string>

class BraggCalculation {
public:
    BraggCalculation();
    void clear();
    bool loadFromCIF(const std::string& filename);
    double calculateBraggIntensity(int h, int k, int l) const;
    double getScatteringFactor(const std::string& element, double s) const;

    // Getter for unit cell (useful for external access)
    const gemmi::UnitCell& getUnitCell() const { return cell; }

private:
    std::vector<double> x; // Cartesian coordinates
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> f; // Scattering factors
    std::vector<std::string> elements; // Element types
    gemmi::UnitCell cell; // Unit cell parameters
};

#endif // BRAGGCALCULATION_H
