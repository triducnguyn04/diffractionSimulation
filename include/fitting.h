#ifndef FITTING_H
#define FITTING_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <gsl/gsl_multifit_nlinear.h>
#include <sstream>
#include <string>

// Structure to hold input data: s (scattering factor) and I (peak intensity)
struct Data 
{
    std::vector<double> s;                  
    std::vector<double> I;                  
};

// Enum to specify peak type
enum class PeakType {Gaussian, Lorentzian};

// Structure to hold initial guess parameters for peaks
struct InitialGuess 
{
    std::vector<double> A;                  // Amplitudes of the peaks
    std::vector<double> mu;                 // Means (centers) of the peaks
    std::vector<double> gamma;              // Width parameter (sigma for Gaussian, gamma for Lorentzian)
    std::vector<PeakType> type;             // Type of each peak (Gaussian or Lorentzian)
};

// Structure to hold the results of the fitting process
struct FitResult 
{
    std::vector<double> A;                  // Fitted or initial amplitudes
    std::vector<double> mu;                 // Fitted or initial means
    std::vector<double> gamma;              // Fitted or initial widths (sigma or gamma)
    std::vector<PeakType> type;             // Type of each peak
    double reduced_chi_square;              // Reduced chi-square statistic for fit quality
    std::vector<double> fit_I;              // Fitted intensity values
};

// Class to handle peak fitting and processing
class PeakFitter 
{
private:
    // Gaussian function: computes a single Gaussian value
    static double gaussian(double s, double A, double mu, double gamma);

    // Lorentzian function: computes a single Lorentzian value
    static double lorentzian(double s, double A, double mu, double gamma);

    // Residual function for GSL solver: computes model-data differences
    static int peak_f(const gsl_vector *x, void *data, gsl_vector *f);

    // Load data from a file into a Data struct
    Data loadData(const std::string& filename);

    // Load initial guesses from a file into an InitialGuess struct
    InitialGuess loadInitialGuesses(const std::string& filename);

    // Fit peaks to data using GSL nonlinear least-squares
    FitResult fitPeaks(const Data& data, const InitialGuess& initial);

public:
    // Default constructor and destructor
    PeakFitter() = default;
    ~PeakFitter() = default;

    // Compute fit using initial guesses without optimization
    FitResult computeFitWithoutOptimization(const Data& data, const InitialGuess& initial);

    // Main function to process data: loads, fits or uses guesses, and outputs results
    bool fitAndProcess(const std::string& dataFile, const std::string& initialGuessFile, bool fit = true);
};

#endif
