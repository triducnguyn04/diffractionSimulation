#include "fitting.h"

// Gaussian function
double PeakFitter::gaussian(double s, double A, double mu, double gamma) 
{
    return A * exp(-0.5 * pow((s - mu) / gamma, 2));
}

// Lorentzian function
double PeakFitter::lorentzian(double s, double A, double mu, double gamma) 
{
    return A * (gamma / (pow(s - mu, 2) + pow(gamma, 2))) / M_PI;
}

// Residual function
int PeakFitter::peak_f(const gsl_vector *x, void *data, gsl_vector *f) 
{
    struct FitData {
        Data* d;
        const std::vector<PeakType>* types;
    };
    FitData* fit_data = static_cast<FitData*>(data);
    Data* d = fit_data->d;
    const std::vector<PeakType>* types = fit_data->types;

    if (!x || !f || !d || !types) 
    {
        std::cerr << "Error: Null pointer in peak_f" << std::endl;
        return GSL_EFAULT;
    }

    size_t num_peaks = x->size / 3;
    for (size_t i = 0; i < d->s.size(); i++) 
    {
        double fit_val = 0.0;
        for (size_t j = 0; j < num_peaks; j++) 
        {
            double A     = gsl_vector_get(x, 3 * j);
            double mu    = gsl_vector_get(x, 3 * j + 1);
            double gamma = gsl_vector_get(x, 3 * j + 2);
            fit_val += (*types)[j] == PeakType::Gaussian ? 
                       gaussian(d->s[i], A, mu, gamma) : lorentzian(d->s[i], A, mu, gamma);
        }
        gsl_vector_set(f, i, fit_val - d->I[i]);
    }
    return GSL_SUCCESS;
}

// Load data from file
Data PeakFitter::loadData(const std::string& filename) 
{
    Data data;
    std::ifstream file(filename);
    if (!file) 
    {
        std::cerr << "Error opening data file: " << filename << std::endl;
        return data;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) 
    {
        line_num++;
        if (line.empty() || line[0] == '#') continue;
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        size_t comma_pos = line.find(',');
        if (comma_pos == std::string::npos) 
        {
            std::cerr << "Error: No comma found in line " << line_num << ": " << line << std::endl;
            continue;
        }

        std::string s_str = line.substr(0, comma_pos);
        std::string I_str = line.substr(comma_pos + 1);
        std::istringstream s_stream(s_str), I_stream(I_str);
        double s_val, I_val;
        if (!(s_stream >> s_val) || !(I_stream >> I_val)) 
        {
            std::cerr << "Error parsing line " << line_num << ": " << line << std::endl;
            continue;
        }
        data.s.push_back(s_val);
        data.I.push_back(I_val);
    }

    file.close();
    if (data.s.size() != data.I.size() || data.s.empty()) 
    {
        std::cerr << "Error: Invalid data size or no data loaded!" << std::endl;
    }
    return data;
}

// Load initial guesses from file
InitialGuess PeakFitter::loadInitialGuesses(const std::string& filename) 
{
    InitialGuess guess;
    std::ifstream file(filename);
    if (!file) 
    {
        std::cerr << "Error opening initial guess file: " << filename << std::endl;
        return guess;
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) 
    {
        line_num++;
        if (line.empty() || line[0] == '#') continue;
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        std::istringstream iss(line);
        std::string A_str, mu_str, gamma_str, type_str;

        if (!std::getline(iss, A_str, ',') || 
            !std::getline(iss, mu_str, ',') || 
            !std::getline(iss, gamma_str, ',')) 
        {
            std::cerr << "Error: Expected A, mu, gamma, type in line " << line_num << ": " << line << std::endl;
            continue;
        }

        std::streampos pos = iss.tellg();
        size_t start_pos = (pos == -1) ? 0 : static_cast<size_t>(pos);
        type_str = iss.str().substr(start_pos);
        type_str.erase(0, type_str.find_first_not_of(" \t"));

        if (type_str.empty()) 
        {
            std::cerr << "Error: Missing type in line " << line_num << ": " << line << std::endl;
            continue;
        }

        double A, mu, gamma;
        std::istringstream A_stream(A_str), mu_stream(mu_str), gamma_stream(gamma_str);
        if (!(A_stream >> A) || !(mu_stream >> mu) || !(gamma_stream >> gamma)) 
        {
            std::cerr << "Error parsing line " << line_num << ": " << line << std::endl;
            continue;
        }

        guess.A.push_back(A);
        guess.mu.push_back(mu);
        guess.gamma.push_back(gamma);
        guess.type.push_back((type_str == "L" || type_str == "lorentzian") ? PeakType::Lorentzian : PeakType::Gaussian);
    }

    file.close();
    if (guess.A.size() != guess.mu.size() || guess.A.size() != guess.gamma.size() || 
        guess.A.size() != guess.type.size() || guess.A.empty()) 
    {
        std::cerr << "Error: Invalid initial guess data!" << std::endl;
        guess = InitialGuess();
    }
    return guess;
}

// Fit peaks without any corrections
FitResult PeakFitter::fitPeaks(const Data& data, const InitialGuess& initial) 
{
    FitResult result;
    if (data.s.empty() || initial.A.empty()) 
    {
        std::cerr << "Error: Invalid input data or initial guesses!" << std::endl;
        return result;
    }

    size_t num_peaks = initial.A.size();
    gsl_vector *x = gsl_vector_alloc(3 * num_peaks);
    for (size_t j = 0; j < num_peaks; j++) 
    {
        gsl_vector_set(x, 3 * j,     initial.A[j]);
        gsl_vector_set(x, 3 * j + 1, initial.mu[j]);
        gsl_vector_set(x, 3 * j + 2, initial.gamma[j]);
    }

    struct FitData {
        Data* d;
        const std::vector<PeakType>* types;
    } fit_data = { const_cast<Data*>(&data), &initial.type };

    gsl_multifit_nlinear_fdf f;
    f.f     = peak_f;
    f.df    = nullptr;
    f.fvv   = nullptr;
    f.n     = data.s.size();
    f.p     = 3 * num_peaks;
    f.params = &fit_data;

    if (f.n < f.p) 
    {
        std::cerr << "Error: Too few data points for parameters!" << std::endl;
        gsl_vector_free(x);
        return result;
    }

    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_parameters params = gsl_multifit_nlinear_default_parameters();
    gsl_multifit_nlinear_workspace *solver = gsl_multifit_nlinear_alloc(T, &params, f.n, f.p);
    if (!solver) 
    {
        std::cerr << "Error: Failed to allocate GSL solver workspace" << std::endl;
        gsl_vector_free(x);
        return result;
    }

    int status = gsl_multifit_nlinear_init(x, &f, solver);
    if (status != GSL_SUCCESS) 
    {
        std::cerr << "Error: GSL initialization failed: " << gsl_strerror(status) << std::endl;
        gsl_multifit_nlinear_free(solver);
        gsl_vector_free(x);
        return result;
    }

    int max_iter = 100, iter = 0, info;
    do 
    {
        status = gsl_multifit_nlinear_iterate(solver);
        if (status != GSL_SUCCESS) 
        {
            std::cerr << "Fit failed: " << gsl_strerror(status) << std::endl;
            break;
        }
        status = gsl_multifit_nlinear_test(1e-8, 1e-8, 1e-8, &info, solver);
        iter++;
    } while (status == GSL_CONTINUE && iter < max_iter);

    // Extract results without correction
    double chi_square = 0.0;
    result.fit_I.resize(data.s.size());
    result.A.resize(num_peaks);
    result.mu.resize(num_peaks);
    result.gamma.resize(num_peaks);
    result.type = initial.type;

    for (size_t i = 0; i < data.s.size(); i++) 
    {
        double fit_val = 0.0;
        for (size_t j = 0; j < num_peaks; j++) 
        {
            double A     = gsl_vector_get(gsl_multifit_nlinear_position(solver), 3 * j);
            double mu    = gsl_vector_get(gsl_multifit_nlinear_position(solver), 3 * j + 1);
            double gamma = gsl_vector_get(gsl_multifit_nlinear_position(solver), 3 * j + 2);
            fit_val += (initial.type[j] == PeakType::Gaussian) ? 
                       gaussian(data.s[i], A, mu, gamma) : lorentzian(data.s[i], A, mu, gamma);
            if (i == 0) 
            {
                result.A[j]     = A;
                result.mu[j]    = mu;
                result.gamma[j] = gamma;
            }
        }
        result.fit_I[i] = fit_val;
        chi_square += pow(data.I[i] - fit_val, 2);
    }
    result.reduced_chi_square = chi_square / (data.s.size() - 3 * num_peaks);

    gsl_multifit_nlinear_free(solver);
    gsl_vector_free(x);
    return result;
}

// Compute fit without optimization
FitResult PeakFitter::computeFitWithoutOptimization(const Data& data, const InitialGuess& initial) 
{
    FitResult result;
    if (data.s.empty() || initial.A.empty()) 
    {
        std::cerr << "Error: Invalid input data or initial guesses!" << std::endl;
        return result;
    }

    size_t num_peaks = initial.A.size();
    result.fit_I.resize(data.s.size());
    result.A     = initial.A;
    result.mu    = initial.mu;
    result.gamma = initial.gamma;
    result.type  = initial.type;

    double chi_square = 0.0;
    for (size_t i = 0; i < data.s.size(); i++) 
    {
        double fit_val = 0.0;
        for (size_t j = 0; j < num_peaks; j++) 
        {
            fit_val += (initial.type[j] == PeakType::Gaussian) ? 
                       gaussian(data.s[i], initial.A[j], initial.mu[j], initial.gamma[j]) : 
                       lorentzian(data.s[i], initial.A[j], initial.mu[j], initial.gamma[j]);
        }
        result.fit_I[i] = fit_val;
        chi_square += pow(data.I[i] - fit_val, 2);
    }
    result.reduced_chi_square = chi_square / (data.s.size() - 3 * num_peaks);

    return result;
}

// Main processing function
bool PeakFitter::fitAndProcess(const std::string& dataFile, const std::string& initialGuessFile, bool fit) 
{
    Data data = loadData(dataFile);
    if (data.s.empty()) 
    {
        std::cerr << "Failed to load data from " << dataFile << std::endl;
        return false;
    }

    InitialGuess initial = loadInitialGuesses(initialGuessFile);
    if (initial.A.empty()) 
    {
        std::cerr << "Failed to load initial guesses from " << initialGuessFile << std::endl;
        return false;
    }

    FitResult result = fit ? fitPeaks(data, initial) : computeFitWithoutOptimization(data, initial);
    if (result.A.empty()) 
    {
        std::cerr << "Processing failed or no results returned!" << std::endl;
        return false;
    }

    std::cout << (fit ? "Fitted" : "Initial") << " Parameters:" << std::endl;
    for (size_t j = 0; j < result.A.size(); j++) 
    {
        std::cout << "Peak " << j + 1 << " (" << (result.type[j] == PeakType::Gaussian ? "Gaussian" : "Lorentzian") 
                  << "): A = " << result.A[j] << ", mu = " << result.mu[j] << ", gamma = " << result.gamma[j] << std::endl;
    }
    std::cout << "Reduced Chi-Square: " << result.reduced_chi_square << std::endl;

    std::ofstream output("fitted_data.txt");
    if (!output) 
    {
        std::cerr << "Error: Could not open fitted_data.txt for writing" << std::endl;
        return false;
    }

    output << "# " << (fit ? "Fitted" : "Initial") << " Parameters:\n";
    for (size_t j = 0; j < result.A.size(); j++) 
    {
        output << "# Peak " << j + 1 << " (" << (result.type[j] == PeakType::Gaussian ? "Gaussian" : "Lorentzian") 
               << "): A = " << result.A[j] << ", mu = " << result.mu[j] << ", gamma = " << result.gamma[j] << "\n";
    }
    output << "# Reduced Chi-Square: " << result.reduced_chi_square << "\n";
    output << "# Data: s, fit_I\n";
    for (size_t i = 0; i < result.fit_I.size(); i++) 
    {
        output << data.s[i] << " " << result.fit_I[i] << "\n";
    }

    output.close();
    std::cout << "Data and parameters saved to fitted_data.txt" << std::endl;
    return true;
}
