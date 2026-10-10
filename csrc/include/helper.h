#ifndef HELPER_H
#define HELPER_H

#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>

// check if a vectoris close to 0
bool is_close_to_zero(const Eigen::VectorXf& vec, float epsilon);

// Function to display a progress bar with elapsed and estimated time
void displayProgressBar(int width, double progress_percentage, double seconds_elapsed);

// Custom random number generator function (uniform on [0, 1))
double custom_rand();

// Standard-normal sample N(0, 1), drawn from the same generator as
// ``custom_rand`` so ``set_random_seed`` makes it reproducible too.
double custom_randn();

// Seed the custom_rand RNG deterministically.  When unset, the generator
// is seeded once at module load via std::random_device, which makes
// cobweb tie-breaking (e.g. in two_best_children, get_best_operation)
// non-deterministic across runs. Calling set_random_seed(seed) makes
// subsequent custom_rand() calls reproducible.
void set_random_seed(uint64_t seed);

// Function to repeat a string 'n' times
std::string repeat(std::string s, int n);

// Function to convert a double to a string with maximum precision
std::string doubleToString(double cnt);

// Function to compute the log-sum-exp of a vector of doubles
double logsumexp(const std::vector<double> &arr);

// Function to compute the log-sum-exp of a vector of floats
float logsumexp(const std::vector<float> &arr);

// Function to compute the log-sum-exp of two doubles
double logsumexp(double n1, double n2);

// logsumexp for floats
float logsumexp_f(float n1, float n2);

// logsumexp for floats
Eigen::VectorXf elementwise_logsumexp(const Eigen::VectorXf &v1, const Eigen::VectorXf &v2);

#endif // HELPER_H
