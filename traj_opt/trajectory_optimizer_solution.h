#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

// Status indicator for the overall success of our trajectory optimization.
enum SolverFlag { kSuccess, kLinesearchMaxIters, kFactorizationFailed };

/**
 * A container for the optimal solution, including generalized positions,
 * velocities, and forces.
 *
 * TODO(vincekurtz): consider holding control inputs u rather than generalized
 * forces tau (tau = B*u)
 */
template <typename T>
struct TrajectoryOptimizerSolution {
  // Optimal sequence of generalized positions at each timestep
  std::vector<VectorX<T>> q;

  // Optimal sequence of generalized velocities at each timestep
  std::vector<VectorX<T>> v;

  // Optimal sequence of generalized forces at each timestep
  std::vector<VectorX<T>> tau;
};

/**
 * A container for data about the solve process
 */
template <typename T>
struct TrajectoryOptimizerStats {
  // Total solve time
  double solve_time;

  // Time for each iteration
  std::vector<double> iteration_times;

  // Cost at each iteration
  std::vector<T> iteration_costs;

  // Number of linesearch iterations, or number of times the trust-region was
  // modified, for each outer iteration
  std::vector<int> subproblem_iterations;

  // Linsearch parameter alpha for each iteration
  std::vector<double> linesearch_alphas;

  // Trust region radius Δ for each iteration
  std::vector<T> trust_region_radii;

  // Norm of the gradient at each iteration
  std::vector<T> gradient_norms;

  // Norm of the search direction at each iteration
  std::vector<T> dq_norms;

  // Trust region ratio (L(q) - L(q+dq)) / (m(q) - m(q+dq)), where m is a
  // quadratic model of the cost
  std::vector<T> trust_region_ratios;

  /**
   * Add the data from one iteration to the stored lists
   *
   * @param iter_time compute time for this iteration
   * @param iter_cost cost at this iteration
   * @param subproblem_iters number of linesearch or trust-region iterations
   * @param alpha linesearch parameter
   * @param delta trust region raidus
   * @param dq_norm norm of the linesearch direction Δq
   * @param tr_ratio trust region ratio: actual cost reduction / expected cost
   * reduction
   * @param grad_norm norm of the gradient
   */
  void push_data(double iter_time, T iter_cost, int subproblem_iters,
                 double alpha, double delta, T dq_norm, T tr_ratio, T grad_norm) {
    iteration_times.push_back(iter_time);
    iteration_costs.push_back(iter_cost);
    subproblem_iterations.push_back(subproblem_iters);
    linesearch_alphas.push_back(alpha);
    trust_region_radii.push_back(delta);
    dq_norms.push_back(dq_norm);
    trust_region_ratios.push_back(tr_ratio);
    gradient_norms.push_back(grad_norm);
  }

  /**
   * Check if the stored lists of data are all empty
   */
  bool is_empty() const {
    return ((iteration_times.size() == 0) && (iteration_costs.size() == 0) &&
            (subproblem_iterations.size() == 0) &&
            (linesearch_alphas.size() == 0) &&
            (trust_region_radii.size() == 0) && (dq_norms.size() == 0) &&
            (trust_region_ratios.size() == 0) && (gradient_norms.size() == 0));
  }

  /**
   * Save the solution to a CSV file that we cn process and make plots from
   * later.
   *
   * @param filename filename for the csv file that we'll write to
   */
  void SaveToCsv(std::string fname) const {
    // Set file to write to
    std::ofstream data_file;
    data_file.open(fname);

    // Write a header
    data_file << "iter, time, cost, sub_iters, alpha, delta, dq_norm, "
                 "trust_region_ratio, grad_norm\n";

    const int num_iters = iteration_times.size();
    for (int i = 0; i < num_iters; ++i) {
      // Write the data
      data_file << i << ", ";
      data_file << iteration_times[i] << ", ";
      data_file << iteration_costs[i] << ", ";
      data_file << subproblem_iterations[i] << ", ";
      data_file << linesearch_alphas[i] << ", ";
      data_file << trust_region_radii[i] << ", ";
      data_file << dq_norms[i] << ", ";
      data_file << trust_region_ratios[i] << ", ";
      data_file << gradient_norms[i] << "\n";
    }

    // Close the file
    data_file.close();
  }
};

}  // namespace traj_opt
}  // namespace drake
