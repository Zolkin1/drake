#pragma once

#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/problem_definition.h"

namespace drake {
namespace traj_opt {

using Eigen::VectorXd;
using multibody::MultibodyForces;
using multibody::MultibodyPlant;
using systems::Context;

class TrajectoryOptimizer {
 public:
  /**
   * Construct a new Trajectory Optimizer object.
   *
   * @param plant A model of the system that we're trying to find an optimal
   *              trajectory for.
   * @param prob Problem definition, including cost, initial and target states,
   *             etc.
   */
  TrajectoryOptimizer(std::unique_ptr<const MultibodyPlant<double>> plant,
                      const ProblemDefinition& prob);

  /**
   * Convienience function to get the timestep of this optimization problem.
   *
   * @return double dt, the time step for this optimization problem
   */
  double time_step() const { return plant_->time_step(); }

  /**
   * Convienience function to get the time horizon of this optimization problem.
   *
   * @return double T, the number of time steps in the optimal trajectory.
   */
  double T() const { return prob_.T; }

  /**
   * Convienience function to get a const reference to the multibody plant that
   * we are optimizing over.
   *
   * @return const MultibodyPlant<double>&, the plant we're optimizing over.
   */
  const MultibodyPlant<double>& plant() const { return *plant_; }

  /**
   * Compute a sequence of generalized velocities v from a sequence of
   * generalized positions, where
   *
   *     v_t = (q_t - q_{t-1})/dt
   *
   * and v_0 is defined by the initial state of the optimization problem.
   *
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities
   */
  void CalcV(const std::vector<VectorXd>& q, std::vector<VectorXd>* v) const;

  /**
   * Compute a sequence of generalized forces t from sequences of generalized
   * velocities and positions, where generalized forces are defined by the
   * inverse dynamics,
   *
   *    tau_t = M*(v_{t+1}-v_t})/dt + D*v_{t+1} - k(q_t,v_t)
   *                               - (1/dt) *J'*gamma(v_{t+1},q_t).
   *
   * N.B. q and v have length T+1,
   *
   *  q = [q0, q1, ..., q_T],
   *  v = [v0, v1, ..., v_T],
   *
   * while tau has length T,
   *
   *  tau = [tau(0), tau(1), ..., tau(T-1)],
   *
   * i.e., tau(t) takes us us from t to t+1.
   *
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities
   * @param f_ext scratch space for computing external forces (e.g., gravity)
   * @param tau sequence of generalized forces
   */
  void CalcTau(const std::vector<VectorXd>& q, const std::vector<VectorXd>& v,
               MultibodyForces<double>* f_ext,
               std::vector<VectorXd>* tau) const;

 private:
  // A model of the system that we are trying to find an optimal trajectory for.
  std::unique_ptr<const MultibodyPlant<double>> plant_;

  // A context corresponding to plant_, to enable dynamics computations.
  std::unique_ptr<Context<double>> context_;

  // Stores the problem definition, including cost, time horizon, initial state,
  // target state, etc.
  const ProblemDefinition prob_;

  // Joint damping coefficients for the plant under consideration
  VectorXd joint_damping_;
};

}  // namespace traj_opt
}  // namespace drake