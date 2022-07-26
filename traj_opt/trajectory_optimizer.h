#pragma once

#include <memory>
#include <vector>

#include "drake/common/eigen_types.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/problem_data.h"

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
  TrajectoryOptimizer(const MultibodyPlant<double>* plant,
                      const ProblemDefinition& prob);

  /**
   * Convienience function to get the timestep of this optimization problem.
   *
   * @return double dt, the time step for this optimization problem
   */
  double time_step() const { return plant_->time_step(); }

  /**
   * Convienience function to get the time horizon (T) of this optimization
   * problem.
   *
   * @return int the number of time steps in the optimal trajectory.
   */
  int num_steps() const { return prob_.num_steps; }

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
   *     v_t = (q_t - q_{t-1})/dt            (1)
   *
   * v and q are each vectors of length num_steps+1,
   *
   *     v = [v(0), v(1), v(2), ..., v(num_steps)],
   *     q = [q(0), q(1), q(2), ..., q(num_steps)].
   *
   * Note that v0 = v_init is defined by the initial state of the optimization
   * problem, rather than Equation (1) above.
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
   * Note that q and v have length num_steps+1,
   *
   *  q = [q(0), q(1), ..., q(num_steps)],
   *  v = [v(0), v(1), ..., v(num_steps)],
   *
   * while tau has length num_steps,
   *
   *  tau = [tau(0), tau(1), ..., tau(num_steps-1)],
   *
   * i.e., tau(t) takes us us from t to t+1.
   *
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities
   * @param a scratch space for computing accelerations
   * @param f_ext scratch space for computing external forces (e.g., gravity)
   * @param tau sequence of generalized forces
   */
  void CalcTau(const std::vector<VectorXd>& q, const std::vector<VectorXd>& v,
               VectorXd* a, MultibodyForces<double>* f_ext,
               std::vector<VectorXd>* tau) const;

  /**
   * Compute partial derivatives of the inverse dynamics 
   * 
   *    tau_t = ID(q_{t-1}, q_t, q_{t+1})
   * 
   * and store them in the given GradientData struct. 
   * 
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities (computed from q)
   * @param grad_data struct for holding dtau/dq
   */
  void CalcInverseDynamicsPartials(const std::vector<VectorXd>& q,
                                   const std::vector<VectorXd>& v,
                                   GradientData* grad_data) const;
  
  /**
   * Compute partial derivatives of the inverse dynamics 
   * 
   *    tau_t = ID(q_{t-1}, q_t, q_{t+1})
   * 
   * using finite differences.
   * 
   * For testing purposes only - this is very inefficient.
   * 
   * @param q sequence of generalized positions
   * @param v sequence of generalized velocities (computed from q)
   * @param grad_data struct for holding dtau/dq
   */
  void CalcInverseDynamicsPartialsFiniteDiff(const std::vector<VectorXd>& q,
                                             const std::vector<VectorXd>& v,
                                             GradientData* grad_data) const;

  /**
   * Compute the partial derivative of generalized forces at the previous
   * timestep, tau_{t-1}, with respect to generalized positions at the current
   * timestep, q_t.
   *
   * @param q sequence of all generalized positions
   * @param t timestep under consideration
   * @param dtaum_dq ∂tau_{t-1} / ∂q_t
   */
  void CalcDtaumDq(const std::vector<VectorXd>& q, const int t,
                   Eigen::Ref<MatrixXd> dtaum_dq) const;
  void CalcDtauDq(const std::vector<VectorXd>& q, const int t,
                  Eigen::Ref<MatrixXd> dtaum_dq) const;
  void CalcDtaupDq(const std::vector<VectorXd>& q, const int t,
                   Eigen::Ref<MatrixXd> dtaum_dq) const;

  /**
   * Compute ∂tau_s / ∂q_t using finite differences.
   *
   * @param q sequence of all generalized positions
   * @param s timestep of tau
   * @param t timestep of q
   * @param dtaus_dqt ∂tau_s / ∂q_t
   */
  void CalcDtausDqtFiniteDiff(const std::vector<VectorXd>& q, const int s,
                              const int t,
                              Eigen::Ref<MatrixXd> dtaus_dqt) const;

 private:
  // A model of the system that we are trying to find an optimal trajectory for.
  const MultibodyPlant<double>* plant_;

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
