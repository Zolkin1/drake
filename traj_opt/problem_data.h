#pragma once

#include <iostream>//DEBUG
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace traj_opt {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Struct containing the gradients of key dynamics terms that are used to
 * compute the (approximate) gradient and Hessian for our Gauss-Newton steps.
 */
struct GradientData {
  /**
   * Constructor which allocates variables of the proper sizes.
   * 
   * @param num_steps number of time steps in the optimization problem
   * @param nv number of generalized velocities (size of tau and v)
   * @param nq number of generalized positions (size of q)
   */
  GradientData(const int num_steps, const int nv, const int nq) {
    dtau_dqm.assign( num_steps, MatrixXd(nv, nq));
    dtau_dq.assign( num_steps, MatrixXd(nv, nq));
    dtau_dqp.assign( num_steps, MatrixXd(nv, nq));

    // Set all derivatives w.r.t q(0) and q(-1) to zero
    dtau_dq[0].setZero();
    dtau_dqm[0].setZero();
    dtau_dqm[1].setZero();
  }

  // Return the number of steps allocated in this object.
  // TODO(vincekurtz): we may eventually want size checks to include the
  // dimensions of each matrix for each timestep.
  int size() const {
    DRAKE_DEMAND(dtau_dq.size() == dtau_dqm.size());
    DRAKE_DEMAND(dtau_dq.size() == dtau_dqp.size());
    return dtau_dq.size();
  }

  // Partials of tau_t with respect to q_{t-1} ("q_t minus"),
  //
  //    d(tau_t)/d(q_{t-1})
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps-1.
  // For values where tau_t or q_{t-1} is undefined or constant, we simply store
  // zeros, i.e.,
  //
  //    [ 0, 0, d(tau_2)/d(q_1), ... , d(tau_{num_steps-1})/d(q_{num_steps-2})]
  //
  std::vector<MatrixXd> dtau_dqm;

  // Partials of tau_t with respect to q_t,
  //
  //    d(tau_t)/d(q_t)
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps-1.
  // For values where tau_t or q_t is undefined or constant, we simply store
  // zeros, i.e.,
  //
  //    [ 0, d(tau_1)/d(q_1), ... , d(tau_{num_steps-1})/d(q_{num_steps-1})]
  //
  std::vector<MatrixXd> dtau_dq;

  // Partial of tau_t with respect to q_{t+1} ("q_t plus"),
  //
  //    d(tau_t)/d(q_{t+1})
  //
  // for each timestep. This is indexed by t ranging from 0 to num_steps-1.
  // For values where tau_t or q_{t+1} is undefined or constant, we simply store
  // zeros, i.e.,
  //
  //    [ d(tau_0)/d(q_1), d(tau_1)/d(q_2), ... ,
  //                                      d(tau_{num_steps-1})/d(q_{num_steps})]
  //
  std::vector<MatrixXd> dtau_dqp;
};

}  // namespace traj_opt
}  // namespace drake
