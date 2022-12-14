#pragma once

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/examples/yaml_config.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {

using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;

/**
 * Abstract base class for trajectory optimization examples.
 */
class TrajOptExample {
 public:
  virtual ~TrajOptExample() = default;

  /**
   * Solve the optimization problem, as defined by the parameters in the given
   * YAML file.
   *
   * @param options_file YAML file containing cost function definition, solver
   * parameters, etc., with fields as defined in yaml_config.h.
   */
  void SolveTrajectoryOptimization(const std::string options_file) const;

  /**
   * Use the trajectory optimizer as a controller in simulation, solving the
   * optimization problem for a fixed number of iterations, applying the
   * corresponding control signal, and resolving the optimization problem.
   *
   * @param options_file YAML file containing the cost definition, initial
   * state, and other parameters.
   * @param iters The number of optimization iterations to take to determine
   * each control tape. Overrides whatever parameters are set in the YAML file.
   * @param simulator_dt Time step for the simulator
   * @param simulator_time Time (in seconds) to run the simulation for
   */
  void RunModelPredictiveControl(const std::string options_file,
                                 const int iters, const double simulator_dt,
                                 const double simulator_time) const;

 private:
  /**
   * Create a MultibodyPlant model of the system that we're optimizing over.
   * This is the only method that needs to be overwritten to specialize to
   * different systems.
   *
   * @param plant the MultibodyPlant that we'll add the system to.
   */
  virtual void CreatePlantModel(MultibodyPlant<double>*) const {}

  /**
   * Play back the given trajectory on the Drake visualizer
   *
   * @param q sequence of generalized positions defining the trajectory
   * @param time_step time step (seconds) for the discretization
   */
  void PlayBackTrajectory(const std::vector<VectorXd>& q,
                          const double time_step) const;

  /**
   * Set an optimization problem from example options which were loaded from
   * YAML.
   *
   * @param options parameters loaded from yaml
   * @param opt_prob the problem definition (cost, initital state, etc)
   */
  void SetProblemDefinition(const TrajOptExampleParams& options,
                            ProblemDefinition* opt_prob) const;

  /**
   * Set solver parameters (used to pass options to the optimizer)
   * from example options (loaded from a YAML file).
   *
   * @param options parameters loaded from yaml
   * @param solver_params parameters for the optimizer that we'll set
   */
  void SetSolverParameters(const TrajOptExampleParams& options,
                           SolverParameters* solver_params) const;

  /**
   * Return a vector that interpolates linearly between q_start and q_end.
   * Useful for setting initial guesses and target trajectories.
   *
   * @param start initial vector
   * @param end final vector
   * @param N number of elements in the linear interpolation
   * @return std::vector<VectorXd> vector that interpolates between start and
   * end.
   */
  std::vector<VectorXd> MakeLinearInterpolation(const VectorXd start,
                                                const VectorXd end,
                                                int N) const {
    std::vector<VectorXd> result;
    double lambda = 0;
    for (int i = 0; i < N; ++i) {
      lambda = i / (N - 1.0);
      result.push_back((1 - lambda) * start + lambda * end);
    }
    return result;
  }
};

}  // namespace examples
}  // namespace traj_opt
}  // namespace drake
