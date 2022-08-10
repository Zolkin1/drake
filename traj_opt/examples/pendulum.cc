#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/traj_opt/problem_definition.h"
#include "drake/traj_opt/trajectory_optimizer.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace pendulum {

// Command line options
DEFINE_double(time_step, 5e-2,
              "Discretization timestep for the optimizer (seconds).");
DEFINE_int32(num_steps, 40,
             "Number of timesteps in the optimization problem.");
DEFINE_int32(max_iters, 100,
             "Maximum number of Gauss-Newton iterations to take.");
DEFINE_double(Qq, 0.0, "Running cost weight on the joint angle.");
DEFINE_double(Qv, 0.1, "Running cost weight on the joint velocity.");
DEFINE_double(R, 1.0, "Running cost weight on control inputs.");
DEFINE_double(Qfq, 100.0, "Terminal cost weight on the joint angle.");
DEFINE_double(Qfv, 1.0, "Terminal cost weight on the joint velocity.");
DEFINE_bool(save_data, false, "Flag for writing solver data to a csv file.");
DEFINE_bool(visualize, true, "Flag for displaying the optimal solution.");
DEFINE_double(gravity, 9.81, "Magnitude of gravity in the z-direction.");
DEFINE_string(linesearch, "armijo",
              "Linesearch strategy, {backtracking} or {armijo}.");

using geometry::DrakeVisualizerd;
using geometry::SceneGraph;
using multibody::AddMultibodyPlant;
using multibody::MultibodyPlantConfig;
using multibody::Parser;
using systems::DiagramBuilder;
using systems::Simulator;

/**
 * Save the solution data to a CSV file that we can process and make plots from
 * later.
 *
 * @param solution_data struct containing iteration and timing data
 */
void save_to_csv(const SolutionData<double>& data) {
  // Set file to write to
  std::ofstream data_file;
  data_file.open("pendulum_data.csv");

  // Write a header
  data_file
      << "iter, time, cost, ls_iters, alpha, grad_norm\n";

  const int num_iters = data.iteration_times.size();
  for (int i = 0; i < num_iters; ++i) {
    // Write the data
    data_file << i << ", ";
    data_file << data.iteration_times[i] << ", ";
    data_file << data.iteration_costs[i] << ", ";
    data_file << data.linesearch_iterations[i] << ", ";
    data_file << data.linesearch_alphas[i] << ", ";
    data_file << data.gradient_norm[i] << "\n";
  }

  // Close the file
  data_file.close();
}

/**
 * Play back the given trajectory on the Drake visualizer.
 *
 * @param q sequence of generalized positions defining the trajectory
 * @param time_step time step (seconds) for the discretization
 */
void play_back_trajectory(std::vector<VectorXd> q, double time_step) {
  // TODO(vincekurtz): verify size of q
  DiagramBuilder<double> builder;
  MultibodyPlantConfig config;
  config.time_step = time_step;

  auto [plant, scene_graph] = AddMultibodyPlant(config, &builder);

  const std::string urdf_file =
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.Finalize();

  DrakeVisualizerd::AddToBuilder(&builder, scene_graph);

  auto diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  const double u = 0;
  plant.get_actuation_input_port().FixValue(&plant_context, u);

  const int N = q.size();
  for (int t = 0; t < N; ++t) {
    diagram_context->SetTime(t * time_step);
    plant.SetPositions(&plant_context, q[t]);
    diagram->Publish(*diagram_context);

    // Hack to make the playback roughly realtime
    std::this_thread::sleep_for(std::chrono::duration<double>(time_step));
  }
}

/**
 * Solve a trajectory optimization problem that swings the pendulum upright.
 *
 * Then play back an animation of the optimal trajectory using the Drake
 * visualizer.
 *
 * @param time_step Time step for discretization (seconds)
 * @param num_steps Number of steps in the optimization problem.
 */
void solve_trajectory_optimization(double time_step, int num_steps) {
  // Create a system model
  MultibodyPlant<double> plant(time_step);
  const std::string urdf_file =
      FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf");
  Parser(&plant).AddAllModelsFromFile(urdf_file);
  plant.mutable_gravity_field().set_gravity_vector(
      Eigen::Vector3d(0, 0, -FLAGS_gravity));
  plant.Finalize();

  // Set up an optimization problem
  ProblemDefinition opt_prob;
  opt_prob.num_steps = num_steps;
  opt_prob.q_init = Vector1d(0.0);
  opt_prob.v_init = Vector1d(0.0);
  opt_prob.Qq = FLAGS_Qq * MatrixXd::Identity(1, 1);
  opt_prob.Qv = FLAGS_Qv * MatrixXd::Identity(1, 1);
  opt_prob.Qf_q = FLAGS_Qfq * MatrixXd::Identity(1, 1);
  opt_prob.Qf_v = FLAGS_Qfv * MatrixXd::Identity(1, 1);
  opt_prob.R = FLAGS_R * MatrixXd::Identity(1, 1);
  opt_prob.q_nom = Vector1d(M_PI);
  opt_prob.v_nom = Vector1d(0.0);

  // Set our solver options
  SolverParameters solver_params;
  if (FLAGS_linesearch == "backtracking") {
    solver_params.linesearch_method = LinesearchMethod::kBacktracking;
  } else if (FLAGS_linesearch == "armijo") {
    solver_params.linesearch_method = LinesearchMethod::kArmijo;
  } else {
    throw std::runtime_error("Unknown linesearch method '" + FLAGS_linesearch +
                             "'.");
  }
  solver_params.max_iterations = FLAGS_max_iters;
  solver_params.max_linesearch_iterations = 50;

  // Establish an initial guess
  std::vector<VectorXd> q_guess;
  for (int t = 0; t <= num_steps; ++t) {
    q_guess.push_back(opt_prob.q_init);
  }

  // Solve the optimzation problem
  TrajectoryOptimizer<double> optimizer(&plant, opt_prob, solver_params);
  Solution<double> solution;
  SolutionData<double> solution_data;

  SolverFlag status = optimizer.Solve(q_guess, &solution, &solution_data);
  DRAKE_ASSERT(status == SolverFlag::kSuccess);
  std::cout << "Solved in " << solution_data.solve_time << " seconds."
            << std::endl;

  // Save data to CSV, if requested
  if (FLAGS_save_data) {
    save_to_csv(solution_data);
  }

  // Play back the result on the visualizer
  if (FLAGS_visualize) {
    play_back_trajectory(solution.q, time_step);
  }
}

int do_main() {
  // For now we'll just run a simple passive simulation of the pendulum
  // run_passive_simulation(1e-2, 2.0);

  // Solve an optimization problem to swing-up the pendulum
  solve_trajectory_optimization(FLAGS_time_step, FLAGS_num_steps);

  return 0;
}

}  // namespace pendulum
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::traj_opt::examples::pendulum::do_main();
}
