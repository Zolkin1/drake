//
// Created by zach on 9/28/23.
//

#ifndef DRAKE_DIFF_DRIVE_H
#define DRAKE_DIFF_DRIVE_H

#include "drake/systems/framework/leaf_system.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/planning/trajectory_optimization/multiple_shooting.h"
#include "drake/planning/trajectory_optimization/direct_transcription.h"
#include "drake/geometry/meshcat.h"
#include <Eigen/Dense>

// ----- Create the RoM ----- //
// TODO: Make a RoM parent class and make this inherit from that
template <typename T>
class DiffDrive final : public drake::systems::LeafSystem<T> {
public:
    DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DiffDrive);

    DiffDrive() : DiffDrive<T>(Eigen::VectorXd::Zero(3), 20) {
    }

    DiffDrive(Eigen::VectorXd state_des, int horizon) : drake::systems::LeafSystem<T>(drake::systems::SystemTypeTag<DiffDrive>{}) {
        this->DeclareVectorOutputPort("y", drake::systems::BasicVector<T>(3), &DiffDrive::CopyStateOut);
        this->DeclareContinuousState(3);
        this->DeclareVectorInputPort("u", drake::systems::BasicVector<T>(2));

        DRAKE_DEMAND(state_des.size() == this->num_continuous_states());
        state_des_ = state_des;
        horizon_ = horizon;
    }

    template<typename U>
    explicit DiffDrive(const DiffDrive<U>& c) : DiffDrive<T>(c.GetDesiredState(), c.GetHorizon()) {}

    ~DiffDrive() = default;

    void SetDesiredState(Eigen::VectorXd x_des) {
        DRAKE_DEMAND(x_des.size() == this->num_continuous_states());
        state_des_ = x_des;
    }

    Eigen::VectorXd GetDesiredState() const {
        return state_des_;
    }

    int GetHorizon() const {
        return horizon_;
    }

    void SetPMatrix(Eigen::VectorXd p_vec) {
        int k = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                P_(i,j) = p_vec(k);
                k++;
            }
        }
    }

    void SetRMatrix(Eigen::VectorXd r_vec) {
        int k = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                R_(i,j) = r_vec(k);
                k++;
            }
        }
    }

    void SetQMatrix(Eigen::VectorXd q_vec) {
        int k = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                Q_(i,j) = q_vec(k);
                k++;
            }
        }
    }

    void SetVelocitySaturation(double vel_sat) {
        velocity_saturation_ = vel_sat;
    }

    void SetAngularSaturation(double omega_sat) {
        angular_saturation_ = omega_sat;
    }

    std::vector<Eigen::VectorXd> GenerateXYZTrajectory(const Eigen::VectorXd& q_init,
                                                       const int mpc_num_steps, const double time_step) {
        auto context = this->CreateDefaultContext();

        const drake::planning::trajectory_optimization::TimeStep TS(time_step);

        drake::planning::trajectory_optimization::DirectTranscription trajopt(this, *context,
                                                                              horizon_, TS);

        const drake::solvers::VectorXDecisionVariable& u = trajopt.input();

        // Set input constraints
        trajopt.AddConstraintToAllKnotPoints(-velocity_saturation_ <= u(0));
        trajopt.AddConstraintToAllKnotPoints(u(0) <= velocity_saturation_);

        trajopt.AddConstraintToAllKnotPoints(-angular_saturation_ <= u(1));
        trajopt.AddConstraintToAllKnotPoints(u(1) <= angular_saturation_);


        // Add costs
        trajopt.AddRunningCost(static_cast<drake::MatrixX<drake::symbolic::Expression>>((R_ * u).transpose() * u));

        const drake::solvers::VectorXDecisionVariable& x = trajopt.state();
        trajopt.AddFinalCost(static_cast<drake::MatrixX<drake::symbolic::Expression>>(
                (P_* (x - state_des_)).transpose() * (x-state_des_)));
        trajopt.AddRunningCost(static_cast<drake::MatrixX<drake::symbolic::Expression>>(
                (Q_ * (x - state_des_)).transpose() * (x-state_des_)));

        // Get the mathematical program
        drake::solvers::MathematicalProgram& math_prog = trajopt.prog();

        Eigen::Vector3d q_pos_init;
        // TODO: Do this with slicing
        for (int i = 0; i < 2; i++) {
            q_pos_init[i] = q_init[i + 4];
        }
        q_pos_init[2] = q_init(3);  // Assuming starting on flat ground, theta should just be the w component

        math_prog.AddLinearConstraint(trajopt.initial_state() == q_pos_init);

        // Print all the math program details:
        //std::cout <<  math_prog.to_string() << std::endl;

        // Create initial trajectory
        auto traj_init = drake::trajectories::PiecewisePolynomial<double>::FirstOrderHold(
                {0, time_step*horizon_},
                {q_pos_init, state_des_});
        trajopt.SetInitialTrajectory(drake::trajectories::PiecewisePolynomial<double>(), traj_init);

        // ---------- Solve --------- //
        auto ipopt_solver = drake::solvers::IpoptSolver();
        const auto result = ipopt_solver.Solve(trajopt.prog());
        const drake::trajectories::PiecewisePolynomial<double> x_traj = trajopt.ReconstructStateTrajectory(result);
        const drake::trajectories::PiecewisePolynomial<double> u_traj = trajopt.ReconstructInputTrajectory(result);

        std::vector<Eigen::VectorXd> states;
        for (int i = 0; i < horizon_; i++) {
            Eigen::Matrix<double, -1, 1> full_order_state = Eigen::VectorXd::Zero(q_init.size());
            std::cout << "trajectory x: " << x_traj.value(time_step*i)(0) << std::endl;
            Eigen::VectorXd reduced_state_val = x_traj.value(time_step*i);
            for (int j = 0; j < 2; j++) {
                full_order_state(j+4) = reduced_state_val(j);
            }
            full_order_state(6) = 0.29; //  height of the mini cheetah

            // Assuming being on flat ground this gives the rotation
            full_order_state(0) = 1;
            full_order_state(3) = reduced_state_val(2);

            // Retain values
            generated_trajectory_.push_back(reduced_state_val);
            if (i < mpc_num_steps+1) {
                states.push_back(full_order_state);
            }
        }
        std::cout << std::endl;
        return states;
    }

    void DisplayTrajectory(std::shared_ptr<drake::geometry::Meshcat> meshcat) {
        if (this->generated_trajectory_.size() == 0) {
            std::cerr << "Attempt to display a trajectory of size 0." << std::endl;
            return;
        }

        Eigen::Matrix3Xd start_points = Eigen::Matrix3Xd::Zero(3, this->generated_trajectory_.size());
        Eigen::Matrix3Xd end_points = Eigen::Matrix3Xd::Zero(3, this->generated_trajectory_.size());

        std::cout << "Generated trajectory: " << std::endl;

        Eigen::VectorXd temp = Eigen::VectorXd::Zero(3);
        for (unsigned int i = 0; i < this->generated_trajectory_.size(); i++) {
            temp = this->generated_trajectory_.at(i);
            if (i < this->generated_trajectory_.size() -1) {
                start_points.col(i) = temp;
                start_points(2,i) = 0.29;
            }
            if (i > 0) {
                end_points.col(i-1) = temp;
                end_points(2,i-1) = 0.29;
            }

            std::cout << "----" << std::endl;
            for (int j = 0; j < 3; j++) {
                std::cout << temp(j) << std::endl;
            }
        }

        meshcat->SetLineSegments("/drake", start_points, end_points, 3);
        std::cout << "Sent trajectory to viewer." << std::endl;

    }


private:
    void DoCalcTimeDerivatives(
            const drake::systems::Context<T>& context,
            drake::systems::ContinuousState<T>* derivatives) const override {

        const T v = this->GetInput(context, 0);
        const T omega = this->GetInput(context, 1);

        const T theta = context.get_continuous_state()[2];

        (*derivatives)[0] = v*cos(theta);
        (*derivatives)[1] = v*sin(theta);
        (*derivatives)[2] = omega;
    }

    void CopyStateOut(const drake::systems::Context<T>& context,
                      drake::systems::BasicVector<T>* output) const {
        for (int i = 0; i < 3; i++) {
            const T x = context.get_continuous_state()[i];
            (*output)[i] = x;
        }

    }

    T GetInput(const drake::systems::Context<T>& context, int i) const {
        const drake::systems::BasicVector<T>* u_vec = this->EvalVectorInput(context, 0);
        return u_vec ? u_vec->GetAtIndex(i) : 0.0;
    }

    Eigen::VectorXd state_des_;
    std::vector<Eigen::VectorXd> generated_trajectory_;
    int horizon_;
    Eigen::Matrix3d P_;
    Eigen::Matrix3d Q_;
    Eigen::Matrix2d R_;

    double angular_saturation_;
    double velocity_saturation_;

};

#endif //DRAKE_DIFF_DRIVE_H
