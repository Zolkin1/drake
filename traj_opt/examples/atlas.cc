#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/traj_opt/examples/example_base.h"

namespace drake {
namespace traj_opt {
namespace examples {
namespace atlas {

using Eigen::Vector3d;
using geometry::Box;
using geometry::Cylinder;
using geometry::Sphere;
using math::RigidTransformd;
using math::RollPitchYawd;
using multibody::CoulombFriction;
using multibody::ModelInstanceIndex;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::RigidBody;
using multibody::SpatialInertia;
using multibody::UnitInertia;

class AtlasExample : public TrajOptExample {
  void CreatePlantModel(MultibodyPlant<double>* plant) const final {
    const Vector4<double> blue(0.1, 0.3, 0.5, 1.0);
    const Vector4<double> green(0.3, 0.6, 0.4, 1.0);
    const Vector4<double> black(0.0, 0.0, 0.0, 1.0);

    // Add the atlas model
    std::string urdf_file =
        FindResourceOrThrow("drake/traj_opt/examples/atlas.urdf");
    Parser(plant).AddAllModelsFromFile(urdf_file);

    // Turn off gravity
    plant->mutable_gravity_field().set_gravity_vector(Vector3d(0, 0, 0));


  }
};

}  // namespace atlas
}  // namespace examples
}  // namespace traj_opt
}  // namespace drake

int main() {
  drake::traj_opt::examples::atlas::AtlasExample example;
  example.SolveTrajectoryOptimization("drake/traj_opt/examples/atlas.yaml");
  return 0;
}
