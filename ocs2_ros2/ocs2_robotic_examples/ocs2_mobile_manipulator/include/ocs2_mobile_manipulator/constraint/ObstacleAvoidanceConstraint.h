/*
Obstacle Avoidance Constraint for the Realman Robtic Arm
inherited from the StateConstraint class

written by: Yufei Lei
*/

#pragma once

#include <memory>

#include <pinocchio/fwd.hpp>
#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>
#include <ocs2_self_collision/SelfCollision.h>
#include <ocs2_sphere_approximation/PinocchioSphereInterface.h>
#include <Eigen/Dense>
#include <ocs2_mobile_manipulator/EsdfClientInterface.h>

namespace ocs2{
    namespace mobile_manipulator {

class ObstacleAvoidanceConstraint : public StateConstraint{
    public:
    /**
   * Constructor
   *
   * @param [in] mapping: The pinocchio mapping from pinocchio states to ocs2 states.
   * @param [in] pinocchioInterface: The pinocchio interface of the robot model.
   * @param [in] pinocchioSphereInterface: The pinocchio sphere interface of the robot model.
   * @param [in] esdfClientInterface: The esdf client interface of the sphere center.
   */

    ObstacleAvoidanceConstraint(const PinocchioStateInputMapping<scalar_t>& mapping, const PinocchioSphereInterface &pinocchioSphereInterface, 
                              EsdfClientInterface &esdfClientInterface);

    ~ObstacleAvoidanceConstraint() override = default;

    ObstacleAvoidanceConstraint* clone() const override { return new ObstacleAvoidanceConstraint(*mappingPtr_, pinocchioSphereInterface_, esdfClientInterface_); }

    vector_t getValue(scalar_t time, const vector_t& state, const PreComputation& preComputation) const override;

    std::pair<vector_t, std::vector<Eigen::Vector3d>> getEsdfConstraintValue(const PinocchioInterface &pinocchioInterface, const PinocchioSphereInterface &pinocchioSphereInterface, EsdfClientInterface &esdfClientInterface);
    
    size_t getNumConstraints(scalar_t time) const override;


    VectorFunctionLinearApproximation getLinearApproximation(scalar_t time, const vector_t& state,
                                                            const PreComputation& preComputation) const override;

    VectorFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time, const vector_t& state,
                                                            const PreComputation& preComputation) const override;
    
    private:

    //const PinocchioInterface& getPinocchioInterface(const PreComputation& preComputation) const override;

    ObstacleAvoidanceConstraint(const ObstacleAvoidanceConstraint& rhs);

    const PinocchioSphereInterface &pinocchioSphereInterface_;
    EsdfClientInterface &esdfClientInterface_;
    
    float center_esdf;
    std::vector<float> center_gradients;

    std::unique_ptr<PinocchioStateInputMapping<scalar_t>> mappingPtr_;

    // mutable std::mutex cache_mutex_;
    // mutable scalar_t cached_time_{-1.0};
    // mutable vector_t cached_state_;
    // mutable vector_t cached_constraint_value_;
    // mutable std::vector<Eigen::Vector3d> cached_gradients_;

    // bool isCacheValid(scalar_t time, const vector_t& state) const;

};
    
    }
}
