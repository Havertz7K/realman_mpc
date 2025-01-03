/*
ObstacleAvoidanceConstraint.cpp

written by: Yufei Lei
*/
#include <ocs2_mobile_manipulator/constraint/ObstacleAvoidanceConstraint.h>
#include <ocs2_mobile_manipulator/MobileManipulatorPreComputation.h>

#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/geometry.hpp>

namespace ocs2{
    namespace mobile_manipulator{

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ObstacleAvoidanceConstraint::ObstacleAvoidanceConstraint(
    const PinocchioStateInputMapping<scalar_t>& mapping, 
    const PinocchioSphereInterface& pinocchioSphereInterface, 
    EsdfClientInterface& esdfClientInterface)
    : StateConstraint(ConstraintOrder::Linear),
      pinocchioSphereInterface_(pinocchioSphereInterface), 
      esdfClientInterface_(esdfClientInterface),
      mappingPtr_(mapping.clone()) {}

/******************************************************************************************************/
/******************************************************************************************************/
ObstacleAvoidanceConstraint::ObstacleAvoidanceConstraint(const ObstacleAvoidanceConstraint& rhs)
    : StateConstraint(rhs), pinocchioSphereInterface_(rhs.pinocchioSphereInterface_), esdfClientInterface_(rhs.esdfClientInterface_),
    mappingPtr_(rhs.mappingPtr_->clone()) {}

vector_t ObstacleAvoidanceConstraint::getValue(scalar_t time, const vector_t& state, const PreComputation& preComputation) const{
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // if (isCacheValid(time, state)) {
    //     std::lock_guard<std::mutex> lock(cache_mutex_);
    //     return cached_constraint_value_;
    // }
    const auto& pinocchioInterface_ = cast<MobileManipulatorPreComputation>(preComputation).getPinocchioInterface();
    
    auto* mutableThis = const_cast<ObstacleAvoidanceConstraint*>(this);
    
    auto result = mutableThis->getEsdfConstraintValue(pinocchioInterface_, pinocchioSphereInterface_, esdfClientInterface_);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    //RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "Total getValue time: %ld microseconds", duration_total.count());
    
    // std::lock_guard<std::mutex> lock(cache_mutex_);
    // cached_time_ = time;
    // cached_state_ = state;
    // cached_constraint_value_ = result.first;
    // cached_gradients_ = result.second;
    
    return result.first;
}

std::pair<vector_t, std::vector<Eigen::Vector3d>> ObstacleAvoidanceConstraint::getEsdfConstraintValue(
    const PinocchioInterface &pinocchioInterface, 
    const PinocchioSphereInterface &pinocchioSphereInterface,
    EsdfClientInterface &esdfClientInterface) {
    
    //auto start_spheres = std::chrono::high_resolution_clock::now();
    
    const auto& sphereCenters = pinocchioSphereInterface.computeSphereCentersInWorldFrame(pinocchioInterface);

    // for(int i = 0; i < sphereCenters.size(); i++){
    //     std::cout << "sphereCenters[" << i << "]: " << sphereCenters[i].transpose() << std::endl;
    // }

    std::vector<Eigen::Vector3d> adjustedSphereCenters = sphereCenters;
    const auto& sphereRadii = pinocchioSphereInterface.getSphereRadii();

    //RCLCPP_INFO(rclcpp::get_logger("radius"), "sphereRadii: [%f,%f,%f,%f,%f,%f,%f]", sphereRadii[0], sphereRadii[1], sphereRadii[2], sphereRadii[3], sphereRadii[4], sphereRadii[5], sphereRadii[6]);
    
    //auto end_spheres = std::chrono::high_resolution_clock::now();
    //auto duration_spheres = std::chrono::duration_cast<std::chrono::microseconds>(end_spheres - start_spheres);
    //RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "Sphere computation time: %ld microseconds", duration_spheres.count());

    //auto start_esdf = std::chrono::high_resolution_clock::now();
    // Convert time_point to duration since epoch and print
    //std::cout << "start_esdf: " << std::chrono::duration_cast<std::chrono::microseconds>(start_esdf.time_since_epoch()).count() << " microseconds" << std::endl;

    std::vector<float> esdfValue;
    std::vector<Eigen::Vector3d> gradients;
    EsdfClientInterface::EsdfResponse esdfResponse = esdfClientInterface.getEsdf(adjustedSphereCenters);
    esdfValue = esdfResponse.esdf_values;
    gradients = esdfResponse.gradients;
    
    //auto end_esdf = std::chrono::high_resolution_clock::now();
    //std::cout << "end_esdf: " << std::chrono::duration_cast<std::chrono::microseconds>(end_esdf.time_since_epoch()).count() << " microseconds" << std::endl;
    
    //auto duration_esdf = std::chrono::duration_cast<std::chrono::microseconds>(end_esdf - start_esdf);
    //RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "ESDF query time: %ld microseconds", duration_esdf.count());

    //auto start_constraint = std::chrono::high_resolution_clock::now();
    
    vector_t constraintValue = vector_t::Zero(esdfValue.size());
    for (size_t i = 0; i < esdfValue.size(); ++i) {
        constraintValue[i] = fabs(esdfValue[i]) - sphereRadii[i];
        // if (constraintValue[i] > 1) {
        //     constraintValue[i] = 1;
        // }
    }
    
    //auto end_constraint = std::chrono::high_resolution_clock::now();
    //auto duration_constraint = std::chrono::duration_cast<std::chrono::microseconds>(end_constraint - start_constraint);
    //RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "Constraint computation time: %ld microseconds", duration_constraint.count());

    return std::make_pair(constraintValue, gradients);
}

size_t ObstacleAvoidanceConstraint::getNumConstraints(scalar_t time) const{
    return 7;
}

/******************************************************************************************************/
/******************************************************************************************************/
//linear approximation dfdq dfdu
VectorFunctionLinearApproximation ObstacleAvoidanceConstraint::getLinearApproximation(scalar_t time, const vector_t& state,
                                                            const PreComputation& preComputation) const {   
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // 1. 预计算数据获取
    auto start_precomp = std::chrono::high_resolution_clock::now();
    const auto& preComp = cast<MobileManipulatorPreComputation>(preComputation);
    const auto& pinocchioInterface_ = preComp.getPinocchioInterface();
    mappingPtr_->setPinocchioInterface(pinocchioInterface_);
    auto end_precomp = std::chrono::high_resolution_clock::now();
    
    VectorFunctionLinearApproximation constraint;
    matrix_t dfdq, dfdv;
    
    // 2. 约束值和梯度计算
    auto start_constraint = std::chrono::high_resolution_clock::now();
    auto* mutableThis = const_cast<ObstacleAvoidanceConstraint*>(this);
    auto [constraintValues, gradients] = mutableThis->getEsdfConstraintValue(pinocchioInterface_, 
                                                                           pinocchioSphereInterface_, 
                                                                           esdfClientInterface_);
    constraint.f = constraintValues;
    auto end_constraint = std::chrono::high_resolution_clock::now();

    // 3. 获取模型数据
    auto start_model = std::chrono::high_resolution_clock::now();
    const auto& model = pinocchioInterface_.getModel();
    const auto& data = pinocchioInterface_.getData();
    const auto& sphereCenters = pinocchioSphereInterface_.computeSphereCentersInWorldFrame(pinocchioInterface_);
    auto end_model = std::chrono::high_resolution_clock::now();

    // 4. 计算关节位置和偏移量
    auto start_positions = std::chrono::high_resolution_clock::now();
    std::vector<Eigen::Vector3d> jointPositions;
    jointPositions.reserve(sphereCenters.size());
    for (size_t i = 1; i <= sphereCenters.size(); ++i) {
        jointPositions.push_back(data.oMi[i].translation());
    }

    std::vector<Eigen::Vector3d> sphereOffsets;
    sphereOffsets.reserve(sphereCenters.size());
    for (size_t i = 0; i < sphereCenters.size(); ++i) {
        sphereOffsets.push_back(sphereCenters[i] - jointPositions[i]);
    }
    auto end_positions = std::chrono::high_resolution_clock::now();

    // 5. 计算雅可比矩阵
    auto start_jacobian = std::chrono::high_resolution_clock::now();
    dfdq.setZero(sphereCenters.size(), model.nq);
    
    for (size_t i = 0; i < sphereCenters.size(); ++i) {
        pinocchio::JointIndex jointId = i + 1;
        matrix_t J = matrix_t::Zero(6, model.nv);
        const auto& M = data.oMi[jointId];
        Eigen::Vector3d p = sphereOffsets[i];
        
        pinocchio::getJointJacobian(model, data, jointId, 
                                   pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
                                   J);
                                   
        matrix_t Jcorrected = matrix_t::Zero(6, model.nv);
        Jcorrected.topRows(3) = J.topRows(3);
        
        Eigen::Matrix3d R = M.rotation();
        Eigen::Matrix3d pSkew;
        pSkew << 0, -p(2), p(1),
                 p(2), 0, -p(0),
                -p(1), p(0), 0;
                
        Jcorrected.topRows(3) += R * pSkew * J.bottomRows(3);
        dfdq.row(i) = -gradients[i].transpose() * Jcorrected.topRows(3);
    }
    auto end_jacobian = std::chrono::high_resolution_clock::now();

    // 6. 状态空间映射
    auto start_mapping = std::chrono::high_resolution_clock::now();
    dfdv.setZero(dfdq.rows(), dfdq.cols());
    std::tie(constraint.dfdx, std::ignore) = mappingPtr_->getOcs2Jacobian(state, dfdq, dfdv);
    auto end_mapping = std::chrono::high_resolution_clock::now();
    
    auto end_total = std::chrono::high_resolution_clock::now();

    // 打印各阶段耗时
    // RCLCPP_INFO(rclcpp::get_logger("Timing"), "\n"
    //     "Total time: %ld us\n"
    //     "1. Precomputation: %ld us\n"
    //     "2. Constraint calculation: %ld us\n"
    //     "3. Model data: %ld us\n"
    //     "4. Position calculation: %ld us\n"
    //     "5. Jacobian calculation: %ld us\n"
    //     "6. State mapping: %ld us",
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count(),
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_precomp - start_precomp).count(),
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_constraint - start_constraint).count(),
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_model - start_model).count(),
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_positions - start_positions).count(),
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_jacobian - start_jacobian).count(),
    //     std::chrono::duration_cast<std::chrono::microseconds>(end_mapping - start_mapping).count());
    
    return constraint;
}

VectorFunctionQuadraticApproximation ObstacleAvoidanceConstraint::getQuadraticApproximation(scalar_t time, const vector_t& state,
                                                            const PreComputation& preComputation) const {
    std::cout << "----------------getQuadraticApproximation----------------" << std::endl;
    VectorFunctionQuadraticApproximation constraint;

    auto linearApprox = getLinearApproximation(time, state, preComputation);
    constraint.f = std::move(linearApprox.f);
    constraint.dfdx = std::move(linearApprox.dfdx);
    constraint.dfdu = std::move(linearApprox.dfdu);

    const auto inputDim_ = 7;
    // TODO: 二阶导数设置为0
    constraint.dfdxx.assign(constraint.f.size(), matrix_t::Zero(state.size(), state.size()));
    constraint.dfdux.assign(constraint.f.size(), matrix_t::Zero(inputDim_, state.size()));
    constraint.dfduu.assign(constraint.f.size(), matrix_t::Zero(inputDim_, inputDim_));

    return constraint;
}

// const PinocchioInterface& ObstacleAvoidanceConstraint::getPinocchioInterface(const PreComputation& preComputation) const{
//     return cast<MobileManipulatorPreComputation>(preComputation).getPinocchioInterface();
// }

// bool ObstacleAvoidanceConstraint::isCacheValid(scalar_t time, const vector_t& state) const {
//     std::lock_guard<std::mutex> lock(cache_mutex_);
//     return (time == cached_time_) && (state == cached_state_);
// }


}//namespace mobile_manipulator
}//namespace ocs2