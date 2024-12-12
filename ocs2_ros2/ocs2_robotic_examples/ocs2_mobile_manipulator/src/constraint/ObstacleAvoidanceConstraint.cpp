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
    
    if (isCacheValid(time, state)) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        return cached_constraint_value_;
    }

    const auto& preComp = cast<MobileManipulatorPreComputation>(preComputation);
    const auto& pinocchioInterface_ = preComp.getPinocchioInterface();
    
    auto* mutableThis = const_cast<ObstacleAvoidanceConstraint*>(this);
    
    auto result = mutableThis->getEsdfConstraintValue(pinocchioInterface_, pinocchioSphereInterface_, esdfClientInterface_);
    
    auto end_total = std::chrono::high_resolution_clock::now();
    auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total);
    RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "Total getValue time: %ld microseconds", duration_total.count());
    
    updateCache(time, state, result.first, result.second);
    return result.first;
}

std::pair<vector_t, std::vector<Eigen::Vector3d>> ObstacleAvoidanceConstraint::getEsdfConstraintValue(
    const PinocchioInterface &pinocchioInterface, 
    const PinocchioSphereInterface &pinocchioSphereInterface,
    EsdfClientInterface &esdfClientInterface) {
    
    auto start_spheres = std::chrono::high_resolution_clock::now();
    
    const auto& sphereCenters = pinocchioSphereInterface.computeSphereCentersInWorldFrame(pinocchioInterface);
    std::vector<Eigen::Vector3d> adjustedSphereCenters = sphereCenters;
    const auto& sphereRadii = pinocchioSphereInterface.getSphereRadii();
    
    auto end_spheres = std::chrono::high_resolution_clock::now();
    auto duration_spheres = std::chrono::duration_cast<std::chrono::microseconds>(end_spheres - start_spheres);
    //RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "Sphere computation time: %ld microseconds", duration_spheres.count());

    auto start_esdf = std::chrono::high_resolution_clock::now();
    // Convert time_point to duration since epoch and print
    //std::cout << "start_esdf: " << std::chrono::duration_cast<std::chrono::microseconds>(start_esdf.time_since_epoch()).count() << " microseconds" << std::endl;

    std::vector<float> esdfValue;
    std::vector<Eigen::Vector3d> gradients;
    EsdfClientInterface::EsdfResponse esdfResponse = esdfClientInterface.getEsdf(adjustedSphereCenters);
    esdfValue = esdfResponse.esdf_values;
    gradients = esdfResponse.gradients;
    
    auto end_esdf = std::chrono::high_resolution_clock::now();
    //std::cout << "end_esdf: " << std::chrono::duration_cast<std::chrono::microseconds>(end_esdf.time_since_epoch()).count() << " microseconds" << std::endl;
    
    auto duration_esdf = std::chrono::duration_cast<std::chrono::microseconds>(end_esdf - start_esdf);
    RCLCPP_INFO(rclcpp::get_logger("ObstacleAvoidanceConstraint"), "ESDF query time: %ld microseconds", duration_esdf.count());

    auto start_constraint = std::chrono::high_resolution_clock::now();
    
    vector_t constraintValue = vector_t::Zero(esdfValue.size());
    for (size_t i = 0; i < esdfValue.size(); ++i) {
        constraintValue[i] = fabs(esdfValue[i]) - sphereRadii[i];
        if (constraintValue[i] > 1) {
            constraintValue[i] = 1;
        }
    }
    
    auto end_constraint = std::chrono::high_resolution_clock::now();
    auto duration_constraint = std::chrono::duration_cast<std::chrono::microseconds>(end_constraint - start_constraint);
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
    std::cout << "----------------getLinearApproximation----------------" << std::endl;
    // 1. 获取预计算数据
    const auto& preComp = cast<MobileManipulatorPreComputation>(preComputation);
    const auto& pinocchioInterface_ = preComp.getPinocchioInterface();
    mappingPtr_->setPinocchioInterface(pinocchioInterface_);

    VectorFunctionLinearApproximation constraint;
    matrix_t dfdq, dfdv;
    
    // 2. 计算约束值和梯度
    auto* mutableThis = const_cast<ObstacleAvoidanceConstraint*>(this);
    auto [constraintValues, gradients] = mutableThis->getEsdfConstraintValue(pinocchioInterface_, 
                                                                           pinocchioSphereInterface_, 
                                                                           esdfClientInterface_);
    constraint.f = constraintValues;

    // 3. 计算雅可比矩阵
    // 获取机器人的雅可比矩阵
    const auto& model = pinocchioInterface_.getModel();
    const auto& data = pinocchioInterface_.getData();
    
    // 为每个球体计算雅可比矩阵
    const auto& spherePositions = pinocchioSphereInterface_.computeSphereCentersInWorldFrame(pinocchioInterface_);
    dfdq.setZero(spherePositions.size(), model.nq);
    
    for (size_t i = 0; i < spherePositions.size(); ++i) {
        // 获取球体所在关节的雅可比矩阵
        matrix_t sphereJacobian = matrix_t::Zero(6, model.nv);
        pinocchio::getJointJacobian(model, data, i+1, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, sphereJacobian);
        
        // 将ESDF梯???与关节雅可比相乘
        dfdq.row(i) = gradients[i].transpose() * sphereJacobian.topRows(3);
    }

    std::cout << "dfdq size: " << dfdq.rows() << " " << dfdq.cols() << std::endl;
    for (size_t i = 0; i < dfdq.rows(); ++i) {
        std::cout << "dfdq row " << i << ": " << dfdq.row(i).transpose() << std::endl;
    }
    
    // 4. 映射到OCS2状态空间
    dfdv.setZero(dfdq.rows(), dfdq.cols());
    std::tie(constraint.dfdx, std::ignore) = mappingPtr_->getOcs2Jacobian(state, dfdq, dfdv);

    std::cout << "dfdx size: " << constraint.dfdx.rows() << " " << constraint.dfdx.cols() << std::endl;
    for (size_t i = 0; i < constraint.dfdx.rows(); ++i) {
        std::cout << "dfdx row " << i << ": " << constraint.dfdx.row(i).transpose() << std::endl;
    }
    
    // 检查雅可比矩阵的数值稳定性
    for (size_t i = 0; i < spherePositions.size(); ++i) {
        matrix_t sphereJacobian = matrix_t::Zero(6, model.nv);
        pinocchio::getJointJacobian(model, data, i+1, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, sphereJacobian);
        
        // 检查雅可比矩阵的条件数
        Eigen::JacobiSVD<matrix_t> svd(sphereJacobian);
        double conditionNumber = svd.singularValues()(0) / 
                               svd.singularValues()(svd.singularValues().size()-1);
        
        if (conditionNumber > 1e6) {  // 设置一个合理的阈值
            throw std::runtime_error("Jacobian poorly conditioned: " + std::to_string(conditionNumber));
        }
    }
    
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




}//namespace mobile_manipulator
}//namespace ocs2