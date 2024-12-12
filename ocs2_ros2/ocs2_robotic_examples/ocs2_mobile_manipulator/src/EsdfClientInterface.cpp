/*
EsdfClientInterface.cpp
written by: Yufei Lei
*/
#include "ocs2_mobile_manipulator/EsdfClientInterface.h"
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>

namespace ocs2{
    namespace mobile_manipulator{
    

EsdfClientInterface::EsdfClientInterface(const std::string& node_name, const std::string& service_name)
    : Node(node_name) {
    client_ = this->create_client<nvblox_msgs::srv::VoxelEsdfAndGradients>(service_name);
    
    while (!client_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_INFO(this->get_logger(), "Waiting for service '%s' to become available...", service_name.c_str());
}
    RCLCPP_INFO(this->get_logger(), "Service '%s' is now available.", service_name.c_str());
}

nvblox_msgs::srv::VoxelEsdfAndGradients::Response EsdfClientInterface::callEsdfService(std::vector<Eigen::Vector3d>& link_positions){
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!client_->service_is_ready()) {
        RCLCPP_ERROR(this->get_logger(), "Service is not ready.");
        return nvblox_msgs::srv::VoxelEsdfAndGradients::Response();
    }

    // Measure request preparation time
    // auto request_start = std::chrono::high_resolution_clock::now();
    auto request = std::make_shared<nvblox_msgs::srv::VoxelEsdfAndGradients::Request>();

    for (const auto& position : link_positions) {
        geometry_msgs::msg::Point point;
        point.x = position(0);
        point.y = position(1);
        point.z = position(2);
        request->link_positions.push_back(point);
    }
    //auto request_end = std::chrono::high_resolution_clock::now();

    // Measure service call time
    //auto service_start = std::chrono::high_resolution_clock::now();
    auto future = client_->async_send_request(request);

    auto service_stage1 = std::chrono::high_resolution_clock::now();
    
    if (rclcpp::spin_until_future_complete(
            this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS) {
            
        auto service_stage2 = std::chrono::high_resolution_clock::now();

        auto response = future.get();

        //auto service_stage3 = std::chrono::high_resolution_clock::now();

        //auto service_end = std::chrono::high_resolution_clock::now();

        if (response) {
            nvblox_msgs::srv::VoxelEsdfAndGradients::Response esdf_response_ = *response;
            has_esdf_response_ = true;

            // Calculate and log timing information
            //auto request_duration = std::chrono::duration_cast<std::chrono::microseconds>(request_end - request_start).count();
            //auto service_duration = std::chrono::duration_cast<std::chrono::microseconds>(service_end - service_start).count();
            //auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(service_end - start_time).count();

            //RCLCPP_INFO(this->get_logger(), "Timing - Request prep: %ld us, Service call: %ld us, Total: %ld us",
            //            request_duration, service_duration, total_duration);
            // auto service_duration0 = std::chrono::duration_cast<std::chrono::microseconds>(service_stage1 - service_start).count();
            auto service_duration1 = std::chrono::duration_cast<std::chrono::microseconds>(service_stage2 - service_stage1).count();
            // auto service_duration2 = std::chrono::duration_cast<std::chrono::microseconds>(service_stage3 - service_stage2).count();
            // auto service_duration3 = std::chrono::duration_cast<std::chrono::microseconds>(service_end - service_stage3).count();
            // RCLCPP_INFO(this->get_logger(), "Timing - Service call: %ld us, %ld us, %ld us, %ld us",
            //             service_duration0, service_duration1, service_duration2, service_duration3);
            RCLCPP_INFO(this->get_logger(), "Timing - Service call: %ld us", service_duration1);

            return esdf_response_;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Empty response from ESDF service.");
            return nvblox_msgs::srv::VoxelEsdfAndGradients::Response();
        }
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to call ESDF service.");
        return nvblox_msgs::srv::VoxelEsdfAndGradients::Response();
    }
}

// void EsdfClientInterface::setPoint(const Eigen::Vector3d& point) {
//     point_ = point;
//     RCLCPP_INFO(this->get_logger(), "Updated point: [%f, %f, %f]", point_.x(), point_.y(), point_.z());
// }


// void EsdfClientInterface::setDirection(const Eigen::Vector3d& direction) {
//     direction_ = direction;
//     RCLCPP_INFO(this->get_logger(), "Updated direction: [%f, %f, %f]", direction_.x(), direction_.y(), direction_.z());
// }

EsdfClientInterface::EsdfResponse EsdfClientInterface::getEsdf(std::vector<Eigen::Vector3d>& link_positions){
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(service_mutex_);
    
    // 调用服务前的时间
    auto before_service = std::chrono::high_resolution_clock::now();
    nvblox_msgs::srv::VoxelEsdfAndGradients::Response esdf_response_ = callEsdfService(link_positions);
    // 调用服务后的时间
    auto after_service = std::chrono::high_resolution_clock::now();
    
    if (has_esdf_response_ && esdf_response_.valid) {
        EsdfResponse esdf_response;
        esdf_response.esdf_values = esdf_response_.esdf_values.data;
        esdf_response.gradients.reserve(esdf_response_.gradients.size());
        
        // 数据处理开始时间
        auto process_start = std::chrono::high_resolution_clock::now();
        for (const auto& gradient : esdf_response_.gradients) {
            Eigen::Vector3d gradient_eigen;
            gradient_eigen << gradient.x, gradient.y, gradient.z;
            esdf_response.gradients.push_back(gradient_eigen);
        }
        // 计算并打印各阶段耗时
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto service_duration = std::chrono::duration_cast<std::chrono::microseconds>(after_service - before_service).count();
        auto process_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - process_start).count();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        
        // RCLCPP_INFO(this->get_logger(), "Time breakdown - Service call: %ld us, Data processing: %ld us, Total: %ld us",
        //             service_duration, process_duration, total_duration);
        
        return esdf_response;
    } else {
        RCLCPP_ERROR(this->get_logger(), "No ESDF response available.");
        return EsdfResponse();
    }
}


}  // namespace mobile_manipulator
}  // namespace ocs2