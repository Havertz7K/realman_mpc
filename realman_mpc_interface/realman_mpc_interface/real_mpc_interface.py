import rclpy
from rclpy.node import Node
from ocs2_msgs.msg import MpcFlattenedController, MpcObservation, MpcTargetTrajectories, MpcState, MpcInput
from ocs2_msgs.srv import Reset, GenerateTraj
from tf2_ros.buffer import Buffer
from tf2_ros import TransformStamped
from tf2_ros.transform_listener import TransformListener
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import JointState
from rclpy.clock import Clock
import threading
import numpy as np
from .utils import generate_traj, interpolate_trajectory, find_nearest_timestamp_index

import bisect
from typing import List


class RealMpcInterface(Node):
    def __init__(self, arm_name, mpc_freq=100, control_freq=200):
        super().__init__(f'{arm_name}_mpc_message_convert')

        self.mpc_freq = mpc_freq
        self.control_freq = control_freq

        # MPC Observation Publisher
        self.mpc_observation_pub = self.create_publisher(
            MpcObservation, f'{arm_name}_mpc_observation', 1)
        self.mpc_observation = MpcObservation()
        self.mpc_observation_lock = threading.Lock()
        self.mpc_timer = self.create_timer(
            1.0/self.mpc_freq, self.mpc_timer_callback)

        # Joint States Subscriber
        self.current_joint = None
        self.joint_states_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_states_callback, 1)

        # MPC Policy Subscriber
        self.mpc_policy_sub = self.create_subscription(
            MpcFlattenedController, f'{arm_name}_mpc_policy', self.mpc_policy_callback, 1)
        self.mpc_policy = MpcFlattenedController()

        # Position Control Command Publisher

        self.position_command_pub = self.create_publisher(
            Jointpos, '/rm_driver/movej_canfd_cmd_fake', 1)
        # self.position_command_pub = self.create_publisher(Jointpos, '/rm_driver/movej_canfd_cmd', 1)
        self.control_timer = self.create_timer(
            1.0/self.control_freq, self.control_timer_callback)

        self.init_mpc()
        # trajectory
        self.init_tf()
        self.arm_pose = None
        self.generate_traj_server = self.create_service(
            GenerateTraj, 'generate_trajectory', self.traj_callback)
        self.publisher_ = self.create_publisher(
            MpcTargetTrajectories, f'{arm_name}_mpc_target', 1)

        self.get_logger().info('Gazebo MPC Interface Node Started')

    def init_mpc(self):
        self.current_input = np.zeros(7)
        self.mpc_observation.input.value = list(self.current_input)
        self.start_time = self.get_clock().now()

    def init_tf(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.source_frame = 'world'  # 
        self.target_frame = 'gripper_tip_link'
        # 
        self.tf_timer = self.create_timer(0.01, self.get_end_effect_transform)

    def get_end_effect_transform(self):
        try:
            
            if self.target_frame is None:
                self.get_logger().error("No valid target frame found.")
                return
            
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                rclpy.time.Time())

            self.arm_pose = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        except Exception as e:
            self.get_logger().error(
                f"Error getting end effector transform: {e}")

    def traj_callback(self, request, response):

        goal_pose = request.goal_pose
        time = request.time
        timestep = request.timestep
        time_now = self.get_elapsed_time()
        time_trajectory, state_trajectory = generate_traj(
            self.arm_pose, goal_pose, time_now, time_now + time, timestep)
        self.get_logger().info('Generated trajectory for goal pose: %s' % str(goal_pose))

        target_trajectories = MpcTargetTrajectories()
        target_trajectories.time_trajectory = [
            float(i) for i in time_trajectory]
        target_trajectories.state_trajectory = [
            MpcState(value=i) for i in state_trajectory]
        target_trajectories.input_trajectory = [
            MpcInput(value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for _ in time_trajectory]

        self.publisher_.publish(target_trajectories)
        self.get_logger().info('Published target trajectories')
        response.done = True

        return response

    def get_elapsed_time(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.start_time).nanoseconds * 1e-9
        return float(elapsed_time)

    def mpc_timer_callback(self):
        current_time = self.get_elapsed_time()
        with self.mpc_observation_lock:
            if len(self.mpc_observation.state.value) > 0:
                self.mpc_observation_pub.publish(self.mpc_observation)
        self.get_logger().info(f'>>> observation publish at {current_time}')

    def joint_states_callback(self, msg: JointState):
        with self.mpc_observation_lock:
            self.mpc_observation.time = self.get_elapsed_time()

            # joint_state
            if 'joint7' in msg.name:
                joint_names = ['joint1', 'joint2', 'joint3',
                               'joint4', 'joint5', 'joint6', 'joint7']
            else:
                joint_names = ['joint1', 'joint2', 'joint3',
                               'joint4', 'joint5', 'joint6']
            self.mpc_observation.state.value = [
                msg.position[msg.name.index(joint_name)] for joint_name in joint_names]
            self.current_joint = [msg.position[msg.name.index(
                joint_names)] for joint_name in joint_names]

    def mpc_policy_callback(self, msg: MpcFlattenedController):
        self.mpc_policy = msg
        self.get_logger().info(
            f'<<< New MPC policy start at {self.mpc_policy.time_trajectory[0]}')

    def control_timer_callback(self):

        current_time = self.get_elapsed_time()
        if len(self.mpc_policy.input_trajectory) > 0:
            # index = find_nearest_timestamp_index(self.mpc_policy.time_trajectory, current_time)
            # current_input = self.mpc_policy.input_trajectory[index].value
            # 
            current_input = interpolate_trajectory(
                self.mpc_policy.time_trajectory,
                [input.value for input in self.mpc_policy.input_trajectory],
                current_time
            )
            dt = 1.0 / self.control_freq
            position_command = Jointpos()
            position_command.dof = 7
            position_command.expand = 0.0
            position_command.follow = False
            position_command.joint = [
                self.current_joint[i] + current_input[i] * dt 
                for i in range(len(self.current_joint))
            ]
            self.position_command_pub.publish(position_command)
    
    def mpc_reset_using_current_state(self):
        while self.arm_pose is None:
            self.get_logger().info('Waiting for arm pose to reset mpc')
            rclpy.spin_once(self)
        self.get_logger().info('Arm pose is ready, reset mpc')
        cli = self.create_client(Reset, f'{arm_name}_mpc_reset')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'Service {arm_name}_mpc_reset not available, waiting again...')

        req = Reset.Request()
        req.reset = True
        mpcstate = MpcState(value=self.arm_pose)
        mpcinput = MpcInput(value=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target_traj = MpcTargetTrajectories(time_trajectory=self.get_elapsed_time(), state_trajectory=[mpcstate], input_trajectory = [mpcinput])
        req.target_trajectories = target_traj
        try:
            future = cli.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response.done:
                self.get_logger().info(f'MPC Reset successful for {arm_name}')
            else:
                self.get_logger().error(f'MPC Reset failed for {arm_name}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
        finally:
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    arm_name = 'mobile_manipulator'
    
    node = RealMpcInterface(arm_name)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()