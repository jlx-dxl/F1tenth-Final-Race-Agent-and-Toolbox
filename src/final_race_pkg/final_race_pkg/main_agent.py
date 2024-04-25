#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# from tf_transformations import euler_from_quaternion
import math

import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Header
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import matplotlib.pyplot as plt

from os.path import join
from .util import *

# get the file path for this package
# csv_loc = '/home/nvidia/f1tenth_ws/src/F1tenth-Final-Race-Agent-and-Toolbox/curve_best_sim.csv'
csv_loc = '/home/lucien/ESE6150/final_race/curve1.csv'

#  Constants from xacro
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        ####################################### Params ##########################################
        # self.declare_parameter('if_real', False)
        self.declare_parameter('lookahead_distance', 2.1)
        self.declare_parameter('lookahead_points', 18)      # to calculate yaw diff
        self.declare_parameter('lookbehind_points', 2)      # to eliminate the influence of latency
        self.declare_parameter('L_slope_atten', 0.4)        # attenuate lookahead distance with large yaw, (larger: smaller L when turning)
        self.declare_parameter('n_cluster', 5)
        self.declare_parameter('kp', 0.55)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.005)
        self.declare_parameter("max_control", MAX_STEER)
        self.declare_parameter("steer_alpha", 1.0)
        
        print("finished declaring parameters")

        # PID Control Params
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0
        
        self.curr_pos = np.array([0.0, 0.0])
        self.curr_yaw = 0.0
        
        #self.flag = self.get_parameter('lookahead_distance').get_parameter_value().bool_value
        self.flag = False
        print("if real world test? ", self.flag)

        # TODO: Get target x and y from pre-calculated waypoints
        waypoints = np.loadtxt(csv_loc, delimiter=',',skiprows=1)
        self.x_list = waypoints[:, 0]
        self.y_list = waypoints[:, 1]
        self.v_list = waypoints[:, 2]
        self.xyv_list = waypoints[:, 0:2]   # (x,y,v)
        self.yaw_list = waypoints[:, 3]
        self.v_max = np.max(self.v_list)
        self.v_min = np.min(self.v_list)
        
        print("finished loading waypoints")
        
        #################################### initialize grid map ################################
        # 这个地图需要被精细建模，后面用于滤掉静态障碍物
        # 初始化代码
        self.lb = (7.0, 4.5)  # 左下角的物理坐标
        self.rt = (23.6, 24.0)  # 右上角的物理坐标
        self.resolution = 0.1
        width = abs(self.lb[0] - self.rt[0]) / self.resolution
        height = abs(self.lb[1] - self.rt[1]) / self.resolution
        self.grid_map = np.zeros((int(height), int(width)))
        self.grid_nx = int(width)
        self.grid_ny = int(height)

        # 物理坐标阈值
        x_low = 10.0
        y_low = 7.5
        x_high = 20.8
        y_high = 21.0

        # 转换为栅格索引
        x_low_idx = int((x_low - self.lb[0]) / self.resolution)
        y_low_idx = int((y_low - self.lb[1]) / self.resolution)
        x_high_idx = int((x_high - self.lb[0]) / self.resolution)
        y_high_idx = int((y_high - self.lb[1]) / self.resolution)

        # 确保索引在地图范围内
        x_low_idx = max(x_low_idx, 0)
        y_low_idx = max(y_low_idx, 0)
        x_high_idx = min(x_high_idx, int(width))
        y_high_idx = min(y_high_idx, int(height))

        # 设置条件
        self.grid_map[:, :x_low_idx] = 1  # x < -4
        self.grid_map[:, x_high_idx:] = 1  # x > 10
        self.grid_map[:y_low_idx, :] = 1  # y < -1.5
        self.grid_map[y_high_idx:, :] = 1  # y > 10
        
        # 添加矩形障碍物
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (9.3, 13.8), (15.5, 21.0))   # 左上角的
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (12.5, 9.5), (18.8, 10.8))   # L弯底边
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (17.6, 9.7), (18.7, 17.1))   # L弯竖边
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (19.2, 7.1), (21.0, 8.0))    # 右下角
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (9.8, 7.3), (11.1, 8.5))   # 左下角
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (15.0, 19.0), (21.0, 21.0))   # 顶部
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (9.8, 7.3), (20.0, 7.7))   # 底部
        mark_rectangle_on_grid(self.grid_map, self.lb, self.rt, self.resolution, (20.8, 7.5), (21.0, 21.0))   # 底部


        # # 使用matplotlib可视化地图
        # plt.figure(figsize=(10, 10))
        # plt.imshow(self.grid_map, cmap='Greys', origin='lower', extent=[lb[0], rt[0], lb[1], rt[1]])
        # plt.colorbar(label='Occupancy')
        # plt.title('Grid Map Visualization')
        # plt.xlabel('X coordinate (meters)')
        # plt.ylabel('Y coordinate (meters)')
        # plt.grid(False)
        # plt.show()
        
        print("finished initializing grid map")

        #################################### Topics & Subs, Pubs ######################################
        
        if self.flag == True:  
            odom_topic = '/pf/viz/inferred_pose'
            self.odom_sub_ = self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)
            print(odom_topic + " initialized")
        else:
            odom_topic = '/ego_racecar/odom'
            self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
            print(odom_topic + " initialized")
            
        drive_topic = '/drive'
        waypoint_topic = '/waypoint'
        waypoint_path_topic = '/waypoint_path'
        occ_grid_topic = '/gridmap'

        self.traj_published = False
        self.traj_sub = self.create_subscription(Float32MultiArray, '/traj_inner', self.listener_callback_inner, 10)
        print("traj_sub initialized, topic: " + '/traj_inner')
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        print("scan_sub_ initialized, topic: " + '/scan')
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        print("drive_pub_ initialized, topic: " + drive_topic)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        print("waypoint_pub_ initialized, topic: " + waypoint_topic)
        self.waypoint_path_pub_ = self.create_publisher(Marker, waypoint_path_topic, 10)
        print("waypoint_path_pub_ initialized, topic: " + waypoint_path_topic)
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, occ_grid_topic, 10)
        print("occ_grid_pub initialized, topic: " + occ_grid_topic)
        self.scatter_pub = self.create_publisher(Marker, '/scatter', 10)
        print("scatter_pub initialized, topic: " + '/scatter')
        self.oppo_curr_pub = self.create_publisher(Marker, '/curr_opp', 10)
        print("oppo_curr_pub initialized, topic: " + '/curr_opp')

        
###################################################### Callbacks ############################################################
    
    def listener_callback_inner(self, msg):
        trajectory_array = np.array(msg.data, dtype=np.float32).reshape((200, 4)).astype(float)
        self.x_list = trajectory_array[:, 0]
        self.y_list = trajectory_array[:, 1]
        self.v_list = trajectory_array[:, 2]
        self.xyv_list = trajectory_array[:, 0:2]   # (x,y,v)
        self.yaw_list = trajectory_array[:, 3]
        self.v_max = np.max(self.v_list)
        self.v_min = np.min(self.v_list)
    
    def laser_scan_callback(self, data):
        x = self.curr_pos[0]
        y = self.curr_pos[1]
        yaw = self.curr_yaw
        
        ranges = data.ranges
        angle_increment = data.angle_increment
        angle_min = data.angle_min
        
        lb = self.lb
        rt = self.rt
        res = self.resolution
        grid_map = self.grid_map
        
        moving_obstacle_list = get_move_obstacle_list(ranges, angle_increment, angle_min, x, y, yaw, lb, rt, res, grid_map)
        
        self.visulaize_scatter(moving_obstacle_list)
        
        k = self.get_parameter('n_cluster').get_parameter_value().integer_value
        
        if len(moving_obstacle_list) > k:
            
            moving_obstacle_list = np.array(moving_obstacle_list, dtype=np.float64)
            curr_opp_position = find_densest_cluster_center(moving_obstacle_list, k)
            print(curr_opp_position)
            
            self.visulaize_curr_opp(curr_opp_position)
        
    def pose_callback(self, pose_msg):
        if self.flag == True:  
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_quat = pose_msg.pose.pose.orientation
        self.curr_pos = np.array([curr_x, curr_y])
        self.curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                              1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        # find the closest current point
        # find index of closest point
        distances = np.linalg.norm(self.xyv_list - self.curr_pos, axis=1)
        min_idx = np.argmin(distances)
        closest_point = self.xyv_list[min_idx, :]

        # change L based on another lookahead distance for yaw difference!
        L = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        lookahead_points = self.get_parameter('lookahead_points').get_parameter_value().integer_value
        lookbehind_points = self.get_parameter('lookbehind_points').get_parameter_value().integer_value
        
        future_yaw_target = self.yaw_list[(min_idx + lookahead_points) % self.yaw_list.shape[0]]
        past_yaw_target = self.yaw_list[(min_idx - lookbehind_points) % self.yaw_list.shape[0]]
        yaw_diff = abs(past_yaw_target - future_yaw_target)
        if yaw_diff > np.pi:
            yaw_diff = yaw_diff - 2 * np.pi
        if yaw_diff < -np.pi:
            yaw_diff = yaw_diff + 2 * np.pi
        yaw_diff = abs(yaw_diff)
        slope = self.get_parameter('L_slope_atten').get_parameter_value().double_value
        L = decrease_lookahead(L, yaw_diff, slope)
        gamma = 2 / L ** 2  # curvature of arc

        # TODO: find the current waypoint to track using methods mentioned in lecture
        self.curr_target_idx = min_idx
        next_idx = min_idx + 1
        next_dist = distances[next_idx % len(distances)]
        while (next_dist <= L):
            min_idx = next_idx
            next_idx = next_idx + 1
            next_dist = distances[next_idx % distances.shape[0]]  # avoid hitting the array's end
        # once points are found, find linear interpolation of them through binary search 
        # until it's at the right distance L
        close_point = self.xyv_list[min_idx % distances.shape[0], :]
        far_point = self.xyv_list[next_idx % distances.shape[0], :]
        dist_btwn_ends = np.linalg.norm(far_point - close_point)
        guess_point = proj_along(close_point, far_point, dist_btwn_ends / 2)
        dist_to_guess = np.linalg.norm(self.curr_pos - guess_point)
        num_iters = 0
        while (abs(dist_to_guess - L) > 0.01):
            if (dist_to_guess > L):  # too far away, set the guess point as the far point
                far_point = guess_point
                dist_btwn_ends = np.linalg.norm(far_point - close_point)
                direction = -1  # go backward
            else:  # too close, set the guess point as the close point
                close_point = guess_point
                dist_btwn_ends = np.linalg.norm(far_point - close_point)
                direction = 1  # go forward
            # recalculate
            guess_point = proj_along(close_point, far_point, direction * dist_btwn_ends / 2)
            dist_to_guess = np.linalg.norm(self.curr_pos - guess_point)
            num_iters = num_iters + 1
        self.target_point = guess_point
        # print(num_iters)

        # TODO: transform goal point to vehicle frame of reference
        R = np.array([[np.cos(self.curr_yaw), np.sin(self.curr_yaw)],
                      [-np.sin(self.curr_yaw), np.cos(self.curr_yaw)]])
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])
        target_v = self.v_list[self.curr_target_idx % len(self.v_list)]
        # compute error using the lookahead distance
        error = gamma * target_y

        # TODO: publish drive message, don't forget to limit the steering angle.
        message = AckermannDriveStamped()
        message.drive.speed = target_v
        message.drive.steering_angle = self.get_steer(error)
        # self.get_logger().info('speed: %f, steer: %f' % (target_v, self.get_steer(error)))
        self.drive_pub_.publish(message)

        # remember to visualize the waypoints
        self.visualize_waypoints()
        
        # visualize the occupancy grid
        self.visulize_occ_grid()

################################################### Visualization ################################################33

    def visualize_waypoints(self):
        # TODO: publish the waypoints properly
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = 0
        marker.ns = 'pursuit_waypoint' + str(0)
        marker.type = 4
        marker.action = 0
        marker.points = []
        marker.colors = []
        length = self.x_list.shape[0]
        for i in range(length + 1):
            this_point = Point()
            this_point.x = self.x_list[i % length]
            this_point.y = self.y_list[i % length]
            marker.points.append(this_point)

            this_color = ColorRGBA()
            normalized_target_speed = (self.v_list[i % length] - self.v_min) / (self.v_max - self.v_min)
            this_color.a = 1.0
            this_color.r = (1 - normalized_target_speed)
            this_color.g = normalized_target_speed
            marker.colors.append(this_color)

        this_scale = 0.1
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.waypoint_path_pub_.publish(marker)

        # also publish the target but larger
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.id = 0
        marker.ns = 'pursuit_waypoint_target'
        marker.type = 1
        marker.action = 0
        marker.pose.position.x = self.target_point[0]
        marker.pose.position.y = self.target_point[1]

        normalized_target_speed = self.v_list[self.curr_target_idx] / self.v_max
        marker.color.a = 1.0
        marker.color.r = (1 - normalized_target_speed)
        marker.color.g = normalized_target_speed

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale

        marker.pose.orientation.w = 1.0

        self.waypoint_pub_.publish(marker)
        
    def visulize_occ_grid(self):
        # 设置地图信息
        resolution = self.resolution  # 栅格分辨率
        width = self.grid_nx       # 栅格宽度
        height = self.grid_ny      # 栅格高度
        
        # 创建OccupancyGrid消息
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.frame_id = "map"
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = self.lb[0]
        grid.info.origin.position.y = self.lb[1]
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0  # 四元数，无旋转
        
        # 栅格地图数据，将NumPy数组转换为列表
        grid.data = self.grid_map.astype(int).flatten().tolist()

        self.occ_grid_pub.publish(grid)
        
    def visulaize_scatter(self, scatter_points):
        marker = Marker()
        marker.header.frame_id = "/map"  # 设置合适的参考框架
        marker.ns = "scatters"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # 设置Marker的尺寸
        marker.scale.x = 0.1  # 点的大小
        marker.scale.y = 0.1
        
        # 设置颜色
        marker.color.a = 1.0  # 不透明度
        marker.color.r = 1.0  # 红色
        marker.color.g = 0.0  # 绿色
        marker.color.b = 0.0  # 蓝色
        
        # 填充点位置
        for (x, y) in scatter_points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0  # 如果是2D地图，z通常为0
            marker.points.append(p)
        
        # 发布Marker
        self.scatter_pub.publish(marker)
        
    def visulaize_curr_opp(self, point):
        marker = Marker()
        marker.header.frame_id = "/map"  # 设置合适的参考框架
        marker.ns = "oppenent_current_estimation"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # 设置Marker的尺寸
        marker.scale.x = 0.2  # 点的大小
        marker.scale.y = 0.2
        
        # 设置颜色
        marker.color.a = 1.0  # 不透明度
        marker.color.r = 1.0  # 红色
        marker.color.g = 0.0  # 绿色
        marker.color.b = 1.0  # 蓝色
        

        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = 0.0  # 如果是2D地图，z通常为0
        marker.points.append(p)
        
        # 发布Marker
        self.oppo_curr_pub.publish(marker)

    def get_steer(self, error):
        """ Get desired steering angle by PID
        """
        kp = self.get_parameter('kp').get_parameter_value().double_value
        ki = self.get_parameter('ki').get_parameter_value().double_value
        kd = self.get_parameter('kd').get_parameter_value().double_value
        max_control = self.get_parameter('max_control').get_parameter_value().double_value
        alpha = self.get_parameter('steer_alpha').get_parameter_value().double_value

        d_error = error - self.prev_error
        self.prev_error = error
        self.integral += error
        steer = kp * error + ki * self.integral + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        new_steer = alpha * new_steer + (1 - alpha) * self.prev_steer
        self.prev_steer = new_steer

        return new_steer


def main(args=None):
    
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuit()
    print("PurePursuit Initialized")
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
