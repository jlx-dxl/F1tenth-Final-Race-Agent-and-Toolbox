#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
# from tf_transformations import euler_from_quaternion
import math

import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Header
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import matplotlib.pyplot as plt

from os.path import join

# get the file path for this package
csv_loc = '/home/lucien/ESE6150/final_race/curve_best_sim.csv'

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

        # Params
        # self.declare_parameter('if_real', False)
        self.declare_parameter('lookahead_distance', 2.1)
        self.declare_parameter('lookahead_points', 12)      # to calculate yaw diff
        self.declare_parameter('lookbehind_points', 2)      # to eliminate the influence of latency
        self.declare_parameter('L_slope_atten', 0.5)        # attenuate lookahead distance with large yaw, (larger: smaller L when turning)

        self.declare_parameter('kp', 0.55)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.005)
        self.declare_parameter("max_control", MAX_STEER)
        self.declare_parameter("steer_alpha", 1.0)

        # PID Control Params
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer = 0.0
        
        self.curr_pos = np.array([0.0, 0.0])
        self.curr_yaw = 0.0
        
        #self.flag = self.get_parameter('lookahead_distance').get_parameter_value().bool_value
        #print(self.flag)
        self.flag = False

        # TODO: Get target x and y from pre-calculated waypoints
        waypoints = np.loadtxt(csv_loc, delimiter=',',skiprows=1)
        self.x_list = waypoints[:, 0]
        self.y_list = waypoints[:, 1]
        self.v_list = waypoints[:, 2]
        self.xyv_list = waypoints[:, 0:2]   # (x,y,v)
        self.yaw_list = waypoints[:, 3]
        self.v_max = np.max(self.v_list)
        self.v_min = np.min(self.v_list)
        
        # initialize grid map
        # 初始化代码
        self.lb = (-6.0, -3.5)  # 左下角的物理坐标
        self.rt = (12.0, 12.0)  # 右上角的物理坐标
        self.resolution = 0.1
        width = abs(self.lb[0] - self.rt[0]) / self.resolution
        height = abs(self.lb[1] - self.rt[1]) / self.resolution
        self.grid_map = np.zeros((int(height), int(width)))
        self.grid_nx = int(width)
        self.grid_ny = int(height)

        # 物理坐标阈值
        x_low = -4
        y_low = -1.5
        x_high = 10
        y_high = 10

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
        self.mark_rectangle_on_grid(self.lb, self.rt, self.resolution, (2.7, 4.0), (10.0, 10.0))
        self.mark_rectangle_on_grid(self.lb, self.rt, self.resolution, (-1.7, 0.6), (-1.0, 6.5))
        self.mark_rectangle_on_grid(self.lb, self.rt, self.resolution, (-1.7, 0.6), (5.5, 1.4))

        # # 使用matplotlib可视化地图
        # plt.figure(figsize=(10, 10))
        # plt.imshow(self.grid_map, cmap='Greys', origin='lower', extent=[lb[0], rt[0], lb[1], rt[1]])
        # plt.colorbar(label='Occupancy')
        # plt.title('Grid Map Visualization')
        # plt.xlabel('X coordinate (meters)')
        # plt.ylabel('Y coordinate (meters)')
        # plt.grid(False)
        # plt.show()

        # Topics & Subs, Pubs
        
        if self.flag == True:  
            odom_topic = '/pf/viz/inferred_pose'
            self.odom_sub_ = self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = '/ego_racecar/odom'
            self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        drive_topic = '/drive'
        waypoint_topic = '/ego_waypoint'
        waypoint_path_topic = '/ego_trajectory'
        occ_grid_topic = '/gridmap'

        self.traj_published = False
        self.traj_sub = self.create_subscription(Float32MultiArray, 'traj_inner', self.listener_callback_inner, 10)
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)
        self.waypoint_path_pub_ = self.create_publisher(Marker, waypoint_path_topic, 10)
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, occ_grid_topic, 10)
        self.scatter_pub = self.create_publisher(Marker, 'scatter', 10)
        
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
        
        moving_obstacle_list = []
        
        # 遍历所有激光点
        for i, distance in enumerate(data.ranges):
            if distance > data.range_min and distance < data.range_max:
                # 计算激光点的角度
                angle = data.angle_min + i * data.angle_increment + yaw
                
                # 计算激光点在地图中的坐标
                x = (x + distance * np.cos(angle)).astype(float)
                y = (y + distance * np.sin(angle)).astype(float)

                # 转换为栅格地图的索引
                grid_x = int((x - self.lb[0]) / self.resolution)
                grid_y = int((y - self.lb[1]) / self.resolution)

                # 更新栅格地图
                if 0 <= grid_x < self.grid_nx and 0 <= grid_y < self.grid_ny:
                    if self.grid_map[grid_y, grid_x] == 0:
                        moving_obstacle_list.append((x, y))
        print(len(moving_obstacle_list))
        self.visulaize_scatter(moving_obstacle_list)
        
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
        L = self.decrease_lookahead(L, yaw_diff)
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
        guess_point = self.proj_along(close_point, far_point, dist_btwn_ends / 2)
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
            guess_point = self.proj_along(close_point, far_point, direction * dist_btwn_ends / 2)
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

########################################################## Helper Functions ########################################################

    def mark_rectangle_on_grid(self, lb, rt, resolution, rect_lb, rect_rt):
        """
        Marks a rectangular area on a grid map as occupied, based on the bottom-left and top-right corners.

        :param grid_map: numpy array representing the grid map
        :param lb: tuple, the physical world coordinates of the bottom-left corner of the grid map
        :param rt: tuple, the physical world coordinates of the top-right corner of the grid map
        :param resolution: float, the resolution of the grid map in meters per grid cell
        :param rect_lb: tuple, the physical world coordinates of the bottom-left corner of the rectangle
        :param rect_rt: tuple, the physical world coordinates of the top-right corner of the rectangle
        """
        # Calculate the grid indices for the rectangle corners
        rect_lb_idx_x = int((rect_lb[0] - lb[0]) / resolution)
        rect_lb_idx_y = int((rect_lb[1] - lb[1]) / resolution)
        rect_rt_idx_x = int((rect_rt[0] - lb[0]) / resolution)
        rect_rt_idx_y = int((rect_rt[1] - lb[1]) / resolution)

        # Ensure indices are within the grid boundaries
        rect_lb_idx_x = max(0, min(rect_lb_idx_x, self.grid_map.shape[1] - 1))
        rect_lb_idx_y = max(0, min(rect_lb_idx_y, self.grid_map.shape[0] - 1))
        rect_rt_idx_x = max(0, min(rect_rt_idx_x, self.grid_map.shape[1] - 1))
        rect_rt_idx_y = max(0, min(rect_rt_idx_y, self.grid_map.shape[0] - 1))

        # Mark the rectangular area on the grid map
        # Ensure to capture all cells within the specified rectangle
        min_x = min(rect_lb_idx_x, rect_rt_idx_x)
        max_x = max(rect_lb_idx_x, rect_rt_idx_x)
        min_y = min(rect_lb_idx_y, rect_rt_idx_y)
        max_y = max(rect_lb_idx_y, rect_rt_idx_y)

        self.grid_map[min_y:max_y+1, min_x:max_x+1] = 1
    
    # travel a certain distance from one point in the direction of another
    def proj_along(self, start, target, dist):
        # find unit vector from start to target
        vect = target - start
        if (np.linalg.norm(vect) < 0.0001):
            return start
        unit = vect / np.linalg.norm(vect)
        # travel that distance
        new_point = dist * unit + start
        return new_point

    def decrease_lookahead(self, L, yaw_diff):
        slope = self.get_parameter('L_slope_atten').get_parameter_value().double_value
        # print(yaw_diff)
        if (yaw_diff > np.pi / 2):
            yaw_diff = np.pi / 2
        L = max(0.5, L * ((np.pi / 2 - yaw_diff * slope) / (np.pi / 2)))

        return L

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
        marker.ns = "points_and_lines"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # 设置Marker的尺寸
        marker.scale.x = 0.05  # 点的大小
        marker.scale.y = 0.05
        
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
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
