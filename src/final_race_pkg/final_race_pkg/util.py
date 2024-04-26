#!/usr/bin/env python3

import math
import numpy as np
from numba import jit

########################################################## Helper Functions ########################################################

@jit(nopython=True)
def euclidean_norm(data):
    return np.sqrt(np.sum(data**2, axis=1))

@jit(nopython=True)
# travel a certain distance from one point in the direction of another
def proj_along(start, target, dist):
    # find unit vector from start to target
    vect = target - start
    if (np.linalg.norm(vect) < 0.0001):
        return start
    unit = vect / np.linalg.norm(vect)
    # travel that distance
    new_point = dist * unit + start
    return new_point

@jit(nopython=True)
def decrease_lookahead(L, yaw_diff, slope):
    # print(yaw_diff)
    if (yaw_diff > np.pi / 2):
        yaw_diff = np.pi / 2
    L = max(0.5, L * ((np.pi / 2 - yaw_diff * slope) / (np.pi / 2)))
    return L

@jit(nopython=True)
def get_lookahead(curr_pos, curr_yaw, xyv_list, yaw_list, v_list, L, lookahead_points, lookbehind_points, slope):
    # find index of closest point
    distances = euclidean_norm(xyv_list - curr_pos)
    min_idx = np.argmin(distances)
    
    future_yaw_target = yaw_list[(min_idx + lookahead_points) % yaw_list.shape[0]]
    past_yaw_target = yaw_list[(min_idx - lookbehind_points) % yaw_list.shape[0]]
    yaw_diff = abs(past_yaw_target - future_yaw_target)
    if yaw_diff > np.pi:
        yaw_diff = yaw_diff - 2 * np.pi
    if yaw_diff < -np.pi:
        yaw_diff = yaw_diff + 2 * np.pi
    yaw_diff = abs(yaw_diff)
    L = decrease_lookahead(L, yaw_diff, slope)
    gamma = 2 / L ** 2  # curvature of arc

    # TODO: find the current waypoint to track using methods mentioned in lecture
    curr_target_idx = min_idx
    next_idx = min_idx + 1
    next_dist = distances[next_idx % len(distances)]
    while (next_dist <= L):
        min_idx = next_idx
        next_idx = next_idx + 1
        next_dist = distances[next_idx % distances.shape[0]]  # avoid hitting the array's end
    # once points are found, find linear interpolation of them through binary search 
    # until it's at the right distance L
    close_point = xyv_list[min_idx % distances.shape[0], :]
    far_point = xyv_list[next_idx % distances.shape[0], :]
    dist_btwn_ends = np.linalg.norm(far_point - close_point)
    guess_point = proj_along(close_point, far_point, dist_btwn_ends / 2)
    dist_to_guess = np.linalg.norm(curr_pos - guess_point)
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
        dist_to_guess = np.linalg.norm(curr_pos - guess_point)
        num_iters = num_iters + 1
    target_point = guess_point
    # print(num_iters)

    # TODO: transform goal point to vehicle frame of reference
    R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                    [-np.sin(curr_yaw), np.cos(curr_yaw)]])
    _, target_y = R @ np.array([target_point[0] - curr_pos[0],
                                        target_point[1] - curr_pos[1]])
    target_v = v_list[curr_target_idx % len(v_list)]
    # compute error using the lookahead distance
    error = gamma * target_y
    
    return error, target_v, target_point, curr_target_idx

def k_means(points: np.ndarray, k: int, max_iters=100):
    # 随机初始化聚类中心
    centers = points[np.random.choice(len(points), k, replace=False)]
    for _ in range(max_iters):
        # 分配点到最近的中心
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = np.linalg.norm(point - centers, axis=1)
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # 更新中心点
        new_centers = [np.mean(cluster, axis=0) if cluster else centers[i] for i, cluster in enumerate(clusters)]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers, clusters

def find_densest_cluster_center(points, k):
    _, clusters = k_means(points, k)
    # 找到最大的簇
    largest_cluster_index = np.argmax([len(cluster) for cluster in clusters])
    # 计算这个簇的中心
    densest_center = np.mean(clusters[largest_cluster_index], axis=0)
    return densest_center

@jit(nopython=True)
def get_move_obstacle_list(ranges, angle_increment, angle_min, x, y, yaw, lb, rt, res, grid_map):

    moving_obstacle_list = []
    
    for i in range(len(ranges)):
        angle = angle_min + i * angle_increment
        # 计算扫描点的局部坐标（相对于机器人）
        local_x = ranges[i] * math.cos(angle)
        local_y = ranges[i] * math.sin(angle)

        # 将局部坐标转换为全局坐标
        # 使用旋转矩阵进行坐标变换
        global_x = x + (local_x * math.cos(yaw) - local_y * math.sin(yaw))
        global_y = y + (local_x * math.sin(yaw) + local_y * math.cos(yaw))
        
        # 转换为栅格地图的索引
        grid_x = int((global_x - lb[0]) / res)
        grid_y = int((global_y - lb[1]) / res)
        # print(grid_x, grid_y)
        
        grid_nx = int(abs(lb[0] - rt[0]) / res)
        grid_ny = int(abs(lb[1] - rt[1]) / res)
        
        if 0 <= grid_x < grid_nx and 0 <= grid_y < grid_ny:
            if grid_map[grid_y, grid_x] == 0:
                moving_obstacle_list.append((global_x, global_y))
                
    return moving_obstacle_list

@jit(nopython=True)
def mark_rectangle_on_grid(grid_map, lb, rt, resolution, rect_lb, rect_rt):
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
    rect_lb_idx_x = max(0, min(rect_lb_idx_x, grid_map.shape[1] - 1))
    rect_lb_idx_y = max(0, min(rect_lb_idx_y, grid_map.shape[0] - 1))
    rect_rt_idx_x = max(0, min(rect_rt_idx_x, grid_map.shape[1] - 1))
    rect_rt_idx_y = max(0, min(rect_rt_idx_y, grid_map.shape[0] - 1))

    # Mark the rectangular area on the grid map
    # Ensure to capture all cells within the specified rectangle
    min_x = min(rect_lb_idx_x, rect_rt_idx_x)
    max_x = max(rect_lb_idx_x, rect_rt_idx_x)
    min_y = min(rect_lb_idx_y, rect_rt_idx_y)
    max_y = max(rect_lb_idx_y, rect_rt_idx_y)

    grid_map[min_y:max_y+1, min_x:max_x+1] = 1
    
    return grid_map