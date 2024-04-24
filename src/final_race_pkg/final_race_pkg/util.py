#!/usr/bin/env python3

import math
import numpy as np
from numba import jit

########################################################## Helper Functions ########################################################
    
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