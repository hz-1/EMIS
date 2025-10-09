import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label, center_of_mass, find_objects

def compute_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum
 
def compute_iou(mask_gt, mask_pred):
  volume_intersect = (mask_gt & mask_pred).sum()
  volume_union = (mask_gt | mask_pred).sum()
  if volume_union == 0:
    return np.NaN
  return volume_intersect / volume_union

def get_max_dist_points(mask):
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    total, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    print(total)
    max_dist_points = []
    for i in range(1, len(centroids)): 
        component_mask = (labels == i)
        component_dist_transform = dist_transform[component_mask]
        max_dist = np.max(component_dist_transform)
        max_dist_idx = np.where(component_dist_transform == max_dist)
        point = (max_dist_idx[1][0], max_dist_idx[0][0])
        max_dist_points.append(point)
    return max_dist_points

def get_centroids_and_bounding_boxes(mask):
    labeled_mask, num_regions = label(mask)
    centroids = []
    bounding_boxes = []
    for i in range(1, num_regions + 1):
        region_mask = (labeled_mask == i)
        centroid = center_of_mass(region_mask)
        centroids.append([centroid[1], centroid[0]]) 
        slices = find_objects(labeled_mask == i)[0]
        y1, x1 = slices[0].start, slices[1].start
        y2, x2 = slices[0].stop - 1, slices[1].stop - 1  
        bounding_boxes.append([x1, y1, x2, y2])
    return centroids, bounding_boxes