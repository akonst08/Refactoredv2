import numpy as np
import cv2
from timing import timeit

 
# Define background class colors from CityScapes palette (roads, buildings, sky, etc.)
BACKGROUND_COLORS = np.array([
    [0, 0, 0], [232, 35, 244], [70, 70, 70], [156, 102, 102],
    [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220],
    [35, 142, 107], [152, 251, 152], [180, 130, 70], [160, 190, 110],
    [50, 120, 170], [80, 90, 55], [150, 60, 45], [50, 234, 157],
    [81, 0, 81], [100, 100, 150], [140, 150, 230], [180, 165, 180],
    [70, 130, 180]
], dtype=np.uint8)


KERNEL_LARGE = np.ones((5, 5), np.uint8)
KERNEL_SMALL = np.ones((3, 3), np.uint8)

# Build camera intrinsic matrix (K) for projecting 3D points to 2D image coordinates
def build_projection_matrix(w, h, fov):
    # Calculate focal length from field of view
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0  # Principal point x (image center)
    K[1, 2] = h / 2.0  # Principal point y (image center)
    return K

# Project a 3D location to 2D image coordinates using camera matrix
def get_image_point(loc, K, w2c):
    # Convert location to homogeneous coordinates
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    # Transform from world to camera coordinates
    pc = w2c.dot(point)
    # CARLA coordinate system: convert to camera frame (right, down, forward)
    pc = np.array([pc[1], -pc[2], pc[0]])
    # Project to image plane using intrinsic matrix
    pi = K.dot(pc)
    # Normalize by depth (z-coordinate)
    if pi[2] != 0:
        pi[0] /= pi[2]
        pi[1] /= pi[2]
    return pi[0:2]

# Compute finite 2D bounding box from projected vertices, clipped to image bounds
@timeit("finite_bbox")
def finite_bbox(verts, W, H, min_size=1):
    xs, ys = zip(*verts)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # Check for invalid (infinite/NaN) coordinates
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    
    # Get bounding box coordinates
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    
    # For very small objects, ensure minimum size before clipping
    width = x_max - x_min
    height = y_max - y_min
    
    if width < min_size:
        center_x = (x_min + x_max) / 2
        x_min = center_x - min_size / 2
        x_max = center_x + min_size / 2
    
    if height < min_size:
        center_y = (y_min + y_max) / 2
        y_min = center_y - min_size / 2
        y_max = center_y + min_size / 2
    
    # Clip bounding box to image dimensions
    x_min = int(np.clip(x_min, 0, W - 1))
    x_max = int(np.clip(x_max, 0, W - 1))
    y_min = int(np.clip(y_min, 0, H - 1))
    y_max = int(np.clip(y_max, 0, H - 1))
    
    # Final check for degenerate boxes after clipping
    if x_min >= x_max or y_min >= y_max:
        return None
    
    return x_min, x_max, y_min, y_max

# Filter bounding boxes using segmentation image to remove false positives
# Rejects boxes where background pixels exceed the threshold ratio
@timeit("filter_boxes_segmentation")
def filter_boxes_segmentation(boxes, seg_img, bg_thr=0.60):
    if not boxes:
        return []
    
    seg_bgr = seg_img[:, :, :3]
    
    # Pre-compute background mask for entire image once
    # Create a lookup by converting RGB to a single integer for fast comparison
    bg_set = set(tuple(color) for color in BACKGROUND_COLORS)
    
    keep = []
    
    for (xmin, ymin, xmax, ymax, cid) in boxes:
        # Ensure box is within image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(seg_bgr.shape[1] - 1, xmax)
        ymax = min(seg_bgr.shape[0] - 1, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue
        
        # Extract region of interest
        roi = seg_bgr[ymin:ymax, xmin:xmax]
        
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            continue

        # Reshape ROI to 2D array of pixels (N x 3)
        roi_reshaped = roi.reshape(-1, 3)
        
        # Vectorized comparison: check if each pixel matches any background color
        # Broadcasting: (N, 3) vs (21, 3) -> (N, 21, 3)
        matches = (roi_reshaped[:, None, :] == BACKGROUND_COLORS[None, :, :]).all(axis=2)
        bg_pixels = matches.any(axis=1).sum()
        
        bg_ratio = bg_pixels / total_pixels
        
        if bg_ratio < bg_thr:
            keep.append((xmin, ymin, xmax, ymax, cid))

    return keep