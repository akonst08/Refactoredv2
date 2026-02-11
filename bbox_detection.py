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
def filter_boxes_segmentation(boxes, seg_img, bg_thr=0.65):
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

def extract_static_boxes(classid_mask, class_id, min_area=80):
    """Find connected components of `class_id` in `classid_mask` and return boxes (x1,y1,x2,y2).

    Args:
        classid_mask (np.ndarray): HxW array of class IDs.
        class_id (int): class id to extract.
        min_area (int): minimum connected component area in pixels.

    Returns:
        List[tuple]: list of (x, y, x+w, y+h) boxes.
    """
    binary = (classid_mask == class_id).astype(np.uint8) # create binary mask, because components work only with binary.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    image_h, image_w = classid_mask.shape
    boxes = []
    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        # Clip coordinates to image bounds
        xmin = max(0, int(x))
        ymin = max(0, int(y))
        xmax = min(image_w - 1, int(x + w))
        ymax = min(image_h - 1, int(y + h))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two axis-aligned boxes.

    Boxes format: (x1, y1, x2, y2) where x1 < x2 and y1 < y2.
    
    IoU = intersection_area / union_area
    - 0.0 means no overlap
    - 1.0 means perfect overlap
    
    Example:
        boxA = (0, 0, 10, 10)  # 100 pixels
        boxB = (5, 5, 15, 15)  # 100 pixels
        # Intersection: (5,5) to (10,10) = 25 pixels
        # Union: 100 + 100 - 25 = 175 pixels
        # IoU: 25/175 = 0.143
    
    Returns:
        float: IoU score in [0, 1].
    """
    # Find intersection rectangle's top-left corner (maximum of both boxes' top-left)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    
    # Find intersection rectangle's bottom-right corner (minimum of both boxes' bottom-right)
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area (0 if boxes don't overlap)
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    # Calculate area of each box
    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    
    # Union = sum of areas minus intersection (to avoid counting overlap twice)
    union = areaA + areaB - inter

    # Return IoU, handling division by zero
    return inter / union if union > 0 else 0.0