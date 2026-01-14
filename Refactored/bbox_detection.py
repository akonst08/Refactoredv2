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
def finite_bbox(verts, W, H):
    xs, ys = zip(*verts)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # Check for invalid (infinite/NaN) coordinates
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    # Clip bounding box to image dimensions
    # np.clip ensures values are within [0, W-1] and [0, H-1]
    x_min = int(np.clip(xs.min(), 0, W - 1))
    x_max = int(np.clip(xs.max(), 0, W - 1))
    y_min = int(np.clip(ys.min(), 0, H - 1))
    y_max = int(np.clip(ys.max(), 0, H - 1))
    # Reject degenerate boxes (zero width or height)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, x_max, y_min, y_max

# Filter bounding boxes using segmentation image to remove false positives
# Rejects boxes where background pixels exceed the threshold ratio
@timeit("filter_boxes_segmentation")
def filter_boxes_segmentation(boxes, seg_img, bg_thr=0.45):
    if not boxes:
        return []
    
    seg_bgr = seg_img[:, :, :3]
    keep = []
    
    # Filter boxes based on background ratio within each bounding box

    for (xmin, ymin, xmax, ymax, cid) in boxes:
        # Ensure box is within image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(seg_bgr.shape[1] - 1, xmax)
        ymax = min(seg_bgr.shape[0] - 1, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue
        
        # Extract region of interest and compute background ratio
        roi = seg_bgr[ymin:ymax, xmin:xmax]
        
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            continue

        bg_pixels = 0
        for color in BACKGROUND_COLORS:
            bg_pixels += np.all(roi == color, axis=-1).sum()
        
        bg_ratio = bg_pixels / total_pixels
        # Keep box if background ratio is below threshold (mostly foreground)
        if bg_ratio < bg_thr:
            keep.append((xmin, ymin, xmax, ymax, cid))

    return keep

# Detect static vehicles from segmentation that aren't already tracked dynamically
# Uses color-based segmentation and morphological operations to find parked vehicles
@timeit("detect_static_vehicles")
def detect_static_vehicles(seg_bgr, boxes_xyxy_cls, SEG_COLORS):
    """
    Detect static vehicles from the semantic segmentation image that are not already
    tracked dynamically.

    Parameters
    ----------
    seg_bgr : np.ndarray
        Semantic segmentation image in BGR format (height × width × 3).
    boxes_xyxy_cls : list of tuples
        List of dynamic bounding boxes as (x1, y1, x2, y2, class_id).
    SEG_COLORS : dict
        Mapping from class IDs to tuples/lists containing segmentation BGR colors.

    Returns
    -------
    list of tuples
        Detected static bounding boxes as (xmin, ymin, xmax, ymax, class_id).
    """
    static_boxes = []
    H, W = seg_bgr.shape[:2]

    # Class IDs to separate large vs small vehicles
    large_vehicle_ids = [14, 15, 16, 17]  # car, truck, bus, train
    small_vehicle_ids = [13, 18, 19]      # rider, motorcycle, bicycle

    # Initialise masks and ID map
    mask_large = np.zeros((H, W), dtype=np.uint8)
    mask_small = np.zeros((H, W), dtype=np.uint8)
    id_mask_map = np.zeros((H, W), dtype=np.uint8)

    # Build mask for large vehicles using cv2.inRange(); update ID map
    for vehicle_id in large_vehicle_ids:
        color = SEG_COLORS[vehicle_id][0]
        lower = np.array(color, dtype=np.uint8)
        upper = lower  # exact match
        matches = cv2.inRange(seg_bgr, lower, upper)
        mask_large[matches > 0] = 255
        id_mask_map[matches > 0] = vehicle_id

    # Build mask for small vehicles using cv2.inRange(); update ID map
    for vehicle_id in small_vehicle_ids:
        color = SEG_COLORS[vehicle_id][0]
        lower = np.array(color, dtype=np.uint8)
        upper = lower
        matches = cv2.inRange(seg_bgr, lower, upper)
        mask_small[matches > 0] = 255
        id_mask_map[matches > 0] = vehicle_id

    # Create dynamic exclusion mask by filling bounding rectangles
    dynamic_mask = np.zeros((H, W), dtype=np.uint8)
    if boxes_xyxy_cls:
        vehicle_id_set = set(large_vehicle_ids + small_vehicle_ids)
        for (x1, y1, x2, y2, cid) in boxes_xyxy_cls:
            if cid in vehicle_id_set:
                x1_clamped = max(0, x1)
                y1_clamped = max(0, y1)
                x2_clamped = min(W - 1, x2)
                y2_clamped = min(H - 1, y2)
                dynamic_mask[y1_clamped:y2_clamped + 1,
                             x1_clamped:x2_clamped + 1] = 255

    # Remove dynamic regions from static masks
    static_mask_large = cv2.bitwise_and(mask_large, cv2.bitwise_not(dynamic_mask))
    static_mask_small = cv2.bitwise_and(mask_small, cv2.bitwise_not(dynamic_mask))

    # Morphological cleanup: opening + dilation for large vehicles
    static_mask_large = cv2.morphologyEx(static_mask_large, cv2.MORPH_OPEN,
                                         KERNEL_LARGE, iterations=1)
    static_mask_large = cv2.dilate(static_mask_large, KERNEL_LARGE, iterations=1)

    # Morphological cleanup: closing for small vehicles
    static_mask_small = cv2.morphologyEx(static_mask_small, cv2.MORPH_CLOSE,
                                         KERNEL_SMALL, iterations=2)

    # Find contours and create bounding boxes for large vehicles
    contours_large, _ = cv2.findContours(static_mask_large, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_large:
        area = cv2.contourArea(c)
        if area > 300:  # minimum area threshold for large vehicles
            x, y, w, h = cv2.boundingRect(c)
            # majority-vote the class ID within the region
            roi = id_mask_map[y:y + h, x:x + w]
            pixels = roi[roi > 0]
            if pixels.size > 0:
                cid = int(np.bincount(pixels).argmax())
                static_boxes.append((x, y, x + w, y + h, cid))

    # Find contours and create bounding boxes for small vehicles
    contours_small, _ = cv2.findContours(static_mask_small, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_small:
        area = cv2.contourArea(c)
        if 30 < area < 2500:  # area range for small vehicles
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 5:  # reject overly elongated shapes
                roi = id_mask_map[y:y + h, x:x + w]
                pixels = roi[roi > 0]
                if pixels.size > 0:
                    cid = int(np.bincount(pixels).argmax())
                    static_boxes.append((x, y, x + w, y + h, cid))

    return static_boxes

