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
def filter_boxes_segmentation(boxes, seg_img, bg_thr=0.65, camera_z = 30.0):
    if not boxes:
        return []
    
    seg_bgr = seg_img[:, :, :3]
    SMALL_CLASSES = {12, 13, 18, 19}  # pedestrian, rider, motorcycle, bicycle
    # Pre-compute background mask for entire image once
    # Create a lookup by converting RGB to a single integer for fast comparison
    
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

        if cid in SMALL_CLASSES:
            effective_bg_thr = bg_thr_for_class(cid, camera_z)
        else:
            effective_bg_thr = bg_thr

        # Reshape ROI to 2D array of pixels (N x 3)
        roi_reshaped = roi.reshape(-1, 3)
        
        # Vectorized comparison: check if each pixel matches any background color
        # Broadcasting: (N, 3) vs (21, 3) -> (N, 21, 3)
        matches = (roi_reshaped[:, None, :] == BACKGROUND_COLORS[None, :, :]).all(axis=2)
        bg_pixels = matches.any(axis=1).sum()
        
        bg_ratio = bg_pixels / total_pixels

        if bg_ratio < effective_bg_thr:
            keep.append((xmin, ymin, xmax, ymax, cid))

    return keep

@timeit("extract_static_boxes")
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
@timeit("iou")
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
@timeit("suppress_contained_boxes")
def suppress_contained_boxes(boxes, overlap_threshold=0.75):
    """
    Remove only obvious duplicate boxes:
    a smaller box must be heavily contained inside a larger one.
    This avoids deleting nearby real vehicles on curves.
    """
    if len(boxes) == 0:
        return boxes

    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    keep = []

    for box_a in boxes_sorted:
        x1a, y1a, x2a, y2a, cid_a = box_a
        area_a = max(1, (x2a - x1a) * (y2a - y1a))
        cx_a = (x1a + x2a) / 2.0
        cy_a = (y1a + y2a) / 2.0
        suppressed = False

        for box_b in keep:
            x1b, y1b, x2b, y2b, cid_b = box_b

            ix1 = max(x1a, x1b)
            iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b)
            iy2 = min(y2a, y2b)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_ratio = inter_area / area_a

            center_inside = (x1b <= cx_a <= x2b) and (y1b <= cy_a <= y2b)

            # suppress only if box_a is really just a duplicate fragment
            if overlap_ratio >= overlap_threshold and center_inside:
                suppressed = True
                break

        if not suppressed:
            keep.append(box_a)

    return keep
@timeit("nms_boxes")
def nms_boxes(boxes, iou_threshold=0.35):
    """
    IoU-based NMS across ALL boxes (dynamic + static combined).
    Keeps the larger box when two boxes of any class overlap above threshold.
    This catches duplicate detections from the 3D projection and segmentation
    pipelines firing on the same physical object.
    """
    if len(boxes) == 0:
        return boxes

    # Sort by box area descending — keep larger box when duplicates found
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    keep = []

    for box_a in boxes_sorted:
        x1a, y1a, x2a, y2a, cid_a = box_a
        suppressed = False

        for box_b in keep:
            x1b, y1b, x2b, y2b, cid_b = box_b
                # Fast reject for clearly separate boxes
            if x2a <= x1b or x2b <= x1a or y2a <= y1b or y2b <= y1a:
                continue
            # Compute IoU
            ix1 = max(x1a, x1b)
            iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b)
            iy2 = min(y2a, y2b)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            area_a = max(1, (x2a - x1a) * (y2a - y1a))
            area_b = max(1, (x2b - x1b) * (y2b - y1b))
            union = area_a + area_b - inter
            iou_score = inter / union if union > 0 else 0.0

            if iou_score > iou_threshold:
                suppressed = True
                break

        if not suppressed:
            keep.append(box_a)

    return keep

def bg_thr_for_class(cid, camera_z):
    """
    Compute background threshold dynamically based on class and camera height.
    Higher z = smaller objects = more background pixels in their bounding box
    needs a higher threshold to avoid rejecting valid detections.
    """
    # Base thresholds per class at z=30 (ground truth from your observations)
    BASE_THR = {
        12: 0.70,   # pedestrian
        13: 0.40,   # rider
        18: 0.40,   # motorcycle
        19: 0.40,   # bicycle
    }
    DEFAULT_BASE = 0.40  # cars, trucks, buses

    base = BASE_THR.get(cid, DEFAULT_BASE)

    # Linear scale: every +10 units of z adds ~0.009 per base unit
    # Derived from: z=30->0.70, z=40->0.80 means +0.10 per 10 units for pedestrian
    z_ref = 30.0
    scale = (camera_z - z_ref) / 5.0 * 0.010  # 0.010 per unit base per 10m height

    # Apply scale proportionally to base, cap at 0.95 to never block everything
    dynamic_thr = min(0.95, base + base * scale)
    return dynamic_thr


def _interval_overlap(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def _interval_gap(a1, a2, b1, b2):
    return max(0, max(a1, b1) - min(a2, b2))


def _box_fill_ratio(classid_mask, box, class_id):
    x1, y1, x2, y2 = box
    roi = classid_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi == class_id))

@timeit("merge_fragmented_static_boxes")
def merge_fragmented_static_boxes(classid_mask, boxes, class_id):
    """
    Merge same-class static boxes that were split by thin occluders
    like poles or trees.
    """
    if len(boxes) < 2:
        return boxes

    if class_id in (18, 19):
        max_gap = 6
        min_axis_overlap = 0.45
        min_fill_ratio = 0.12
        max_area_growth = 2.2
    else:
        max_gap = 12
        min_axis_overlap = 0.60
        min_fill_ratio = 0.18
        max_area_growth = 2.6

    merged_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    changed = True

    while changed:
        changed = False
        next_boxes = []
        used = [False] * len(merged_boxes)

        for i, box_a in enumerate(merged_boxes):
            if used[i]:
                continue

            current = box_a #x_min, y_min, x_max, y_max
            # ( x_max - x_min ) * (y_max - y_min) 
            area_a = max(1, (current[2] - current[0]) * (current[3] - current[1]))

            for j in range(i + 1, len(merged_boxes)):
                if used[j]:
                    continue

                box_b = merged_boxes[j]
                area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

                x_overlap = _interval_overlap(current[0], current[2], box_b[0], box_b[2])
                y_overlap = _interval_overlap(current[1], current[3], box_b[1], box_b[3])
                x_gap = _interval_gap(current[0], current[2], box_b[0], box_b[2])
                y_gap = _interval_gap(current[1], current[3], box_b[1], box_b[3])

                aligned_horizontally = (
                    x_gap <= max_gap and
                    y_overlap / max(1, min(current[3] - current[1], box_b[3] - box_b[1])) >= min_axis_overlap
                )
                aligned_vertically = (
                    y_gap <= max_gap and
                    x_overlap / max(1, min(current[2] - current[0], box_b[2] - box_b[0])) >= min_axis_overlap
                )

                if not (aligned_horizontally or aligned_vertically):
                    continue

                candidate = (
                    min(current[0], box_b[0]),
                    min(current[1], box_b[1]),
                    max(current[2], box_b[2]),
                    max(current[3], box_b[3]),
                )

                candidate_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
                fill_ratio = _box_fill_ratio(classid_mask, candidate, class_id)

                if fill_ratio < min_fill_ratio:
                    continue
                if candidate_area > (area_a + area_b) * max_area_growth:
                    continue

                current = candidate
                area_a = candidate_area
                used[j] = True
                changed = True

            used[i] = True
            next_boxes.append(current)

        merged_boxes = sorted(next_boxes, key=lambda b: (b[0], b[1]))

    return merged_boxes
