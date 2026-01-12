import numpy as np
import cv2

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    pc = w2c.dot(point)
    pc = np.array([pc[1], -pc[2], pc[0]])
    pi = K.dot(pc)
    if pi[2] != 0:
        pi[0] /= pi[2]
        pi[1] /= pi[2]
    return pi[0:2]

def finite_bbox(verts, W, H):
    xs, ys = zip(*verts)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return None
    x_min = int(np.clip(xs.min(), 0, W - 1))
    x_max = int(np.clip(xs.max(), 0, W - 1))
    y_min = int(np.clip(ys.min(), 0, H - 1))
    y_max = int(np.clip(ys.max(), 0, H - 1))
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, x_max, y_min, y_max

def filter_boxes_segmentation(boxes, seg_img, bg_thr=0.55):
    if not boxes:
        return []
    
    BACKGROUND_COLORS = np.array([
        [0, 0, 0], [232, 35, 244], [70, 70, 70], [156, 102, 102],
        [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220],
        [35, 142, 107], [152, 251, 152], [180, 130, 70], [160, 190, 110],
        [50, 120, 170], [80, 90, 55], [150, 60, 45], [50, 234, 157],
        [81, 0, 81], [100, 100, 150], [140, 150, 230], [180, 165, 180],
        [70, 130, 180]
    ], dtype=np.uint8)
    
    H, W, _ = seg_img.shape
    seg_bgr = seg_img[:, :, :3]
    
    bg_mask = np.zeros((H, W), dtype=bool)
    for color in BACKGROUND_COLORS:
        bg_mask |= np.all(seg_bgr == color, axis=-1)
    
    keep = []
    for (xmin, ymin, xmax, ymax, cid) in boxes:
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W - 1, xmax)
        ymax = min(H - 1, ymax)
        
        if xmax <= xmin or ymax <= ymin:
            continue
        
        roi_mask = bg_mask[ymin:ymax, xmin:xmax]
        if roi_mask.size == 0:
            continue
        
        bg_ratio = roi_mask.sum() / roi_mask.size
        
        if bg_ratio < bg_thr:
            keep.append((xmin, ymin, xmax, ymax, cid))
    
    return keep

def detect_static_vehicles(seg_bgr, boxes_xyxy_cls, SEG_COLORS):
    static_boxes = []
    H, W = seg_bgr.shape[:2]
    
    large_vehicle_ids = [14, 15, 16, 17]
    small_vehicle_ids = [13, 18, 19]
    
    mask_large = np.zeros((H, W), dtype=np.uint8)
    mask_small = np.zeros((H, W), dtype=np.uint8)
    id_mask_map = np.zeros((H, W), dtype=np.uint8)
    
    for vehicle_id in large_vehicle_ids:
        color = SEG_COLORS[vehicle_id][0]
        matches = np.all(seg_bgr == color, axis=-1).astype(np.uint8) * 255
        mask_large = cv2.bitwise_or(mask_large, matches)
        id_mask_map[matches > 0] = vehicle_id
    
    for vehicle_id in small_vehicle_ids:
        color = SEG_COLORS[vehicle_id][0]
        matches = np.all(seg_bgr == color, axis=-1).astype(np.uint8) * 255
        mask_small = cv2.bitwise_or(mask_small, matches)
        id_mask_map[matches > 0] = vehicle_id
    
    if boxes_xyxy_cls:
        y_grid, x_grid = np.ogrid[:H, :W]
        dynamic_mask = np.zeros((H, W), dtype=bool)
        
        for (x1, y1, x2, y2, cid) in boxes_xyxy_cls:
            if cid in [13, 14, 15, 16, 17, 18, 19]:
                dynamic_mask |= ((x_grid >= x1) & (x_grid <= x2) & 
                                 (y_grid >= y1) & (y_grid <= y2))
        
        dynamic_mask = dynamic_mask.astype(np.uint8) * 255
    else:
        dynamic_mask = np.zeros((H, W), dtype=np.uint8)
    
    static_mask_large = cv2.bitwise_and(mask_large, cv2.bitwise_not(dynamic_mask))
    static_mask_small = cv2.bitwise_and(mask_small, cv2.bitwise_not(dynamic_mask))
    
    kernel_large = np.ones((5, 5), np.uint8)
    static_mask_large = cv2.morphologyEx(static_mask_large, cv2.MORPH_OPEN, kernel_large, iterations=1)
    static_mask_large = cv2.dilate(static_mask_large, kernel_large, iterations=1)
    
    kernel_small = np.ones((3, 3), np.uint8)
    static_mask_small = cv2.morphologyEx(static_mask_small, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    
    contours_large, _ = cv2.findContours(static_mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_large:
        area = cv2.contourArea(c)
        if area > 300:
            x, y, w, h = cv2.boundingRect(c)
            roi = id_mask_map[y:y+h, x:x+w]
            pixels = roi[roi > 0]
            if pixels.size > 0:
                cid = int(np.bincount(pixels).argmax())
                static_boxes.append((x, y, x + w, y + h, cid))
    
    contours_small, _ = cv2.findContours(static_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_small:
        area = cv2.contourArea(c)
        if 30 < area < 2500:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio < 5:
                roi = id_mask_map[y:y+h, x:x+w]
                pixels = roi[roi > 0]
                if pixels.size > 0:
                    cid = int(np.bincount(pixels).argmax())
                    static_boxes.append((x, y, x + w, y + h, cid))
    
    return static_boxes
