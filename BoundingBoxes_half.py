import sys
import carla
import random
import time
import numpy as np
import pygame
import cv2
import os
import time as time_module

# Import refactored modules for configuration, CARLA management, 
# camera handling, detection algorithms, and label export
import bbox_config
import bbox_carla
import bbox_camera
import bbox_detection
import bbox_labels




#  INITIALIZATION 
cfg = bbox_config.load_config("config.yaml")
# Create fresh output directories for images, labels, and videos
bbox_config.setup_output_dirs()

# Initialize Pygame for visualization window
pygame.init()

#  CARLA CONNECTION & WORLD SETUP 
# Connect to CARLA simulator and get the world instance
client = bbox_carla.connect_client()
world = client.get_world()
# Set initial weather conditions
weather_name = cfg["carla"]["weather"]
world.set_weather(getattr(carla.WeatherParameters, weather_name))
# Clean up any actors from previous runs
bbox_carla.destroy_leftovers(world, client)
# Enable synchronous mode for deterministic simulation (20 FPS = 0.05s per tick)
bbox_carla.setup_synchronous_mode(world, fixed_delta_seconds=cfg["carla"]["fixed_delta_seconds"])

#  CAMERA SPAWNING 
# Get blueprint library and available spawn points from the map
bp_lib = world.get_blueprint_library()
original_spawn_points = world.get_map().get_spawn_points()

# Augment spawn points for heterogeneous dataset coverage
camera_spawn_points = bbox_carla.augment_spawn_points(
    original_spawn_points,
    world,
    variants     = bbox_config.SPAWN_AUG_VARIANTS,
    forward_max  = bbox_config.SPAWN_AUG_FORWARD_MAX,
    lateral_max  = bbox_config.SPAWN_AUG_LATERAL_MAX,
    yaw_max      = bbox_config.SPAWN_AUG_YAW_MAX
)

print(f"[SPAWN POINTS] Total available camera positions: {len(camera_spawn_points)} ")
print(f"[SPAWN POINTS] Total available pedestrian positions: {len(original_spawn_points)} ")
# Position camera at specified spawn point but elevated (z=60) with top-down view (pitch=-90)
cam_index = cfg["run"]["cam_index"]

if cam_index >= len(camera_spawn_points):
    raise ValueError(f"cam_index {cam_index} out of range (map has {len(camera_spawn_points)} spawn points)")

base_sp = camera_spawn_points[cam_index]

cam_trans = carla.Transform(carla.Location(x=base_sp.location.x, y=base_sp.location.y, z=base_sp.location.z +cfg["camera"]["z"]),
                             carla.Rotation(pitch=cfg["camera"]["pitch"], yaw=0.0, roll=0.0))

# Create RGB and semantic segmentation cameras with specified resolution and FOV
image_w = cfg['camera']['width']
image_h = cfg['camera']['height']
camera, segmentation_cam, camera_data, segmentation_data, camera_bp, semantic_bp = bbox_camera.create_cameras(
    world, bp_lib, cam_trans, cfg["camera"]["fov"], image_w, image_h
)

all_sensors = [camera, segmentation_cam]
# Create Pygame window for live visualization
screen = pygame.display.set_mode((image_w, image_h))
pygame.display.set_caption("CARLA view")

raw_writer = None
boxed_writer = None
segmentation_writer = None

#  PROJECTION MATRIX 
# Build camera intrinsic matrix for 3D to 2D projection
K = bbox_detection.build_projection_matrix(image_w, image_h, cfg["camera"]["fov"])

#  TRAFFIC SPAWNING 
# Add this block right after the vehicle spawning loop (after line 105):
traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(2.0)

# THIS is the key line — stops TM from spawning its own vehicles
traffic_manager.set_synchronous_mode(True)
#Spawn vehicles with autopilot for dynamic scene
vehicle_bps = bp_lib.filter('*vehicle*')
spawned_vehicles = []
for _ in range(cfg["traffic"]["vehicles"]):
    sp = random.choice(original_spawn_points)
    vbp = random.choice(vehicle_bps)
    spawned_vehicle = world.try_spawn_actor(vbp, sp)
    if spawned_vehicle:
        spawned_vehicle.set_autopilot(True, traffic_manager.get_port())
        spawned_vehicles.append(spawned_vehicle)
        print(
            "[SPAWNED]",
            spawned_vehicle.type_id,
            "semantic_tags =",
            spawned_vehicle.semantic_tags
        )

print(f"[TRAFFIC] Spawned {len(spawned_vehicles)} vehicles")

# Spawn pedestrian walkers with AI controllers for realistic movement
walker_bps = bp_lib.filter('*walker*')
ctrl_bp = bp_lib.find('controller.ai.walker')

spawned_walkers = []
walker_controllers = []

# Step 1: Spawn ALL walkers first
for _ in range(cfg["traffic"]["walkers"]):
    sp = carla.Transform()
    sp.location = world.get_random_location_from_navigation()
    if sp.location:
        wbp = random.choice(walker_bps)
        walker = world.try_spawn_actor(wbp, sp)
        if walker:
            spawned_walkers.append(walker)

# Step 2: Tick once so all walkers are registered in simulation
world.tick()

# Step 3: Attach AI controller to each walker
for walker in spawned_walkers:
    try:
        ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
        walker_controllers.append(ctrl)
    except Exception as e:
        print(f"[WARN] Controller attach failed: {e}")

# Step 4: Tick once so all controllers are registered
world.tick()

# Step 5: Start all controllers
for ctrl in walker_controllers:
    try:
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(1 + random.random())
    except RuntimeError as e:
        print(f"[WARN] ctrl.start() failed (skipping): {e}")

print(f"[TRAFFIC] {len(spawned_walkers)} walkers | {len(walker_controllers)} controllers ready")

#  PERFORMANCE & EXPORT SETTINGS 

# Enable exporting annotated frames to disk
ENABLE_EXPORTS = cfg["export"]["enable"]
# Export every Nth frame within the export window
EXPORT_INTERVAL = cfg["export"]["export_interval"]
# Start exporting after initial frames to allow scene stabilization
EXPORT_START_FRAME = cfg["export"]["export_start_frame"]
# Stop exporting before simulation ends (95% of total frames)
EXPORT_END_PERCENT = cfg["export"]["export_end_percent"]
# Maximum number of frames to export
MAX_EXPORTS = cfg["export"]["max_exports"]
# Changing spawn points and weather every N frames for dataset diversity
SPAWN_CHANGE_INTERVAL = cfg["export"]["spawn_change_interval"]
WEATHER_CHANGE_INTERVAL = cfg["export"]["weather_change_interval"]

#  STATE TRACKING VARIABLES 
# Counter for exported frames
export_count = 0
# Counter for frame synchronization timeouts
timeout_count = 0
# Last frame where FPS was reported
last_fps_report = 0
# Track ALL camera pairs ever created so cleanup never misses one

# Timing and weather control
start_time = time.time()
frame_count = 0
weather_idx = 0
# Calculate target frame count based on requested duration
duration = cfg["run"]["duration"]
target_frames = int(duration / world.get_settings().fixed_delta_seconds)
EXPORT_END_FRAME = int(target_frames * EXPORT_END_PERCENT)
try:
    #  MAIN SIMULATION LOOP 
    while True:

        
        # Check if simulation duration has been reached
        if frame_count >= target_frames:
            print(f"Finished {duration}s capture ({frame_count} frames, {export_count} exported).")
            break
        
        # Advance simulation by one tick
        world.tick()
        
        if camera_data['frame'] < frame_count or segmentation_data['frame'] < frame_count:
            frame_count += 1
            continue

        # Retrieve synchronized images from both cameras
        img = camera_data['image']
        seg_img = segmentation_data['image']
        
        # Validate that images are not empty
        if img is None or img.size == 0 or seg_img is None or seg_img.size == 0:
            frame_count += 1
            continue

        #  CAMERA TRANSFORM & PROJECTION 
        # Get current camera transform and compute world-to-camera matrix
        cam_tf = camera.get_transform()
        try:
            w2c = np.array(cam_tf.get_inverse_matrix())
        except AttributeError:
            w2c = np.linalg.inv(np.array(cam_tf.get_matrix(), dtype=float))

        #  DYNAMIC ACTOR DETECTION 
        # Project 3D bounding boxes of all vehicles and walkers onto 2D image plane
        boxes_xyxy_cls = []
        all_spawned_boxes = []  # Store ALL projected boxes (before filtering)
        actors = world.get_actors()
        
        # Detect vehicles (cars, trucks, buses, motorcycles, bicycles)
        for a in actors.filter('*vehicle*'):
            bb = a.bounding_box
            # Transform bounding box vertices from world to camera coordinates
            verts = [bbox_detection.get_image_point(v, K, w2c) for v in bb.get_world_vertices(a.get_transform())]
            
            # Get semantic tag (class ID) or default to 'car' (14)
            cid = a.semantic_tags[0] if a.semantic_tags else 14
            
            # Use larger minimum size for small vehicles (motorcycles, bicycles)
            min_size = 5 if cid in [18, 19] else 1
            
            # Compute 2D bounding box and validate it's within image bounds
            mm = bbox_detection.finite_bbox(verts, image_w, image_h, min_size=min_size)
            if mm:
                x_min, x_max, y_min, y_max = mm
                boxes_xyxy_cls.append((x_min, y_min, x_max, y_max, cid))
                # Store BEFORE filtering - we need to erase ALL actors from static mask
                all_spawned_boxes.append((x_min, y_min, x_max, y_max))
        
        # Detect walkers/pedestrians (riders, pedestrians)
        for a in actors.filter('*walker*'):
            bb = a.bounding_box
            # Transform bounding box vertices from world to camera coordinates
            verts = [bbox_detection.get_image_point(v, K, w2c) for v in bb.get_world_vertices(a.get_transform())]
            
            # Get semantic tag (class ID) or default to 'pedestrian' (12)
            cid = a.semantic_tags[0] if a.semantic_tags else 12
            
            # Use larger minimum size for riders
            min_size = 5 if cid == 13 else 1
            
            # Compute 2D bounding box and validate it's within image bounds
            mm = bbox_detection.finite_bbox(verts, image_w, image_h, min_size=min_size)
            if mm:
                x_min, x_max, y_min, y_max = mm
                boxes_xyxy_cls.append((x_min, y_min, x_max, y_max, cid))
                # Store BEFORE filtering - we need to erase ALL actors from static mask
                all_spawned_boxes.append((x_min, y_min, x_max, y_max))

        # REMOVE ghost / occluded bounding boxes using semantic segmentation
        boxes_xyxy_cls = bbox_detection.filter_boxes_segmentation(
            boxes_xyxy_cls,
            seg_img,
            bg_thr=0.40,
            camera_z=cfg["camera"]["z"]
        )
        # REMOVE small ghost boxes that follow/overlap a larger vehicle box
        boxes_xyxy_cls = bbox_detection.suppress_contained_boxes(
            boxes_xyxy_cls,
            iou_threshold=0.2
        )

        # We erase ALL spawned actors from static mask, even occluded ones
        # This prevents dynamic vehicles from being detected as static when occluded

        # Determine if this frame should be exported based on settings
        is_export_frame = (ENABLE_EXPORTS and 
                          frame_count >= EXPORT_START_FRAME and 
                          frame_count <= EXPORT_END_FRAME and
                          frame_count % EXPORT_INTERVAL == 0 and
                          export_count < MAX_EXPORTS and len(boxes_xyxy_cls) > 0) 
        
        #  FRAME EXPORT 
        # Save frame and annotations if this is an export frame
        if is_export_frame:
            
            frame_id = f"{frame_count:06d}"
            img_bgr = img[:, :, :3]
            seg_bgr = seg_img[:, :, :3]

            # 1) Export Save RGB image to disk
            img_path = os.path.join(bbox_config.IMG_RGB_DIR, f"frame_{frame_id}.png")
            cv2.imwrite(img_path, img_bgr)

            # 2) Export RGB frame WITH bounding boxes (BOXED IMAGE)
            boxed_path = os.path.join(bbox_config.IMG_BOXED_DIR, f"frame_{frame_id}_boxed.png")

            # 3) Export segmentation mask
            seg_path = os.path.join(bbox_config.IMG_SEG_DIR, f"frame_{frame_id}_seg.png")
            cv2.imwrite(seg_path, seg_bgr)

            # 4) Export CLASS-ID semantic mask (raw IDs)
            classid_mask = segmentation_data['labels']  # H x W, uint8
            # DEBUG: verify semantic IDs (print once or a few times)
            if export_count > 10:   # avoid spamming
                ids, counts = np.unique(classid_mask, return_counts=True)
                # DEBUG SHOWS IDS AND PIXEL COUNTS
                #print(dict(zip(ids, counts)))
            
            classid_path = os.path.join(
                bbox_config.IMG_DETMASK_DIR,
                f"frame_{frame_id}_classid.png"
            )

            cv2.imwrite(classid_path, classid_mask)



            # Collect dynamic boxes to avoid duplicates
            dynamic_boxes = [
                (xmin, ymin, xmax, ymax)
                for (xmin, ymin, xmax, ymax, _) in boxes_xyxy_cls
            ]
            
            
          
            # CREATE STATIC-ONLY MASK: Erase all spawned actors (FAST)
            # Reuse already-projected boxes from detection phase above
            
            static_classid_mask = classid_mask.copy()
            static_seg_bgr = seg_bgr.copy()
            pad = 8  # small padding to ensure full erasure
            # Zero out all spawned actor regions (already computed above!)
            # for (x_min, y_min, x_max, y_max) in all_spawned_boxes:
            #     static_classid_mask[y_min:y_max, x_min:x_max] = 0
            #     static_seg_bgr[y_min:y_max, x_min:x_max] = (0, 0, 0)

            for (x_min, y_min, x_max, y_max) in all_spawned_boxes:
                box_w = x_max - x_min
                box_h = y_max - y_min
                dynamic_pad = max(pad, int(max(box_w, box_h) * 0.15))
                y1c = max(0, y_min - dynamic_pad)
                y2c = min(image_h, y_max + dynamic_pad)
                x1c = max(0, x_min - dynamic_pad)
                x2c = min(image_w, x_max + dynamic_pad)
                static_classid_mask[y1c:y2c, x1c:x2c] = 0
                static_seg_bgr[y1c:y2c, x1c:x2c] = (0, 0, 0)            
         
            # DERIVE STATIC BOXES FROM CLEANED MASK
            # Now this mask contains ONLY map objects, no spawned actors

            mask = static_classid_mask.copy()

            # 1) bicycles / motorcycles - use erosion to separate touching objects
            kernel_erode = np.ones((2, 2), np.uint8)  # Small erosion to separate
            kernel_dilate = np.ones((3, 3), np.uint8)  # Slightly larger dilation to restore size
            for cid in (18, 19):
                m = (mask == cid).astype(np.uint8)
                # Erode to separate touching objects
                m = cv2.erode(m, kernel_erode, iterations=1)
                # Dilate to restore approximate original size
                m = cv2.dilate(m, kernel_dilate, iterations=1)
                # Light closing to reconnect slightly fragmented parts (not touching objects!)
                kernel_close = np.ones((3, 3), np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                mask[mask == cid] = 0  # Clear old pixels
                mask[m == 1] = cid     # Set new pixels

            # 2) cars / trucks / buses (conservative - just light cleanup)
            kernel_car = np.ones((7, 7), np.uint8)
            for cid in (13, 14, 15, 16, 17):  # rider, car, truck, bus, train
                m = (mask == cid).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_car, iterations=1)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_car, iterations=1)
                mask[mask == cid] = 0
                mask[m == 1] = cid

            # extract static boxes
            for cid in bbox_config.STATIC_CLASS_IDS:
                static_boxes = bbox_detection.extract_static_boxes(
                    mask,
                    cid,
                    min_area=1 if cid in (18, 19) else 80
                )

                for (xmin, ymin, xmax, ymax) in static_boxes:
                    if (xmax - xmin) < 6 or (ymax - ymin) < 6:
                        continue

                    # Reject boxes where the object doesn't fill enough of the bbox
                    # This prevents poles/lines splitting one car into two boxes
                    roi = mask[ymin:ymax, xmin:xmax]
                    total_pixels = roi.shape[0] * roi.shape[1]
                    if total_pixels == 0:
                        continue
                    visible_pixels = np.sum(roi == cid)
                    visibility_ratio = visible_pixels / total_pixels
                    
                    min_vis = 0.15 if cid in (18, 19) else 0.40  # allow lower visibility for small objects
                    if visibility_ratio < min_vis:
                        continue

                    boxes_xyxy_cls.append((xmin, ymin, xmax, ymax, cid))
            # Deduplicate across dynamic + static pipelines
            boxes_xyxy_cls = bbox_detection.nms_boxes(boxes_xyxy_cls, iou_threshold=0.35)            

            overlay = img_bgr.copy()
            for (x1, y1, x2, y2, cid) in boxes_xyxy_cls:
                color = bbox_config.CLASS_COLORS.get(cid, bbox_config.DEFAULT_COLOR)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

            static_seg_path = os.path.join(bbox_config.IMG_STATIC_DEBUG_DIR, f"frame_{frame_id}_static_seg.png")
            cv2.imwrite(static_seg_path, static_seg_bgr)
            cv2.imwrite(boxed_path, overlay)

            # 5) Export bounding box annotations in Pascal VOC XML format
            voc_path = os.path.join(bbox_config.VOC_DIR, f"frame_{frame_id}.xml")
            bbox_labels.write_voc_xml(voc_path, boxes_xyxy_cls, image_w, image_h, img_path, bbox_config.CLASS_NAMES)
            
            export_count += 1
            print(f"[EXPORT {export_count}/{MAX_EXPORTS}] Frame {frame_count}: {len(boxes_xyxy_cls)} boxes")
            
        
        frame_id = f"{frame_count:06d}"
        img_bgr = img[:, :, :3]
        
        #  VIDEO ANNOTATION 
        # Draw bounding boxes on image and write to boxed video
        overlay = img_bgr.copy()
        for (x1, y1, x2, y2, cid) in boxes_xyxy_cls:
            # Green for vehicles/riders/etc, blue for others
            color = (0, 255, 0) if cid in [13,14,15,16,17,18,19] else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        #boxed_writer.write(overlay)
        
        seg_bgr = seg_img[:, :, :3]
        # Optional: Write segmentation video (currently disabled)
        #segmentation_writer.write(seg_bgr)

        #  PYGAME VISUALIZATION 
        # Display current frame in Pygame window
        rgb = img_bgr[:, :, ::-1]
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
    
        #  PERFORMANCE REPORTING 
        # Periodically report FPS and statistics
        if frame_count - last_fps_report >= 100:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[FPS] Frame {frame_count}: {current_fps:.1f} FPS | Timeouts: {timeout_count} | Boxes: {len(boxes_xyxy_cls)}")
            last_fps_report = frame_count

        #  USER INPUT HANDLING 
        # Check for quit events
        quit_now = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_now = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit_now = True

        if quit_now:
            raise KeyboardInterrupt

        keys = pygame.key.get_pressed()

        #  WEATHER CONTROL 
        # Use bracket keys to cycle through weather presets
        if keys[pygame.K_LEFTBRACKET]:
            weather_idx = (weather_idx - 1) % len(bbox_config.weather_presets)
            world.set_weather(bbox_config.weather_presets[weather_idx])
        elif keys[pygame.K_RIGHTBRACKET]:
            weather_idx = (weather_idx + 1) % len(bbox_config.weather_presets)
            world.set_weather(bbox_config.weather_presets[weather_idx])
        
        #  CAMERA REPOSITIONING 
        # Press 'C' key to move cameras to a random spawn point
        if keys[pygame.K_c]:
            print("Repositioning cameras...")
            all_sensors = bbox_carla.safe_destroy_cameras(camera, segmentation_cam, all_sensors)
            world.tick()

            new_sp = random.choice(camera_spawn_points)     # ← augmented only
            new_cam_trans = carla.Transform(
                carla.Location(x=new_sp.location.x, y=new_sp.location.y,
                               z=new_sp.location.z + cfg["camera"]["z"]),
                carla.Rotation(pitch=cfg["camera"]["pitch"], yaw=0.0, roll=0.0)
            )
            camera, segmentation_cam, camera_data, segmentation_data, _, _ = \
                bbox_camera.create_cameras(world, bp_lib, new_cam_trans,
                                           cfg["camera"]["fov"], image_w, image_h)
            all_sensors.extend([camera, segmentation_cam])
            world.tick()
            print("[Reposition] Camera moved.")

        #  AUTO CAMERA REPOSITIONING 
        if SPAWN_CHANGE_INTERVAL > 0 and frame_count > 0 and frame_count % SPAWN_CHANGE_INTERVAL == 0:
            print(f"[AUTO REPOSITION] Frame {frame_count} — moving camera...")
            all_sensors = bbox_carla.safe_destroy_cameras(camera, segmentation_cam, all_sensors)
            world.tick()

            new_sp = random.choice(camera_spawn_points)     # ← augmented only
            new_cam_trans = carla.Transform(
                carla.Location(x=new_sp.location.x, y=new_sp.location.y,
                               z=new_sp.location.z + cfg["camera"]["z"]),
                carla.Rotation(pitch=cfg["camera"]["pitch"], yaw=0.0, roll=0.0)
            )
            camera, segmentation_cam, camera_data, segmentation_data, _, _ = \
                bbox_camera.create_cameras(world, bp_lib, new_cam_trans,
                                           cfg["camera"]["fov"], image_w, image_h)
            all_sensors.extend([camera, segmentation_cam])
            world.tick()
            print("[AUTO REPOSITION] Done.")

        frame_count += 1

    #  SIMULATION COMPLETE 
    # Report final statistics
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"Captured {frame_count} frames in {elapsed:.2f}s -> {actual_fps:.2f} FPS")
    print(f"Exported {export_count} frames to {bbox_config.IMG_DIR} and {bbox_config.VOC_DIR}")
    print(f"Sync timeouts: {timeout_count} ({100*timeout_count/frame_count:.1f}%)")
    print(f"{'='*60}")

except KeyboardInterrupt:
    #  INTERRUPTED SHUTDOWN 
    # User manually stopped the simulation
    print(f"\nInterrupted. Exported {export_count} frames before exit.")
    print(f"Sync timeouts: {timeout_count}")

finally:
    #  CLEANUP & RESOURCE RELEASE 
    # Release all video writers
    for writer in [raw_writer, segmentation_writer, boxed_writer]:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
    
    # Destroy all spawned actors (cameras, vehicles, walkers)
    #bbox_carla.cleanup_actors(client, world, sensors=[camera, segmentation_cam])
    bbox_carla.cleanup_actors(client, world, sensors=all_sensors)
    # Close Pygame window
    pygame.quit()
    print("Shutdown complete.")