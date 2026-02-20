import sys
import carla
import random
import time
import numpy as np
import pygame
import cv2
import os
import time as time_module
import concurrent.futures

# Import refactored modules for configuration, CARLA management, 
# camera handling, detection algorithms, and label export
import bbox_config
import bbox_carla
import bbox_camera
import bbox_detection
import bbox_labels
import bbox_camera_specs


# ── Background I/O helpers ──────────────────────────────────────────
io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def _write_image(path, img):
    """Write an image to disk (runs in background thread)."""
    cv2.imwrite(path, img)

def _write_xml(path, boxes, w, h, img_path, class_names):
    """Write a VOC XML file to disk (runs in background thread)."""
    bbox_labels.write_voc_xml(path, boxes, w, h, img_path, class_names)
# ────────────────────────────────────────────────────────────────────


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
spawn_points = world.get_map().get_spawn_points()

image_w = cfg['camera']['width']
image_h = cfg['camera']['height']

# Position camera at specified spawn point but elevated (z=60) with top-down view (pitch=-90)
cam_index = cfg["run"]["cam_index"]

if cam_index >= len(spawn_points):
    raise ValueError(f"cam_index {cam_index} out of range (map has {len(spawn_points)} spawn points)")

base_sp = spawn_points[cam_index]


# Spawn 3 overlapping cameras at this location
print(f"\n=== SPAWNING 3 CAMERAS AT SPAWN POINT {cam_index} ===")

camera_sets = bbox_camera.spawn_three_cameras_at_spawn_point(
    world, bp_lib, base_sp, cfg, 
    spacing=25.0,          # 15 meters between cameras
    lateral_offset=0.5    # Small lateral shift for overlap
)
# Unpack all cameras
all_cameras = []
all_seg_cameras = []
all_camera_data = []
all_seg_data = []

for cam_set in camera_sets:
    cam, seg_cam, cam_data, seg_data, cam_bp, seg_bp = cam_set
    all_cameras.append(cam)
    all_seg_cameras.append(seg_cam)
    all_camera_data.append(cam_data)
    all_seg_data.append(seg_data)

# Use middle camera (index 1) for Pygame display
primary_idx = 1

print(f"\nPrimary display: Camera {primary_idx} (middle)")
print(f"Total cameras processing: {len(all_cameras)}\n")

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
#Spawn vehicles with autopilot for dynamic scene
vehicle_bps = bp_lib.filter('*vehicle*')
for _ in range(cfg["traffic"]["vehicles"]):
    sp = random.choice(world.get_map().get_spawn_points())
    vbp = random.choice(vehicle_bps)
    spawned_vehicle = world.try_spawn_actor(vbp, sp)
    if spawned_vehicle:
        spawned_vehicle.set_autopilot(True)
                # DEBUG: print semantic class IDs
        print(
            "[SPAWNED]",
            spawned_vehicle.type_id,
            "semantic_tags =",
            spawned_vehicle.semantic_tags
        )

# Spawn pedestrian walkers with AI controllers
walker_bps = [bp for bp in bp_lib.filter('walker.pedestrian.*')]
spawned_walkers = []
spawned_controllers = []

print(f"\nSpawning {cfg['traffic']['walkers']} walkers...")

for i in range(cfg["traffic"]["walkers"]):
    # Get random spawn location
    spawn_location = world.get_random_location_from_navigation()
    if spawn_location is None:
        print(f"  [WALKER {i+1}] Failed: no navigation location")
        continue
    
    # Spawn walker
    walker_bp = random.choice(walker_bps)
    walker_transform = carla.Transform(spawn_location)
    walker = world.try_spawn_actor(walker_bp, walker_transform)
    
    if walker is None:
        print(f"  [WALKER {i+1}] Failed: spawn returned None")
        continue
    
    spawned_walkers.append(walker)
    print(f"[WALKER {len(spawned_walkers)}] {walker.type_id} id={walker.id} semantic={walker.semantic_tags} at ({spawn_location.x:.1f}, {spawn_location.y:.1f})")
    
    world.tick()  # Let walker fully spawn
    
    # Spawn controller
    controller_bp = bp_lib.find('controller.ai.walker')
    controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
    
    if controller is None:
        print(f"  └─ [CONTROLLER] FAILED for walker {walker.id}")
        continue
    
    spawned_controllers.append(controller)
    world.tick()  # Let controller attach
    
    # Start walking
    try:
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.0 + random.random())
        print(f"  └─ [CONTROLLER] id={controller.id} attached and started")
    except Exception as e:
        print(f"  └─ [CONTROLLER] Failed to start: {e}")

world.tick()
print(f"\nSpawned {len(spawned_walkers)} walkers with {len(spawned_controllers)} controllers")

# Verify world state
actors_now = world.get_actors()
vehicles = actors_now.filter('vehicle.*')
walkers = actors_now.filter('walker.pedestrian.*')
controllers = actors_now.filter('controller.ai.walker')

print(f"\n[WORLD STATE VERIFICATION]")
print(f"  Vehicles:    {len(vehicles)}")
print(f"  Walkers:     {len(walkers)}")
print(f"  Controllers: {len(controllers)}")

# Debug: print first few walker IDs
print(f"\n  Walker actor IDs in world:")
for w in list(walkers)[:5]:
    print(f"    {w.id}: {w.type_id}")

print()

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



#  STATE TRACKING VARIABLES 
# Counter for exported frames
export_count = 0
# Counter for frame synchronization timeouts
timeout_count = 0
# Last frame where FPS was reported
last_fps_report = 0

# Timing and weather control
start_time = time.time()
frame_count = 0
weather_idx = 0
bbox_camera_specs.print_camera_specs(
    fov_horizontal_deg=cfg['camera']['fov'],
    image_width=image_w,
    image_height=image_h,
    altitude_m=cfg['camera']['z']
)

# Helper to respawn all 3 camera sets at a new spawn point
def respawn_all_cameras(base_spawn_point):
    """Destroy old cameras and spawn new ones. Returns updated lists."""
    global all_cameras, all_seg_cameras, all_camera_data, all_seg_data
    
    # Stop and destroy all existing cameras
    for i in range(len(all_cameras)):
        try:
            all_cameras[i].stop()
        except Exception:
            pass
        try:
            all_seg_cameras[i].stop()
        except Exception:
            pass
    
    world.tick()
    time.sleep(0.05)
    
    for i in range(len(all_cameras)):
        try:
            all_cameras[i].destroy()
        except Exception:
            pass
        try:
            all_seg_cameras[i].destroy()
        except Exception:
            pass
    
    world.tick()
    
    # Spawn new cameras
    new_camera_sets = bbox_camera.spawn_three_cameras_at_spawn_point(
        world, bp_lib, base_spawn_point, cfg,
        spacing=25.0,
        lateral_offset=0.5
    )
    
    all_cameras = []
    all_seg_cameras = []
    all_camera_data = []
    all_seg_data = []
    
    for cam_set in new_camera_sets:
        cam, seg_cam, cam_data, seg_data, cam_bp, seg_bp = cam_set
        all_cameras.append(cam)
        all_seg_cameras.append(seg_cam)
        all_camera_data.append(cam_data)
        all_seg_data.append(seg_data)
    
    world.tick()
    time.sleep(0.05)
    
    return all_cameras, all_seg_cameras, all_camera_data, all_seg_data


try:
    #  MAIN SIMULATION LOOP 
    while True:
        # Calculate target frame count based on requested duration
        duration = cfg["run"]["duration"]
        target_frames = int(duration / world.get_settings().fixed_delta_seconds)
        EXPORT_END_FRAME = int(target_frames * EXPORT_END_PERCENT)
        
        # Check if simulation duration has been reached
        if frame_count >= target_frames:
            print(f"Finished {duration}s capture ({frame_count} frames, {export_count} exported).")
            break
        
        # Advance simulation by one tick
        world.tick()
        
        # Check primary camera sync (quick rejection)
        if all_camera_data[primary_idx]['frame'] < frame_count or all_seg_data[primary_idx]['frame'] < frame_count:
            frame_count += 1
            continue

        # Retrieve synchronized images from primary camera for display
        img = all_camera_data[primary_idx]['image']
        seg_img = all_seg_data[primary_idx]['image']
        
        # Validate that images are not empty
        if img is None or img.size == 0 or seg_img is None or seg_img.size == 0:
            frame_count += 1
            continue

        # ============================================================
        # PERFORMANCE OPTIMIZATION: Fetch actors and transforms ONCE
        # ============================================================
        actors = world.get_actors()
        
        # Pre-fetch all actor data in one pass (avoid repeated IPC calls)
        actor_cache_vehicles = []
        for a in actors.filter('*vehicle*'):
            bb = a.bounding_box
            tf = a.get_transform()
            cid = a.semantic_tags[0] if a.semantic_tags else 14
            verts_world = bb.get_world_vertices(tf)
            actor_cache_vehicles.append((verts_world, cid))
        
        actor_cache_walkers = []
        for a in actors.filter('*walker*'):
            bb = a.bounding_box
            tf = a.get_transform()
            cid = a.semantic_tags[0] if a.semantic_tags else 12
            verts_world = bb.get_world_vertices(tf)
            actor_cache_walkers.append((verts_world, cid))

        # Determine if this frame should be exported
        is_export_frame = (ENABLE_EXPORTS and 
                          frame_count >= EXPORT_START_FRAME and 
                          frame_count <= EXPORT_END_FRAME and
                          frame_count % EXPORT_INTERVAL == 0 and
                          export_count < MAX_EXPORTS)

        # Decide which cameras to process this frame
        if is_export_frame:
            cameras_to_process = list(range(len(all_cameras)))
        else:
            # Only process primary camera for display
            cameras_to_process = [primary_idx]

        #  DYNAMIC ACTOR DETECTION 
        all_boxes_per_camera = {}

        for cam_idx in cameras_to_process:
            cam = all_cameras[cam_idx]
            cam_data_dict = all_camera_data[cam_idx]
            seg_data_dict = all_seg_data[cam_idx]

            # Check frame sync
            if cam_data_dict['frame'] < frame_count or seg_data_dict['frame'] < frame_count:
                continue

            img_cam = cam_data_dict['image']
            seg_img_cam = seg_data_dict['image']

            if img_cam is None or img_cam.size == 0 or seg_img_cam is None or seg_img_cam.size == 0:
                continue

            # Get camera w2c matrix
            cam_tf = cam.get_transform()
            try:
                w2c_cam = np.array(cam_tf.get_inverse_matrix())
            except AttributeError:
                w2c_cam = np.linalg.inv(np.array(cam_tf.get_matrix(), dtype=float))

            # Project cached actor data into THIS camera
            boxes_cam = []
            spawned_boxes_cam = []

            for (verts_world, cid) in actor_cache_vehicles:
                verts = [bbox_detection.get_image_point(v, K, w2c_cam) for v in verts_world]
                min_size = 5 if cid in [18, 19] else 1
                mm = bbox_detection.finite_bbox(verts, image_w, image_h, min_size=min_size)
                if mm:
                    x_min, x_max, y_min, y_max = mm
                    boxes_cam.append((x_min, y_min, x_max, y_max, cid))
                    spawned_boxes_cam.append((x_min, y_min, x_max, y_max))

            for (verts_world, cid) in actor_cache_walkers:
                verts = [bbox_detection.get_image_point(v, K, w2c_cam) for v in verts_world]
                min_size = 5 if cid == 13 else 1
                mm = bbox_detection.finite_bbox(verts, image_w, image_h, min_size=min_size)
                if mm:
                    x_min, x_max, y_min, y_max = mm
                    boxes_cam.append((x_min, y_min, x_max, y_max, cid))
                    spawned_boxes_cam.append((x_min, y_min, x_max, y_max))

            # Filter occluded boxes using segmentation
            boxes_cam = bbox_detection.filter_boxes_segmentation(boxes_cam, seg_img_cam, bg_thr=0.40)

            all_boxes_per_camera[cam_idx] = {
                'camera_idx': cam_idx,
                'boxes': boxes_cam,
                'spawned_boxes': spawned_boxes_cam,
                'image': img_cam,
                'seg_image': seg_img_cam,
                'seg_labels': seg_data_dict['labels']
            }

        # Get primary camera results for display
        if primary_idx not in all_boxes_per_camera:
            frame_count += 1
            continue

        primary_cam_result = all_boxes_per_camera[primary_idx]
        boxes_xyxy_cls = primary_cam_result['boxes']
        img = primary_cam_result['image']
        seg_img = primary_cam_result['seg_image']

        #  FRAME EXPORT (ALL 3 CAMERAS) — ASYNC DISK WRITES 
        if is_export_frame:
            frame_id = f"{frame_count:06d}"

            for cam_idx, cam_data_result in all_boxes_per_camera.items():
                boxes_cam = cam_data_result['boxes']
                spawned_boxes_cam = cam_data_result['spawned_boxes']
                img_cam = cam_data_result['image']
                seg_img_cam = cam_data_result['seg_image']
                classid_mask_cam = cam_data_result['seg_labels']

                cam_suffix = f"_cam{cam_idx}"

                img_bgr = img_cam[:, :, :3]
                seg_bgr = seg_img_cam[:, :, :3]

                # 1) Export RGB image (background thread)
                img_path = os.path.join(bbox_config.IMG_RGB_DIR, f"frame_{frame_id}{cam_suffix}.png")
                io_executor.submit(_write_image, img_path, img_bgr.copy())

                # 2) Export segmentation mask (background thread)
                seg_path = os.path.join(bbox_config.IMG_SEG_DIR, f"frame_{frame_id}{cam_suffix}_seg.png")
                io_executor.submit(_write_image, seg_path, seg_bgr.copy())

                # 3) Export CLASS-ID semantic mask (background thread)
                classid_path = os.path.join(
                    bbox_config.IMG_DETMASK_DIR,
                    f"frame_{frame_id}{cam_suffix}_classid.png"
                )
                io_executor.submit(_write_image, classid_path, classid_mask_cam.copy())

                # ===========================================================
                # CREATE STATIC-ONLY MASK: Erase ALL dynamic actors
                # ===========================================================
                static_classid_mask = classid_mask_cam.copy()
                static_seg_bgr = seg_bgr.copy()

                expansion = 10
                for (x_min, y_min, x_max, y_max) in spawned_boxes_cam:
                    x_min_exp = max(0, x_min - expansion)
                    y_min_exp = max(0, y_min - expansion)
                    x_max_exp = min(image_w, x_max + expansion)
                    y_max_exp = min(image_h, y_max + expansion)
                    static_classid_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 0
                    static_seg_bgr[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = (0, 0, 0)

                # ===========================================================
                # DERIVE STATIC BOXES FROM CLEANED MASK
                # ===========================================================
                mask = static_classid_mask.copy()

                kernel_erode = np.ones((2, 2), np.uint8)
                kernel_dilate = np.ones((3, 3), np.uint8)
                for cid in (18, 19):
                    m = (mask == cid).astype(np.uint8)
                    m = cv2.erode(m, kernel_erode, iterations=1)
                    m = cv2.dilate(m, kernel_dilate, iterations=1)
                    kernel_close = np.ones((3, 3), np.uint8)
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                    mask[mask == cid] = 0
                    mask[m == 1] = cid

                kernel_car = np.ones((3, 3), np.uint8)
                for cid in (13, 14, 15, 16, 17):
                    m = (mask == cid).astype(np.uint8)
                    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_car, iterations=1)
                    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_car, iterations=1)
                    mask[mask == cid] = 0
                    mask[m == 1] = cid

                all_boxes_cam = list(boxes_cam)

                for cid in bbox_config.STATIC_CLASS_IDS:
                    static_boxes = bbox_detection.extract_static_boxes(
                        mask, cid, min_area=30 if cid in (18, 19) else 80
                    )
                    for (xmin, ymin, xmax, ymax) in static_boxes:
                        if (xmax - xmin) < 6 or (ymax - ymin) < 6:
                            continue
                        roi = mask[ymin:ymax, xmin:xmax]
                        total_pixels = roi.shape[0] * roi.shape[1]
                        if total_pixels == 0:
                            continue
                        visible_pixels = np.sum(roi == cid)
                        visibility_ratio = visible_pixels / total_pixels
                        if visibility_ratio < 0.60:
                            continue
                        all_boxes_cam.append((xmin, ymin, xmax, ymax, cid))

                # 4) Export boxed image (background thread)
                overlay = img_bgr.copy()
                for (x1, y1, x2, y2, cid) in all_boxes_cam:
                    color = bbox_config.CLASS_COLORS.get(cid, bbox_config.DEFAULT_COLOR)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

                boxed_path = os.path.join(bbox_config.IMG_BOXED_DIR, f"frame_{frame_id}{cam_suffix}_boxed.png")
                io_executor.submit(_write_image, boxed_path, overlay.copy())

                # 5) Export VOC XML (background thread)
                voc_path = os.path.join(bbox_config.VOC_DIR, f"frame_{frame_id}{cam_suffix}.xml")
                io_executor.submit(_write_xml, voc_path, list(all_boxes_cam), image_w, image_h, img_path, bbox_config.CLASS_NAMES)

                print(f"[EXPORT CAM {cam_idx}] Frame {frame_count}: {len(all_boxes_cam)} boxes")

            export_count += 1
            print(f"[EXPORT {export_count}/{MAX_EXPORTS}] Frame {frame_count}: {len(cameras_to_process)} cameras exported\n")

        frame_id = f"{frame_count:06d}"
        img_bgr = img[:, :, :3]

        #  VIDEO ANNOTATION 
        overlay = img_bgr.copy()
        for (x1, y1, x2, y2, cid) in boxes_xyxy_cls:
            color = (0, 255, 0) if cid in [13,14,15,16,17,18,19] else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

        seg_bgr = seg_img[:, :, :3]

        #  PYGAME VISUALIZATION 
        rgb = img_bgr[:, :, ::-1]
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        #  PERFORMANCE REPORTING 
        if frame_count - last_fps_report >= 100:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[FPS] Frame {frame_count}: {current_fps:.1f} FPS | Timeouts: {timeout_count} | Boxes: {len(boxes_xyxy_cls)}")
            last_fps_report = frame_count

        #  USER INPUT HANDLING 
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
        if keys[pygame.K_LEFTBRACKET]:
            weather_idx = (weather_idx - 1) % len(bbox_config.weather_presets)
            world.set_weather(bbox_config.weather_presets[weather_idx])
        elif keys[pygame.K_RIGHTBRACKET]:
            weather_idx = (weather_idx + 1) % len(bbox_config.weather_presets)
            world.set_weather(bbox_config.weather_presets[weather_idx])

        #  CAMERA REPOSITIONING (FIXED: respawn ALL 3 camera sets) 
        if keys[pygame.K_c]:
            print("Repositioning ALL cameras...")
            new_sp = random.choice(spawn_points)
            respawn_all_cameras(new_sp)
            print(f"All cameras moved to spawn {spawn_points.index(new_sp)}")
            world.tick()

        frame_count += 1

    #  SIMULATION COMPLETE 
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"Captured {frame_count} frames in {elapsed:.2f}s -> {actual_fps:.2f} FPS")
    print(f"Exported {export_count} frames to {bbox_config.IMG_DIR} and {bbox_config.VOC_DIR}")
    print(f"Sync timeouts: {timeout_count} ({100*timeout_count/frame_count:.1f}%)")
    print(f"{'='*60}")

except KeyboardInterrupt:
    print(f"\nInterrupted. Exported {export_count} frames before exit.")
    print(f"Sync timeouts: {timeout_count}")

finally:
    # Wait for all background writes to finish before exiting
    print("Waiting for background disk writes to complete...")
    io_executor.shutdown(wait=True)
    print("All disk writes finished.")

    for writer in [raw_writer, segmentation_writer, boxed_writer]:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print("CLEANUP")
    print(f"{'='*70}")
    print(f"Spawned walkers: {len(spawned_walkers)}")
    print(f"Spawned controllers: {len(spawned_controllers)}")
    print(f"Total cameras: {len(all_cameras)}")

    for ctrl in spawned_controllers:
        try:
            ctrl.stop()
        except:
            pass

    world.tick()

    all_sensors = []
    for i in range(len(all_cameras)):
        all_sensors.append(all_cameras[i])
        all_sensors.append(all_seg_cameras[i])

    bbox_carla.cleanup_actors(
        client, 
        world, 
        sensors=all_sensors,
        walkers=spawned_walkers,
        controllers=spawned_controllers
    )

    pygame.quit()
    print("Shutdown complete.")