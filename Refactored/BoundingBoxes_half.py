import carla
import random
import time
import numpy as np
import pygame
import cv2
import os
import time as time_module

#  MODULE IMPORTS 
# Import refactored modules for configuration, CARLA management, 
# camera handling, detection algorithms, and label export
import bbox_config
import bbox_carla
import bbox_camera
import bbox_detection
import bbox_labels

#  INITIALIZATION 
# Parse command-line arguments (FOV, duration, camera spawn index)
args = bbox_config.parse_args()
# Create fresh output directories for images, labels, and videos
bbox_config.setup_output_dirs()

# Initialize Pygame for visualization window
pygame.init()

#  CARLA CONNECTION & WORLD SETUP 
# Connect to CARLA simulator and get the world instance
client = bbox_carla.connect_client()
world = client.get_world()
# Set initial weather conditions
world.set_weather(carla.WeatherParameters.ClearNoon)
# Clean up any actors from previous runs
bbox_carla.destroy_leftovers(world, client)
# Enable synchronous mode for deterministic simulation (20 FPS = 0.05s per tick)
bbox_carla.setup_synchronous_mode(world, fixed_delta_seconds=0.05)

#  CAMERA SPAWNING 
# Get blueprint library and available spawn points from the map
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# Validate the requested camera spawn index
if args.cam_index >= len(spawn_points):
    raise ValueError(f"--cam-index {args.cam_index} out of range (map has {len(spawn_points)} spawn points)")

# Position camera at specified spawn point but elevated (z=60) with top-down view (pitch=-90)
base_sp = spawn_points[args.cam_index]
cam_trans = carla.Transform(carla.Location(x=base_sp.location.x, y=base_sp.location.y, z=60.0),
                             carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))

# Create RGB and semantic segmentation cameras with specified resolution and FOV
image_w, image_h = 1280, 720
camera, segmentation_cam, camera_data, segmentation_data, camera_bp, semantic_bp = bbox_camera.create_cameras(
    world, bp_lib, cam_trans, args.fov, image_w, image_h
)

# Create Pygame window for live visualization
screen = pygame.display.set_mode((image_w, image_h))
pygame.display.set_caption("CARLA view")

#  VIDEO WRITERS SETUP 
# Initialize video writers for raw, annotated, and segmentation outputs
# FPS = 20
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# raw_writer = cv2.VideoWriter(os.path.join(bbox_config.VID_DIR, "raw.mp4"), fourcc, FPS, (image_w, image_h))
# boxed_writer = cv2.VideoWriter(os.path.join(bbox_config.VID_DIR, "boxed.mp4"), fourcc, FPS, (image_w, image_h))
# segmentation_writer = cv2.VideoWriter(os.path.join(bbox_config.VID_DIR, "segmented.mp4"), fourcc, FPS, (image_w, image_h))
raw_writer = None
boxed_writer = None
segmentation_writer = None

#  PROJECTION MATRIX 
# Build camera intrinsic matrix for 3D to 2D projection
K = bbox_detection.build_projection_matrix(image_w, image_h, args.fov)

#  TRAFFIC SPAWNING 
# Spawn vehicles with autopilot for dynamic scene
vehicle_bps = bp_lib.filter('*vehicle*')
for _ in range(10):
    sp = random.choice(world.get_map().get_spawn_points())
    vbp = random.choice(vehicle_bps)
    spawned_vehicle = world.try_spawn_actor(vbp, sp)
    if spawned_vehicle:
        spawned_vehicle.set_autopilot(True)

# Spawn pedestrian walkers with AI controllers for realistic movement
walker_bps = bp_lib.filter('*walker*')
for _ in range(3):
    sp = carla.Transform()
    sp.location = world.get_random_location_from_navigation()
    if sp.location:
        wbp = random.choice(walker_bps)
        walker = world.try_spawn_actor(wbp, sp)
        if walker:
            # Attach AI controller to make the walker move autonomously
            ctrl_bp = bp_lib.find('controller.ai.walker')
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1 + random.random())

# Spawn a stationary "dummy" vehicle to keep traffic manager active
vehicle_bp = bp_lib.filter('*mini*')[0]
dummy_sp = spawn_points[0]
dummy_vehicle = world.try_spawn_actor(vehicle_bp, dummy_sp)
if dummy_vehicle:
    dummy_vehicle.set_autopilot(False)
    dummy_vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))

#  PERFORMANCE & EXPORT SETTINGS 
# Process detection every N frames to reduce computational load
PROCESS_INTERVAL = 2
# Enable detection of static vehicles from segmentation masks
ENABLE_STATIC_DETECTION = True
# Enable exporting annotated frames to disk
ENABLE_EXPORTS = True
# Export every Nth frame within the export window
EXPORT_INTERVAL = 10
# Start exporting after initial frames to allow scene stabilization
EXPORT_START_FRAME = 50
# Stop exporting before simulation ends (95% of total frames)
EXPORT_END_PERCENT = 0.95
# Maximum number of frames to export
MAX_EXPORTS = 100


#  STATE TRACKING VARIABLES 
# Counter for exported frames
export_count = 0
# Counter for frame synchronization timeouts
timeout_count = 0
# Last frame where FPS was reported
last_fps_report = 0
# Cached detection results to avoid recomputing every frame
cached_filtered = []
cached_static = []
# Frame numbers when cache was last updated
last_filter_frame = -1
last_static_frame = -1

# Timing and weather control
start_time = time.time()
frame_count = 0
weather_idx = 0

try:
    #  MAIN SIMULATION LOOP 
    while True:
        # Calculate target frame count based on requested duration
        target_frames = int(args.duration / world.get_settings().fixed_delta_seconds)
        EXPORT_END_FRAME = int(target_frames * EXPORT_END_PERCENT)
        
        # Check if simulation duration has been reached
        if frame_count >= target_frames:
            print(f"Finished {args.duration}s capture ({frame_count} frames, {export_count} exported).")
            break
        
        # Advance simulation by one tick
        world.tick()
        
        #  CAMERA FRAME SYNCHRONIZATION 
        # Wait for both cameras to deliver frames for the current simulation tick
        max_wait = 50
        synced = False
        for wait_iter in range(max_wait):
            if (camera_data['frame'] >= frame_count and segmentation_data['frame'] >= frame_count):
                synced = True
                break
            time.sleep(0.001)
        
        # Skip frame if synchronization failed
        if not synced:
            timeout_count += 1
            if timeout_count % 10 == 0:
                print(f"[WARN] Frame sync timeout #{timeout_count} at frame {frame_count}")
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

        t_start = time_module.perf_counter()

        #  DYNAMIC VEHICLE DETECTION 
        # Project 3D bounding boxes of all vehicles onto 2D image plane
        boxes_xyxy_cls = []
        actors = world.get_actors()
        for a in actors.filter('*vehicle*'):
            bb = a.bounding_box
            # Transform bounding box vertices from world to camera coordinates
            verts = [bbox_detection.get_image_point(v, K, w2c) for v in bb.get_world_vertices(a.get_transform())]
            # Compute 2D bounding box and validate it's within image bounds
            mm = bbox_detection.finite_bbox(verts, image_w, image_h)
            if mm:
                x_min, x_max, y_min, y_max = mm
                # Get semantic tag (class ID) or default to 'car' (14)
                cid = a.semantic_tags[0] if a.semantic_tags else 14
                boxes_xyxy_cls.append((x_min, y_min, x_max, y_max, cid))
        
        #  EXPORT DECISION 
        # Determine if this frame should be exported based on settings
        is_export_frame = (ENABLE_EXPORTS and 
                          frame_count >= EXPORT_START_FRAME and 
                          frame_count <= EXPORT_END_FRAME and
                          frame_count % EXPORT_INTERVAL == 0 and
                          export_count < MAX_EXPORTS)
        
        t_projection = time_module.perf_counter()

        #  FILTERING & STATIC DETECTION 
        # Only reprocess detection if interval reached or exporting this frame
        should_process = (frame_count % PROCESS_INTERVAL == 0) or is_export_frame
        
        if should_process:
            # Filter boxes using segmentation to remove false positives (background regions)
            cached_filtered = bbox_detection.filter_boxes_segmentation(boxes_xyxy_cls, seg_img)
            last_filter_frame = frame_count
            t_filter = time_module.perf_counter()
            if ENABLE_STATIC_DETECTION:
                # Detect static vehicles from segmentation that aren't already tracked
                seg_bgr = seg_img[:, :, :3]
                cached_static = bbox_detection.detect_static_vehicles(seg_bgr, cached_filtered, bbox_config.SEG_COLORS)
                last_static_frame = frame_count
                t_static = time_module.perf_counter()
            else:
                cached_static = []

        # Combine dynamic (filtered) and static detections
        boxes_xyxy_cls = cached_filtered + cached_static
        
        #  FRAME EXPORT 
        # Save frame and annotations if this is an export frame
        if is_export_frame:
            frame_id = f"{frame_count:06d}"
            img_bgr = img[:, :, :3]
            
            # Save RGB image to disk
            img_path = os.path.join(bbox_config.IMG_DIR, f"frame_{frame_id}.png")
            cv2.imwrite(img_path, img_bgr)
            
            # Export bounding box annotations in Pascal VOC XML format
            voc_path = os.path.join(bbox_config.VOC_DIR, f"frame_{frame_id}.xml")
            bbox_labels.write_voc_xml(voc_path, boxes_xyxy_cls, image_w, image_h, img_path, bbox_config.CLASS_NAMES)
            
            export_count += 1
            print(f"[EXPORT {export_count}/{MAX_EXPORTS}] Frame {frame_count}: {len(boxes_xyxy_cls)} boxes")
        
        frame_id = f"{frame_count:06d}"
        img_bgr = img[:, :, :3]
        
        # Optional: Write raw video (currently disabled)
        #raw_writer.write(img_bgr)
        
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
        
        t_video = time_module.perf_counter()

        #  PYGAME VISUALIZATION 
        # Display current frame in Pygame window
        rgb = img_bgr[:, :, ::-1]
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        t_display = time_module.perf_counter()

        #  PERFORMANCE REPORTING 
        # Periodically report FPS and statistics
        if frame_count - last_fps_report >= 100:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[FPS] Frame {frame_count}: {current_fps:.1f} FPS | Timeouts: {timeout_count} | Boxes: {len(boxes_xyxy_cls)}")
            last_fps_report = frame_count

        # Detailed timing breakdown every 100 frames
        # Detailed timing breakdown every 100 frames
        if frame_count % 100 == 0:
            print(f"Timing breakdown (ms):")
            print(f"  Projection: {(t_projection - t_start)*1000:.1f}")
            print(f"  Filtering:  {(t_filter - t_projection)*1000:.1f}")
            print(f"  Static:     {(t_static - t_filter)*1000:.1f}")
            print(f"  Video:      {(t_video - t_static)*1000:.1f}")
            print(f"  Display:    {(t_display - t_video)*1000:.1f}")
            print(f"  TOTAL:      {(t_display - t_start)*1000:.1f}")

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
            
            # Clear detection cache when camera moves
            cached_filtered = []
            cached_static = []
            last_filter_frame = -1
            last_static_frame = -1
            
            # Stop and destroy old cameras
            try:
                camera.stop()
                segmentation_cam.stop()
            except Exception:
                pass
            
            time.sleep(0.05)
            
            try:
                camera.destroy()
                segmentation_cam.destroy()
            except Exception:
                pass
            
            world.tick()

            # Create new cameras at random location
            new_sp = random.choice(spawn_points)
            new_cam_trans = carla.Transform(
                carla.Location(x=new_sp.location.x, y=new_sp.location.y, z=60.0),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            )

            camera, segmentation_cam, camera_data, segmentation_data, _, _ = bbox_camera.create_cameras(
                world, bp_lib, new_cam_trans, args.fov, image_w, image_h
            )

            world.tick()
            time.sleep(0.05)
            
            # Reset cache frames to trigger reprocessing
            last_filter_frame = frame_count - PROCESS_INTERVAL
            last_static_frame = frame_count - PROCESS_INTERVAL
            
            print(f"Cameras moved to spawn {spawn_points.index(new_sp)}")

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
    bbox_carla.cleanup_actors(client, world, sensors=[camera, segmentation_cam])
    # Close Pygame window
    pygame.quit()
    print("Shutdown complete.")
