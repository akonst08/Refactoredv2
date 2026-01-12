import carla
import random
import time
import numpy as np
import pygame
import cv2
import os
import time as time_module

# Import refactored modules
import bbox_config
import bbox_carla
import bbox_camera
import bbox_detection
import bbox_labels

# Parse args
args = bbox_config.parse_args()
bbox_config.setup_output_dirs()

# Pygame init
pygame.init()

# CARLA setup
client = bbox_carla.connect_client()
world = client.get_world()
world.set_weather(carla.WeatherParameters.ClearNoon)
bbox_carla.destroy_leftovers(world, client)
bbox_carla.setup_synchronous_mode(world, fixed_delta_seconds=0.05)

# Spawn cameras
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

if args.cam_index >= len(spawn_points):
    raise ValueError(f"--cam-index {args.cam_index} out of range (map has {len(spawn_points)} spawn points)")

base_sp = spawn_points[args.cam_index]
cam_trans = carla.Transform(carla.Location(x=base_sp.location.x, y=base_sp.location.y, z=60.0),
                             carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0))

image_w, image_h = 1280, 720
camera, segmentation_cam, camera_data, segmentation_data, camera_bp, semantic_bp = bbox_camera.create_cameras(
    world, bp_lib, cam_trans, args.fov, image_w, image_h
)

screen = pygame.display.set_mode((image_w, image_h))
pygame.display.set_caption("CARLA view")

# Video writers
FPS = 20
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
raw_writer = cv2.VideoWriter(os.path.join(bbox_config.VID_DIR, "raw.mp4"), fourcc, FPS, (image_w, image_h))
boxed_writer = cv2.VideoWriter(os.path.join(bbox_config.VID_DIR, "boxed.mp4"), fourcc, FPS, (image_w, image_h))
segmentation_writer = cv2.VideoWriter(os.path.join(bbox_config.VID_DIR, "segmented.mp4"), fourcc, FPS, (image_w, image_h))

# Projection matrix
K = bbox_detection.build_projection_matrix(image_w, image_h, args.fov)

# Spawn traffic
vehicle_bps = bp_lib.filter('*vehicle*')
for _ in range(30):
    sp = random.choice(world.get_map().get_spawn_points())
    vbp = random.choice(vehicle_bps)
    spawned_vehicle = world.try_spawn_actor(vbp, sp)
    if spawned_vehicle:
        spawned_vehicle.set_autopilot(True)

walker_bps = bp_lib.filter('*walker*')
for _ in range(5):
    sp = carla.Transform()
    sp.location = world.get_random_location_from_navigation()
    if sp.location:
        wbp = random.choice(walker_bps)
        walker = world.try_spawn_actor(wbp, sp)
        if walker:
            ctrl_bp = bp_lib.find('controller.ai.walker')
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1 + random.random())

# Dummy vehicle for stable traffic
vehicle_bp = bp_lib.filter('*mini*')[0]
dummy_sp = spawn_points[0]
dummy_vehicle = world.try_spawn_actor(vehicle_bp, dummy_sp)
if dummy_vehicle:
    dummy_vehicle.set_autopilot(False)
    dummy_vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))

# Performance settings
PROCESS_INTERVAL = 2
ENABLE_STATIC_DETECTION = True
ENABLE_EXPORTS = True
EXPORT_INTERVAL = 10
EXPORT_START_FRAME = 50
EXPORT_END_PERCENT = 0.95
MAX_EXPORTS = 100


export_count = 0
timeout_count = 0
last_fps_report = 0
cached_filtered = []
cached_static = []
last_filter_frame = -1
last_static_frame = -1

start_time = time.time()
frame_count = 0
weather_idx = 0

try:
    while True:
        target_frames = int(args.duration / world.get_settings().fixed_delta_seconds)
        EXPORT_END_FRAME = int(target_frames * EXPORT_END_PERCENT)
        
        if frame_count >= target_frames:
            print(f"Finished {args.duration}s capture ({frame_count} frames, {export_count} exported).")
            break
        
        world.tick()
        
        max_wait = 50
        synced = False
        for wait_iter in range(max_wait):
            if (camera_data['frame'] >= frame_count and segmentation_data['frame'] >= frame_count):
                synced = True
                break
            time.sleep(0.001)
        
        if not synced:
            timeout_count += 1
            if timeout_count % 10 == 0:
                print(f"[WARN] Frame sync timeout #{timeout_count} at frame {frame_count}")
            frame_count += 1
            continue
        
        img = camera_data['image']
        seg_img = segmentation_data['image']
        
        if img is None or img.size == 0 or seg_img is None or seg_img.size == 0:
            frame_count += 1
            continue

        cam_tf = camera.get_transform()
        try:
            w2c = np.array(cam_tf.get_inverse_matrix())
        except AttributeError:
            w2c = np.linalg.inv(np.array(cam_tf.get_matrix(), dtype=float))

        t_start = time_module.perf_counter()

        boxes_xyxy_cls = []
        actors = world.get_actors()
        for a in actors.filter('*vehicle*'):
            bb = a.bounding_box
            verts = [bbox_detection.get_image_point(v, K, w2c) for v in bb.get_world_vertices(a.get_transform())]
            mm = bbox_detection.finite_bbox(verts, image_w, image_h)
            if mm:
                x_min, x_max, y_min, y_max = mm
                cid = a.semantic_tags[0] if a.semantic_tags else 14
                boxes_xyxy_cls.append((x_min, y_min, x_max, y_max, cid))
        
        is_export_frame = (ENABLE_EXPORTS and 
                          frame_count >= EXPORT_START_FRAME and 
                          frame_count <= EXPORT_END_FRAME and
                          frame_count % EXPORT_INTERVAL == 0 and
                          export_count < MAX_EXPORTS)
        
        t_projection = time_module.perf_counter()

        should_process = (frame_count % PROCESS_INTERVAL == 0) or is_export_frame
        
        if should_process:
            cached_filtered = bbox_detection.filter_boxes_segmentation(boxes_xyxy_cls, seg_img)
            last_filter_frame = frame_count
            t_filter = time_module.perf_counter()
            if ENABLE_STATIC_DETECTION:
                seg_bgr = seg_img[:, :, :3]
                cached_static = bbox_detection.detect_static_vehicles(seg_bgr, cached_filtered, bbox_config.SEG_COLORS)
                last_static_frame = frame_count
                t_static = time_module.perf_counter()
            else:
                cached_static = []

        boxes_xyxy_cls = cached_filtered + cached_static
        
        if is_export_frame:
            frame_id = f"{frame_count:06d}"
            img_bgr = img[:, :, :3]
            
            img_path = os.path.join(bbox_config.IMG_DIR, f"frame_{frame_id}.png")
            cv2.imwrite(img_path, img_bgr)
            
            voc_path = os.path.join(bbox_config.VOC_DIR, f"frame_{frame_id}.xml")
            bbox_labels.write_voc_xml(voc_path, boxes_xyxy_cls, image_w, image_h, img_path, bbox_config.CLASS_NAMES)
            
            export_count += 1
            print(f"[EXPORT {export_count}/{MAX_EXPORTS}] Frame {frame_count}: {len(boxes_xyxy_cls)} boxes")
        
        frame_id = f"{frame_count:06d}"
        img_bgr = img[:, :, :3]
        
        #raw_writer.write(img_bgr)
        
        overlay = img_bgr.copy()
        for (x1, y1, x2, y2, cid) in boxes_xyxy_cls:
            color = (0, 255, 0) if cid in [13,14,15,16,17,18,19] else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        boxed_writer.write(overlay)
        
        seg_bgr = seg_img[:, :, :3]
        #segmentation_writer.write(seg_bgr)
        
        t_video = time_module.perf_counter()

        rgb = img_bgr[:, :, ::-1]
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        t_display = time_module.perf_counter()

        if frame_count - last_fps_report >= 100:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[FPS] Frame {frame_count}: {current_fps:.1f} FPS | Timeouts: {timeout_count} | Boxes: {len(boxes_xyxy_cls)}")
            last_fps_report = frame_count

        if frame_count % 100 == 0:
            print(f"Timing breakdown (ms):")
            print(f"  Projection: {(t_projection - t_start)*1000:.1f}")
            print(f"  Filtering:  {(t_filter - t_projection)*1000:.1f}")
            print(f"  Static:     {(t_static - t_filter)*1000:.1f}")
            print(f"  Video:      {(t_video - t_static)*1000:.1f}")
            print(f"  Display:    {(t_display - t_video)*1000:.1f}")
            print(f"  TOTAL:      {(t_display - t_start)*1000:.1f}")

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

        if keys[pygame.K_LEFTBRACKET]:
            weather_idx = (weather_idx - 1) % len(bbox_config.weather_presets)
            world.set_weather(bbox_config.weather_presets[weather_idx])
        elif keys[pygame.K_RIGHTBRACKET]:
            weather_idx = (weather_idx + 1) % len(bbox_config.weather_presets)
            world.set_weather(bbox_config.weather_presets[weather_idx])
        
        if keys[pygame.K_c]:
            print("Repositioning cameras...")
            
            cached_filtered = []
            cached_static = []
            last_filter_frame = -1
            last_static_frame = -1
            
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
            
            last_filter_frame = frame_count - PROCESS_INTERVAL
            last_static_frame = frame_count - PROCESS_INTERVAL
            
            print(f"Cameras moved to spawn {spawn_points.index(new_sp)}")

        frame_count += 1

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
    for writer in [raw_writer, segmentation_writer, boxed_writer]:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
    
    bbox_carla.cleanup_actors(client, world, sensors=[camera, segmentation_cam])
    pygame.quit()
    print("Shutdown complete.")
