import carla
import numpy as np
import math

# Callback to process RGB camera images
def cam_callback(image, data_dict):
    # Convert raw image data to numpy array (BGRA format)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    data_dict['image'] = arr.reshape((image.height, image.width, 4))
    data_dict['frame'] = image.frame

# Callback to process semantic segmentation images
def seg_callback(image, data_dict):
    # Capture raw class IDs before converting to a color palette
    arr_raw = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
    arr_raw = arr_raw.reshape((image.height, image.width, 4))
    
    data_dict['labels'] = arr_raw[:, :, 2] # Correct class IDs

    # DEBUG MESSAGE
    # print(data_dict['labels'])

    # Apply CityScapes color palette for semantic classes
    image.convert(carla.ColorConverter.CityScapesPalette)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
    data_dict['image'] = arr.reshape((image.height, image.width, 4))
    data_dict['frame'] = image.frame

# Create RGB and segmentation cameras with specified parameters
def create_cameras(world, bp_lib, cam_trans, fov, image_w, image_h):
    # Setup RGB cameral
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_w))
    camera_bp.set_attribute('image_size_y', str(image_h))
    camera_bp.set_attribute('fov', str(fov))
    camera = world.spawn_actor(camera_bp, cam_trans)
    
    # Setup semantic segmentation camera
    semantic_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    semantic_bp.set_attribute('image_size_x', str(image_w))
    semantic_bp.set_attribute('image_size_y', str(image_h))
    semantic_bp.set_attribute('fov', str(fov))
    segmentation_cam = world.spawn_actor(semantic_bp, cam_trans)
    
    # Initialize data storage for both cameras
    camera_data = {'image': np.zeros((image_h, image_w, 4), dtype=np.uint8), 'frame': -1}
    segmentation_data = {
        'image': np.zeros((image_h, image_w, 4), dtype=np.uint8),
        'labels': np.zeros((image_h, image_w), dtype=np.uint8),
        'frame': -1
    }
    
    # Register callbacks to capture images
    camera.listen(lambda image: cam_callback(image, camera_data))
    segmentation_cam.listen(lambda image: seg_callback(image, segmentation_data))
    
    return camera, segmentation_cam, camera_data, segmentation_data, camera_bp, semantic_bp


def spawn_three_cameras_at_spawn_point(world, bp_lib, spawn_point, cfg, spacing=15.0, lateral_offset=0.5):
    """
    Spawn 3 cameras that follow the road's natural curve using waypoints.
    
    Args:
        world: CARLA world
        bp_lib: Blueprint library
        spawn_point: carla.Transform (the base spawn point)
        cfg: Configuration dict
        spacing: Distance between cameras along the road (meters)
        lateral_offset: Small perpendicular offset for overlap (meters)
    
    Returns:
        List of camera_sets: [(camera, seg_cam, cam_data, seg_data, cam_bp, seg_bp), ...]
    """

    
    # Get the road waypoint at spawn point
    carla_map = world.get_map()
    center_wp = carla_map.get_waypoint(spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    if center_wp is None:
        print("[WARNING] Could not find waypoint at spawn point, using straight line fallback")
        # Fallback to straight line placement
        yaw = spawn_point.rotation.yaw
        rad = math.radians(yaw)
        forward = np.array([math.cos(rad), math.sin(rad)])
        base_pos = np.array([spawn_point.location.x, spawn_point.location.y])
        
        positions = [
            (base_pos - forward * spacing, yaw),
            (base_pos, yaw),
            (base_pos + forward * spacing, yaw)
        ]
    else:
        # Follow the road's natural curve using waypoints
        waypoints = []
        
        # Get waypoint behind (upstream)
        prev_wp = center_wp
        for _ in range(int(spacing / 2.0)):  # Walk backwards in 2m steps
            prev_list = prev_wp.previous(2.0)
            if not prev_list:
                break
            prev_wp = prev_list[0]
        waypoints.append(prev_wp)
        
        # Center waypoint
        waypoints.append(center_wp)
        
        # Get waypoint ahead (downstream)
        next_wp = center_wp
        for _ in range(int(spacing / 2.0)):  # Walk forward in 2m steps
            next_list = next_wp.next(2.0)
            if not next_list:
                break
            next_wp = next_list[0]
        waypoints.append(next_wp)
        
        # Extract positions and yaws from waypoints (following road curve)
        positions = []
        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            yaw = wp.transform.rotation.yaw
            
            # Add small lateral offset for FOV overlap
            rad = math.radians(yaw)
            lateral_vec = np.array([-math.sin(rad), math.cos(rad)])  # perpendicular
            offset = lateral_vec * (i - 1) * lateral_offset
            
            final_pos = np.array([loc.x, loc.y]) + offset
            positions.append((final_pos, yaw))
    
    # Spawn cameras at computed positions
    camera_sets = []
    
    for i, (pos, yaw) in enumerate(positions):
        # Create camera transform
        transform = carla.Transform(
            carla.Location(x=float(pos[0]), y=float(pos[1]), z=cfg['camera']['z']),
            carla.Rotation(pitch=cfg['camera']['pitch'], yaw=yaw, roll=0.0)
        )
        
        # Spawn camera pair (RGB + segmentation)
        cam_set = create_cameras(
            world, bp_lib, transform,
            cfg['camera']['fov'],
            cfg['camera']['width'],
            cfg['camera']['height']
        )
        
        camera_sets.append(cam_set)
        world.tick()
        
        print(f"[CAMERA {i}] Spawned at ({pos[0]:.1f}, {pos[1]:.1f}, {cfg['camera']['z']:.1f}) yaw={yaw:.1f}°")
    
    return camera_sets