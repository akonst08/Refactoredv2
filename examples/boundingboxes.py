import carla
import math
import random
import queue
import numpy as np
import pygame
from pygame.locals import *

# ----------------------------
# Helper functions
# ----------------------------
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1.0])
    point_cam = w2c @ point

    # CARLA -> camera coordinates -> screen coordinates
    x, y, z = point_cam[0], -point_cam[2], point_cam[1]
    if z <= 0:
        return None
    point_img = K @ np.array([x, y, z])
    point_img[:2] /= point_img[2]
    return point_img[:2]

# ----------------------------
# Main code
# ----------------------------
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicle
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle")

    # Spawn camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
    if camera is None:
        raise RuntimeError("Failed to spawn camera")

    vehicle.set_autopilot(True)

    # Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Image queue
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    # Camera intrinsics
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(image_w, image_h, fov)

    # Bounding boxes
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
    edges = [[0,1],[1,3],[3,2],[2,0],[0,4],[4,5],[5,1],[5,7],[7,6],[6,4],[6,2],[7,3]]

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((image_w, image_h))
    pygame.display.set_caption("CARLA RGB Camera")

    # Main loop
    running = True
    while running:
        world.tick()

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        if not image_queue.empty():
            image = image_queue.get()
            img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            img_rgb = img_bgra[:, :, :3]
            img_rgb = np.flip(img_rgb, axis=0)  # vertical flip
            img_rgb = img_rgb[:, :, ::-1]  # BGR -> RGB

            # World-to-camera matrix
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Draw bounding boxes
            for bb in bounding_box_set:
                if bb.location.distance(vehicle.get_transform().location) > 50:
                    continue
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                if forward_vec.dot(ray) <= 1:
                    continue
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                for edge in edges:
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                    if p1 is not None and p2 is not None:
                        x1, y1 = int(p1[0]), int(p1[1])
                        x2, y2 = int(p2[0]), int(p2[1])
                        if 0 <= x1 < image_w and 0 <= y1 < image_h and 0 <= x2 < image_w and 0 <= y2 < image_h:
                            cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 1)

            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0,1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

finally:
    print("Cleaning up...")
    if 'camera' in locals():
        camera.stop()
        camera.destroy()
    if 'vehicle' in locals():
        vehicle.destroy()
    if 'settings' in locals():
        settings.synchronous_mode = False
        world.apply_settings(settings)
    pygame.quit()
    print("Cleanup complete.")
