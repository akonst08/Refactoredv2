import carla
import random
import numpy as np
import cv2
import math

# ---------- Camera Intrinsics Helper ----------
def get_camera_intrinsic(width, height, fov):
    focal = width / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

# ---------- Project 3D Bounding Box to 2D ----------
def get_2d_bbox(actor, K, world_2_camera_matrix):
    bb = actor.bounding_box
    verts = [carla.Location(x=x, y=y, z=z) for x in (-bb.extent.x, bb.extent.x)
             for y in (-bb.extent.y, bb.extent.y)
             for z in (-bb.extent.z, bb.extent.z)]
    pts = []
    for v in verts:
        v = bb.location + v
        v_world = actor.get_transform().transform(v)
        # Convert v_world to homogeneous coordinates
        v_world_hom = np.array([v_world.x, v_world.y, v_world.z, 1.0])
        # Apply the world-to-camera transformation matrix
        v_cam_hom = world_2_camera_matrix @ v_world_hom
        v_cam = carla.Location(v_cam_hom[0], v_cam_hom[1], v_cam_hom[2])

        if v_cam.z <= 0:
            continue
        p_img = K @ np.array([v_cam.x, -v_cam.y, v_cam.z])
        p_img = p_img[:2] / p_img[2]
        pts.append((int(p_img[0]), int(p_img[1])))

    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

# Camera callback (inside main function)
def process_image(image):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    img = img[:, :, :3]

    world_2_camera = camera.get_transform().get_inverse_matrix()
    actors = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*"))

    for actor in actors:
        if actor.id == ego.id:
            continue
        # Pass the world_2_camera matrix directly
        bbox = get_2d_bbox(actor, K, world_2_camera)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Ego Camera with BBoxes", img)
    if cv2.waitKey(1) == ord('q'):
        return
# ---------- Main ----------
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Ego vehicle
    ego_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    ego_transform = random.choice(world.get_map().get_spawn_points())
    ego = world.try_spawn_actor(ego_bp, ego_transform)
    if ego is None:
        print("Failed to spawn ego car.")
        return

    # Camera
    image_w, image_h, fov = 800, 600, 90
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(image_w))
    cam_bp.set_attribute("image_size_y", str(image_h))
    cam_bp.set_attribute("fov", str(fov))
    cam_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_init_trans, attach_to=ego)

    # Spawn traffic (vehicles + pedestrians + cyclists)
    blueprints = world.get_blueprint_library()
    for _ in range(10):
        v_bp = random.choice(blueprints.filter("vehicle.*"))
        v_spawn = random.choice(world.get_map().get_spawn_points())
        world.try_spawn_actor(v_bp, v_spawn)

    for _ in range(8):
        w_bp = random.choice(blueprints.filter("walker.pedestrian.*"))
        loc = world.get_random_location_from_navigation()
        if loc:
            world.try_spawn_actor(w_bp, carla.Transform(loc))

    for _ in range(3):
        c_bp = blueprints.find("vehicle.bh.crossbike")
        v_spawn = random.choice(world.get_map().get_spawn_points())
        world.try_spawn_actor(c_bp, v_spawn)

    # Camera calibration
    K = get_camera_intrinsic(image_w, image_h, fov)


    camera.listen(process_image)

    try:
        while True:
            world.tick()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        ego.destroy()
        cv2.destroyAllWindows()
        for actor in world.get_actors().filter("vehicle.*"):
            actor.destroy()
        for actor in world.get_actors().filter("walker.pedestrian.*"):
            actor.destroy()

if __name__ == "__main__":
    main()