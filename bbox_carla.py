import carla
import time
import math
import random
# Connect to CARLA simulator server
def connect_client(host="localhost", port=2000, timeout=10.0):
    client = carla.Client(host, port, worker_threads=4)
    client.set_timeout(timeout)
    return client

# Remove any leftover actors from previous runs
def destroy_leftovers(world, client):
    actors = world.get_actors()
    destroy_cmds = []
    
    for a in actors.filter('controller.ai.walker'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
    for a in actors.filter('walker.*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
    for a in actors.filter('sensor.*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
    for a in actors.filter('vehicle.*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))

    if destroy_cmds:
        results = client.apply_batch_sync(destroy_cmds, True)
        destroyed = sum(1 for r in results if not r.error)
        print(f"Pre-run cleanup: destroyed {destroyed} leftover actors.")
    else:
        print("Pre-run cleanup: nothing to destroy.")

# Enable synchronous mode for deterministic simulation
def setup_synchronous_mode(world, fixed_delta_seconds=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    print("Synchronous mode:", world.get_settings().synchronous_mode)

# Clean up all sensors and actors at the end of simulation
def cleanup_actors(client, world, sensors):
    print("Starting cleanup...")
    
    # Stop all sensors first
    for sensor in sensors:
        if sensor is not None:
            try:
                sensor.stop()
                print(f"Stopped sensor: {sensor.type_id}")
            except Exception as e:
                print(f"[WARN] Could not stop sensor: {e}")
    
    time.sleep(0.1)
    
    # Get current actors in the world
    try:
        current_actors = world.get_actors()
        current_ids = {a.id for a in current_actors}
    except Exception:
        current_actors = []
        current_ids = set()
    
    # Destroy sensor actors
    for sensor in sensors:
        if sensor is not None and sensor.id in current_ids:
            try:
                sensor.destroy()
                print(f"Destroyed sensor: {sensor.id}")
                current_ids.remove(sensor.id)
            except Exception as e:
                print(f"[WARN] Could not destroy sensor {sensor.id}: {e}")
    
    try:
        world.tick()
    except Exception:
        pass
    
    # Batch destroy remaining actors (controllers, walkers, vehicles)
    # Batch destroy remaining actors with parallel type tracking
    destroy_cmds = []
    actor_types  = []

    for a in current_actors.filter('controller.ai.walker'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
        actor_types.append('controller')
    for a in current_actors.filter('walker.pedestrian.*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
        actor_types.append('walker')
    for a in current_actors.filter('vehicle.*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
        actor_types.append('vehicle')

    if destroy_cmds:
        try:
            results = client.apply_batch_sync(destroy_cmds, do_tick=False)
            destroyed_count = {'vehicle': 0, 'walker': 0, 'controller': 0}
            for atype, result in zip(actor_types, results):
                if not result.error:
                    destroyed_count[atype] += 1
            print(f"Batch cleanup: {destroyed_count} SUCCESSFULLY destroyed")
        except Exception as e:
            print(f"[WARN] Batch destroy failed: {e}")
            for cmd in destroy_cmds:
                try:
                    client.apply_batch_sync([cmd], False)
                except Exception:
                    pass
    else:
        print("No actors to destroy")
    
    try:
        world.tick()
    except Exception:
        pass


def augment_spawn_points(spawn_points, world, variants=6, forward_max=20.0, lateral_max=15.0, yaw_max=30.0):
    """
    Augment spawn points by snapping candidates to nearest road waypoint.
    This prevents augmented positions going out-of-bounds or off the road.
    Camera z is always added on top of the waypoint ground z in BoundingBoxes_half.py.
    """
    aug = []
    for sp in spawn_points:
        aug.append(sp)  # always keep original
        try:
            wp = world.get_map().get_waypoint(sp.location)
        except Exception:
            wp = None

        for _ in range(variants):
            fwd        = random.uniform(-forward_max, forward_max)
            lat        = random.uniform(-lateral_max, lateral_max)
            yaw_offset = random.uniform(-yaw_max, yaw_max)

            if wp is not None:
                wloc        = wp.transform.location
                forward_vec = wp.transform.get_forward_vector()
                right_vec   = wp.transform.get_right_vector()
                cand_x = wloc.x + forward_vec.x * fwd + right_vec.x * lat
                cand_y = wloc.y + forward_vec.y * fwd + right_vec.y * lat
                try:
                    # snap to nearest road waypoint — guarantees on-road position
                    cand_wp  = world.get_map().get_waypoint(
                        carla.Location(x=cand_x, y=cand_y, z=wloc.z)
                    )
                    cand_rot     = cand_wp.transform.rotation
                    cand_rot.yaw += yaw_offset
                    # use waypoint ground z — camera altitude added in BoundingBoxes_half.py
                    aug.append(carla.Transform(cand_wp.transform.location, cand_rot))
                except Exception:
                    aug.append(sp)  # fallback to original
            else:
                # no waypoint found — simple offset fallback
                aug.append(carla.Transform(
                    carla.Location(
                        x = sp.location.x + fwd,
                        y = sp.location.y + lat,
                        z = sp.location.z          # keep ground z
                    ),
                    carla.Rotation(
                        pitch = sp.rotation.pitch,
                        yaw   = sp.rotation.yaw + yaw_offset,
                        roll  = sp.rotation.roll
                    )
                ))
    return aug

def safe_destroy_cameras(camera, segmentation_cam, all_sensors):
    """
    Safely stop and destroy a camera pair and remove them from all_sensors.
    Prevents 'sensor out of scope' warnings and core dumps.
    """
    dead_ids = set()
    for cam in [camera, segmentation_cam]:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            dead_ids.add(cam.id)
            cam.destroy()
        except Exception:
            pass
    # Remove destroyed sensors from tracking list
    return [s for s in all_sensors if s.id not in dead_ids]