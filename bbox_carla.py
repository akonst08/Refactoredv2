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
    kill = []
    # Collect all actors to destroy (walkers, sensors, vehicles)
    kill += [a.id for a in actors.filter('controller.ai.walker')]
    kill += [a.id for a in actors.filter('walker.pedestrian.*')]
    kill += [a.id for a in actors.filter('sensor.*')]
    kill += [a.id for a in actors.filter('vehicle.*')]
    
    if kill:
        destroyed = 0
        for a in kill:
            try:
                a.destroy()
                destroyed += 1
            except Exception:
                pass
        try:
            world.tick()
        except Exception:
            pass
        print(f"Pre-run Destroyed {destroyed} leftover actors.")

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
    destroy_cmds = []
    for a in current_actors.filter('*controller*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
    for a in current_actors.filter('*walker*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))
    for a in current_actors.filter('*vehicle*'):
        destroy_cmds.append(carla.command.DestroyActor(a.id))

    if destroy_cmds:
        try:
            results = client.apply_batch_sync(destroy_cmds, do_tick=False)
            destroyed_count = {'vehicle': 0, 'walker': 0, 'controller': 0}
            for cmd, result in zip(destroy_cmds, results):
                if not result.error:
                    actor_type = ("controller" if "*controller*" in str(cmd) else
                                  "walker" if "*walker*" in str(cmd) else
                                  "vehicle")
                    destroyed_count[actor_type] += 1
            print(f"Batch cleanup: {destroyed_count} SUCCESSFULLY destroyed")
        except Exception as e:
            print(f"[WARN] Batch destroy failed: {e}")
            for cmd in destroy_cmds:
                try:
                    client.apply_batch_sync([cmd], False)
                except:
                    pass
    else:
        print("No actors to destroy")
    
    try:
        world.tick()
    except Exception:
        pass


def augment_spawn_points(spawn_points, variants=6,
                         forward_max=10.0,
                         lateral_max=3.0,
                         yaw_max=15.0):
    """
    Augment a list of spawn points by applying small random offsets.
    Each real spawn point generates `variants` extra positions.

    Args:
        spawn_points : list of carla.Transform  (from world.get_map().get_spawn_points())
        variants     : number of augmented variants per real spawn point
        forward_max  : max forward/backward offset in meters
        lateral_max  : max left/right offset in meters
        yaw_max      : max yaw rotation offset in degrees

    Returns:
        list of carla.Transform  (original + augmented, shuffled)
    """
    augmented = list(spawn_points)  # keep originals

    for sp in spawn_points:
        yaw_rad = math.radians(sp.rotation.yaw)

        # Unit vectors along and perpendicular to the road
        forward = ( math.cos(yaw_rad),  math.sin(yaw_rad))
        lateral = (-math.sin(yaw_rad),  math.cos(yaw_rad))

        for _ in range(variants):
            # Random offsets
            fwd_off = random.uniform(-forward_max, forward_max)
            lat_off = random.uniform(-lateral_max, lateral_max)
            yaw_off = random.uniform(-yaw_max,     yaw_max)

            new_x = sp.location.x + forward[0] * fwd_off + lateral[0] * lat_off
            new_y = sp.location.y + forward[1] * fwd_off + lateral[1] * lat_off
            new_z = sp.location.z  # keep original height

            new_transform = carla.Transform(
                carla.Location(x=new_x, y=new_y, z=new_z),
                carla.Rotation(
                    pitch=sp.rotation.pitch,
                    yaw=sp.rotation.yaw + yaw_off,
                    roll=sp.rotation.roll
                )
            )
            augmented.append(new_transform)

    random.shuffle(augmented)
    print(f"[SPAWN POINTS] {len(spawn_points)} real → {len(augmented)} total (×{variants+1})")
    return augmented