import carla
import time

def connect_client(host="localhost", port=2000, timeout=10.0):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client

def destroy_leftovers(world, client):
    actors = world.get_actors()
    kill = []
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

def setup_synchronous_mode(world, fixed_delta_seconds=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    print("Synchronous mode:", world.get_settings().synchronous_mode)

def cleanup_actors(client, world, sensors):
    print("Starting cleanup...")
    
    for sensor in sensors:
        if sensor is not None:
            try:
                sensor.stop()
                print(f"Stopped sensor: {sensor.type_id}")
            except Exception as e:
                print(f"[WARN] Could not stop sensor: {e}")
    
    time.sleep(0.1)
    
    try:
        current_actors = world.get_actors()
        current_ids = {a.id for a in current_actors}
    except Exception:
        current_actors = []
        current_ids = set()
    
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
