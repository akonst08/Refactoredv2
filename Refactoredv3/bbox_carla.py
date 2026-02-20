import carla
import time

# Connect to CARLA simulator server
def connect_client(host="localhost", port=2000, timeout=10.0):
    client = carla.Client(host, port, worker_threads=4)
    client.set_timeout(timeout)
    return client

# Remove any leftover actors from previous runs
def destroy_leftovers(world, client):
    """Remove any leftover actors from previous runs."""
    actors = world.get_actors()
    
    controllers = list(actors.filter('controller.ai.walker'))
    walkers = list(actors.filter('walker.pedestrian.*'))
    sensors = list(actors.filter('sensor.*'))
    vehicles = list(actors.filter('vehicle.*'))
    
    destroyed = {'vehicle': 0, 'walker': 0, 'controller': 0, 'sensor': 0}
    
    # Destroy in order: controllers -> walkers -> sensors -> vehicles
    for ctrl in controllers:
        try:
            ctrl.destroy()
            destroyed['controller'] += 1
        except:
            pass
    for walker in walkers:
        try:
            walker.destroy()
            destroyed['walker'] += 1
        except:
            pass
    for sensor in sensors:
        try:
            sensor.destroy()
            destroyed['sensor'] += 1
        except:
            pass
    for vehicle in vehicles:
        try:
            vehicle.destroy()
            destroyed['vehicle'] += 1
        except:
            pass
    try:
        world.tick()
    except:
        pass
    
    total = sum(destroyed.values())
    if total > 0:
        print(f"Pre-run cleanup: {destroyed}")

# Enable synchronous mode for deterministic simulation
def setup_synchronous_mode(world, fixed_delta_seconds=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)
    print("Synchronous mode:", world.get_settings().synchronous_mode)

# Clean up all sensors and actors at the end of simulation
def cleanup_actors(client, world, sensors, walkers=None, controllers=None):
    """
    Clean up all sensors and actors at the end of simulation.
    
    Args:
        client: CARLA client
        world: CARLA world
        sensors: List of sensor actors to destroy
        walkers: List of spawned walker actors (optional)
        controllers: List of spawned controller actors (optional)
    """
    print("\nStarting cleanup...")
    
    destroyed = {'vehicle': 0, 'walker': 0, 'controller': 0, 'sensor': 0}
    
    # 1) Stop and destroy sensors
    if sensors:
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                    print(f"Stopped sensor: {sensor.type_id}")
                except Exception as e:
                    print(f"[WARN] Could not stop sensor: {e}")
        
        time.sleep(0.1)
        
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.destroy()
                    destroyed['sensor'] += 1
                    print(f"Destroyed sensor: {sensor.id}")
                except Exception as e:
                    print(f"[WARN] Could not destroy sensor: {e}")
    
    try:
        world.tick()
    except:
        pass
    
    # 2) Destroy controllers from spawn list
    if controllers:
        print(f"\nDestroying {len(controllers)} controllers...")
        for ctrl in controllers:
            try:
                ctrl.stop()
                ctrl.destroy()
                destroyed['controller'] += 1
            except Exception as e:
                print(f"[WARN] Controller {ctrl.id}: {e}")
    
    try:
        world.tick()
    except:
        pass
    
    # 3) Destroy walkers from spawn list
    if walkers:
        print(f"Destroying {len(walkers)} walkers...")
        for walker in walkers:
            try:
                walker.destroy()
                destroyed['walker'] += 1
                print(f"  Destroyed walker {walker.id}")
            except Exception as e:
                print(f"[WARN] Walker {walker.id}: {e}")
    
    try:
        world.tick()
    except:
        pass
    
    # 4) Destroy remaining actors from world
    print(f"\nDestroying remaining actors from world...")
    current_actors = world.get_actors()
    
    # Get remaining actors by type
    remaining_controllers = list(current_actors.filter('controller.ai.walker'))
    remaining_walkers = list(current_actors.filter('walker.pedestrian.*'))
    remaining_vehicles = list(current_actors.filter('vehicle.*'))
    
    print(f"  Remaining: controllers={len(remaining_controllers)}, walkers={len(remaining_walkers)}, vehicles={len(remaining_vehicles)}")
    
    # Destroy remaining controllers
    for ctrl in remaining_controllers:
        try:
            ctrl.stop()
            ctrl.destroy()
            destroyed['controller'] += 1
        except:
            pass
    
    try:
        world.tick()
    except:
        pass
    
    # Destroy remaining walkers
    for walker in remaining_walkers:
        try:
            walker.destroy()
            destroyed['walker'] += 1
            print(f"  Destroyed remaining walker {walker.id}")
        except:
            pass
    
    try:
        world.tick()
    except:
        pass
    
    # Destroy remaining vehicles
    for vehicle in remaining_vehicles:
        try:
            vehicle.destroy()
            destroyed['vehicle'] += 1
        except:
            pass
    
    try:
        world.tick()
    except:
        pass
    
    print(f"\nCleanup complete: {destroyed}")
    return destroyed
   # # Batch destroy remaining actors (controllers, walkers, vehicles)
    # destroy_cmds = []
    # for a in current_actors.filter('*controller.ai.walker*'):
    #     destroy_cmds.append(carla.command.DestroyActor(a.id))
    # for a in current_actors.filter('*walker.pedestrian*'):
    #     destroy_cmds.append(carla.command.DestroyActor(a.id))
    # for a in current_actors.filter('*vehicle*'):
    #     destroy_cmds.append(carla.command.DestroyActor(a.id))