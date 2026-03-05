# CARLA 0.9.15 — Scene Configuration & Camera Setup

---

## Configuration — `config.yaml`

Every scene parameter is controlled from a single file.

```yaml
carla:
  host: localhost            # CARLA server IP — keep localhost if running on same machine
  port: 2000                 # Default CARLA port - do not change
  timeout: 10.0              # Seconds to wait for server connection
  synchronous: true          # Deterministic frame-by-frame control 
  fixed_delta_seconds: 0.05  # Simulation timestep  20 FPS
  weather: ClearSunset       # Starting weather preset (see Weather section below)

camera:
  width: 1280                # Output image width in pixels
  height: 720                # Output image height in pixels
  fov: 80                    # Field of view in degrees
  z: 50.0                    # Camera height above ground in metres
  pitch: -90.0               # -90 = straight down (bird's eye)

run:
  duration: 250               #  Simulation length in seconds
  cam_index: 45              #  Which spawn point to centre the camera over (Initially)

traffic:
  vehicles: 15               # NPC vehicles spawned with autopilot (Set in the pipeline)
  walkers: 10                # Pedestrian walkers spawned with AI controllers

processing:
  process_interval: 2        # Process every Nth tick
  enable_static_detection: true  # Detect map-embedded parked vehicles

export:
  enable: true               # Master switch for saving frames
  export_interval: 4         # Save every Nth processed frame
  export_start_frame: 30     # Skip first N frames — lets traffic settle
  export_end_percent: 0.95   # Stop at 95% of run duration
  spawn_change_interval: 8   # Move camera every N exported frames
  weather_change_interval: 4 # Change weather every N exported frames
  max_exports: 1300           # Hard cap on total exported frames
```

## Camera Placement — `cam_index`

The camera is placed at the **XY coordinates of a road spawn point**, elevated to
`z` metres above ground with `pitch = -90` (straight down bird's eye view).

### Initial placement

```python
# Get all road spawn points from the map
spawn_points = world.get_map().get_spawn_points()

# Augment with offset variants for more coverage positions
spawn_points = bbox_carla.augment_spawn_points(
    spawn_points,
    variants     = SPAWN_AUG_VARIANTS,      # 6 offset variants per real point
    forward_max  = SPAWN_AUG_FORWARD_MAX,   # ±20m along road
    lateral_max  = SPAWN_AUG_LATERAL_MAX,   # ±15m sideways
    yaw_max      = SPAWN_AUG_YAW_MAX        # ±30° rotation
)

# Pick spawn point from config
base_sp = spawn_points[cam_index]   # e.g. cam_index = 45

# Build camera transform:
# - XY from the spawn point (road position)
# - Z from config  (altitude above ground)
# - pitch = -90    (straight down)
cam_trans = carla.Transform(
    carla.Location(
        x = base_sp.location.x,
        y = base_sp.location.y,
        z = cfg["camera"]["z"]           # 50.0m
    ),
    carla.Rotation(
        pitch = cfg["camera"]["pitch"],  # -90.0
        yaw   = 0.0,
        roll  = 0.0
    )
)
```

---

## Dynamic Camera Repositioning (Runtime)

During a live run the camera can be moved **without stopping the pipeline** —
traffic, export counters, and all state remain untouched.

| Key | Action |
|-----|--------|
| `C` | Move cameras to a random new spawn point |
| `[` | Switch to previous weather preset |
| `]` | Switch to next weather preset |
| `Q` | Quit cleanly |

```python
# Press C  stop old cameras, pick a new random position, respawn
new_sp = random.choice(spawn_points)
new_cam_trans = carla.Transform(
    carla.Location(
        x = new_sp.location.x,
        y = new_sp.location.y,
        z = cfg["camera"]["z"]           # same altitude
    ),
    carla.Rotation(
        pitch = cfg["camera"]["pitch"],  # same angle
        yaw   = 0.0,
        roll  = 0.0
    )
)
# Respawn both RGB and segmentation cameras at the new position
camera, segmentation_cam, ... = bbox_camera.create_cameras(
    world, bp_lib, new_cam_trans, cfg["camera"]["fov"], image_w, image_h
)
```

---

## Weather Presets

| Value | Description |
|-------|-------------|
| `ClearNoon` | Bright midday, no clouds |
| `ClearSunset` | Golden hour, long shadows |
| `CloudyNoon` | Overcast midday |
| `WetNoon` | Wet roads, midday |
| `HardRainNoon` | Heavy rain, midday |
| `SoftRainNoon` | Light rain, midday |
| `MidRainyNoon` | Moderate rain, midday |
| `WetSunset` | Wet roads, sunset |
| `HardRainSunset` | Heavy rain, sunset |
| `SoftRainSunset` | Light rain, sunset |
