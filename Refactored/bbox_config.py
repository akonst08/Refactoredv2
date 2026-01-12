import argparse
import os
import shutil
import carla

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal camera FOV in degrees")
    parser.add_argument("--duration", type=float, default=25.0, help="Capture duration in seconds")
    parser.add_argument("--cam-index", type=int, default=0, help="Spawn point index for static camera position")
    return parser.parse_args()

# Output directories
OUT_ROOT = "out"
IMG_DIR = os.path.join(OUT_ROOT, "images")
YOLO_DIR = os.path.join(OUT_ROOT, "labels_yolo")
VID_DIR = os.path.join(OUT_ROOT, "videos")
VOC_DIR = os.path.join(OUT_ROOT, "annotations_voc")

def setup_output_dirs():
    for d in [IMG_DIR, YOLO_DIR, VID_DIR, VOC_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

# Weather presets
weather_presets = [
    carla.WeatherParameters.Default,
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.SoftRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.WetSunset,
    carla.WeatherParameters.WetCloudySunset,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainSunset,
    carla.WeatherParameters.SoftRainSunset,
    carla.WeatherParameters.ClearNight,
    carla.WeatherParameters.CloudyNight,
    carla.WeatherParameters.WetNight,
    carla.WeatherParameters.WetCloudyNight,
    carla.WeatherParameters.SoftRainNight,
    carla.WeatherParameters.MidRainyNight,
    carla.WeatherParameters.HardRainNight,
    carla.WeatherParameters.DustStorm
]

# Class mappings
SEG_COLORS = {
    12: [(60,20,220)],
    13: [(0,0,255)],
    14: [(142,0,0)],
    15: [(70,0,0)],
    16: [(100,60,0)],
    17: [(100,80,0)],
    18: [(230,0,0)],
    19: [(32,11,119)]
}

CLASS_NAMES = {
    12: "pedestrian",
    13: "rider",
    14: "car",
    15: "truck",
    16: "bus",
    17: "train",
    18: "motorcycle",
    19: "bicycle"
}
