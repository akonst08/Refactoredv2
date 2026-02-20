import argparse
import os
import shutil
import carla
import yaml


def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Output directories for images, labels, videos, and annotations
OUT_ROOT = "out"
IMG_DIR = os.path.join(OUT_ROOT, "images")
IMG_RGB_DIR = os.path.join(IMG_DIR, "rgb")
IMG_SEG_DIR = os.path.join(IMG_DIR, "seg")
IMG_BOXED_DIR = os.path.join(IMG_DIR, "boxed")
IMG_DETMASK_DIR = os.path.join(IMG_DIR, "detmask")
IMG_STATIC_DEBUG_DIR = os.path.join(IMG_DIR, "static_debug")  # NEW: Debug static mask
VID_DIR = os.path.join(OUT_ROOT, "videos")
VOC_DIR = os.path.join(OUT_ROOT, "annotations_voc")


# Clean and recreate output directories
def setup_output_dirs():
    for d in [IMG_DIR, IMG_RGB_DIR, IMG_SEG_DIR, IMG_BOXED_DIR, IMG_DETMASK_DIR, IMG_STATIC_DEBUG_DIR, VID_DIR, VOC_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

# Weather presets for diverse data collection
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
# Class ID to color mapping (BGR format for OpenCV)
CLASS_COLORS = {
    12: (0, 255, 255),   # pedestrian -> cyan
    13: (255, 165, 0),   # rider -> orange
    14: (0, 255, 0),     # car -> green
    15: (0, 128, 255),   # truck -> light orange
    16: (128, 0, 128),   # bus -> purple
    17: (75, 0, 130),    # train -> indigo
    18: (0, 0, 255),     # motorcycle -> red
    19: (255, 0, 0),     # bicycle -> blue
}
DEFAULT_COLOR = (200, 200, 200)  # gray fallback
# Semantic segmentation BGR colors for each dynamic actor class
SEG_COLORS = {
    12: [(60, 20, 220)],
    13: [(0, 0, 255)],
    14: [(142, 0, 0)],
    15: [(70, 0, 0)],
    16: [(100, 60, 0)],
    17: [(100, 80, 0)],
    18: [(230, 0, 0)],
    19: [(32, 11, 119)]
}

STATIC_CLASS_IDS = set(range(12, 20))  # 12–19 inclusive
# Class ID to human-readable name mapping
CLASS_NAMES = {
    1: "road",
    2: "sidewalk",
    3: "building",
    4: "wall",
    5: "fence",
    6: "pole",
    7: "traffic light",
    8: "traffic sign",
    9: "vegetation",
    10: "terrain",
    11: "sky",
    12: "pedestrian",
    13: "rider",
    14: "car",
    15: "truck",
    16: "bus",
    17: "train",
    18: "motorcycle",
    19: "bicycle"
}
