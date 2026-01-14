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
YOLO_DIR = os.path.join(OUT_ROOT, "labels_yolo")
VID_DIR = os.path.join(OUT_ROOT, "videos")
VOC_DIR = os.path.join(OUT_ROOT, "annotations_voc")

# Clean and recreate output directories
def setup_output_dirs():
    for d in [IMG_DIR, IMG_RGB_DIR, IMG_SEG_DIR, IMG_BOXED_DIR, IMG_DETMASK_DIR, YOLO_DIR, VID_DIR, VOC_DIR]:
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

# Semantic segmentation BGR colors for each object class
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

# Class ID to human-readable name mapping
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
