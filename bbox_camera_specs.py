"""
Camera field of view and coverage area calculations.
"""

import numpy as np
import math


def compute_vertical_fov(fov_horizontal_deg, image_width, image_height):
    """
    Calculate vertical field of view from horizontal FOV and image dimensions.
    
    Args:
        fov_horizontal_deg (float): Horizontal field of view in degrees
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
    
    Returns:
        float: Vertical field of view in degrees
    """
    aspect_ratio = image_height / image_width
    fov_h_rad = math.radians(fov_horizontal_deg)
    fov_v_rad = 2 * math.atan(aspect_ratio * math.tan(fov_h_rad / 2))
    fov_v_deg = math.degrees(fov_v_rad)
    
    return fov_v_deg


def compute_ground_coverage_area(fov_horizontal_deg, fov_vertical_deg, altitude_m):
    """
    Calculate ground coverage area for a top-down camera.
    
    Args:
        fov_horizontal_deg (float): Horizontal field of view in degrees
        fov_vertical_deg (float): Vertical field of view in degrees
        altitude_m (float): Camera altitude above ground in meters
    
    Returns:
        tuple: (width_m, height_m, area_m2)
            - width_m: Ground coverage width in meters
            - height_m: Ground coverage height in meters
            - area_m2: Total ground area in square meters
    """
    fov_h_rad = math.radians(fov_horizontal_deg)
    fov_v_rad = math.radians(fov_vertical_deg)
    
    ground_width = 2 * altitude_m * math.tan(fov_h_rad / 2)
    ground_height = 2 * altitude_m * math.tan(fov_v_rad / 2)
    ground_area = ground_width * ground_height
    
    return ground_width, ground_height, ground_area


def print_camera_specs(fov_horizontal_deg, image_width, image_height, altitude_m):
    """
    Print camera field of view and ground coverage area.
    
    Args:
        fov_horizontal_deg (float): Horizontal field of view in degrees
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        altitude_m (float): Camera altitude above ground in meters
    """
    # Calculate vertical FOV
    fov_v = compute_vertical_fov(fov_horizontal_deg, image_width, image_height)
    
    # Calculate ground coverage
    width_m, height_m, area_m2 = compute_ground_coverage_area(
        fov_horizontal_deg, fov_v, altitude_m
    )
    
    print("\n" + "="*70)
    print("CAMERA SPECIFICATIONS")
    print("="*70)
    
    print("\nFIELD OF VIEW")
    print("-"*70)
    print(f"  Horizontal FOV:      {fov_horizontal_deg:.2f}°")
    print(f"  Vertical FOV:        {fov_v:.2f}°")
    
    print("\nGROUND COVERAGE AREA")
    print("-"*70)
    print(f"  Width:               {width_m:.2f} meters")
    print(f"  Height:              {height_m:.2f} meters")
    print(f"  Total area:          {area_m2:.2f} m²")
    
    print("\n" + "="*70 + "\n")
