import carla
import numpy as np

def cam_callback(image, data_dict):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    data_dict['image'] = arr.reshape((image.height, image.width, 4))
    data_dict['frame'] = image.frame

def seg_callback(image, data_dict):
    image.convert(carla.ColorConverter.CityScapesPalette)
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    data_dict['image'] = arr.reshape((image.height, image.width, 4))
    data_dict['frame'] = image.frame

def create_cameras(world, bp_lib, cam_trans, fov, image_w, image_h):
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_w))
    camera_bp.set_attribute('image_size_y', str(image_h))
    camera_bp.set_attribute('fov', str(fov))
    camera = world.spawn_actor(camera_bp, cam_trans)
    
    semantic_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    semantic_bp.set_attribute('image_size_x', str(image_w))
    semantic_bp.set_attribute('image_size_y', str(image_h))
    semantic_bp.set_attribute('fov', str(fov))
    segmentation_cam = world.spawn_actor(semantic_bp, cam_trans)
    
    camera_data = {'image': np.zeros((image_h, image_w, 4), dtype=np.uint8), 'frame': -1}
    segmentation_data = {'image': np.zeros((image_h, image_w, 4), dtype=np.uint8), 'frame': -1}
    
    camera.listen(lambda image: cam_callback(image, camera_data))
    segmentation_cam.listen(lambda image: seg_callback(image, segmentation_data))
    
    return camera, segmentation_cam, camera_data, segmentation_data, camera_bp, semantic_bp
