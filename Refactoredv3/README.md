
---

## File Descriptions

### `rgb/` - RGB Frames
Raw RGB images captured from the camera. 

### `seg/` - Color-Coded Segmentation
Color visualization of semantic segmentation using the CityScapes palette. Each pixel is colored according to its semantic class (see color table below). 

### `detmask/` - Class ID Segmentation
Grayscale semantic segmentation masks where each pixel intensity represents the class ID (0-29). Pixel value directly corresponds to class ID in the table below.

### `annotations_voc/` - Bounding Box Annotations
- Pascal VOC XML format
- Contains bounding boxes only for classes 12-19
- **Note:** Every frame has a corresponding XML file, even if no objects are detected. Frames without objects contain an empty annotation with only size information (see example below)

- Coordinates: `<xmin>`, `<ymin>`, `<xmax>`, `<ymax>` (pixels, 0-indexed)
- All coordinates clipped to image bounds: 0 ≤ x < 1280, 0 ≤ y < 720

---
**Example VOC XML without objects:**
```xml
<annotation>
  <folder>out/images/rgb</folder>
  <filename>frame_000208.png</filename>
  <size>
    <width>1280</width>
    <height>720</height>
    <depth>3</depth>
  </size>
</annotation>
```
## Semantic Segmentation Classes

All 30 classes (0-29) are present in both `seg/` and `detmask/` folders. The table below shows the complete CityScapes palette used in this dataset.

### Complete Color Palette

| ID | Class Name | RGB Color | Category |
|----|------------|-----------|----------|
| 0 | Unlabeled | (0, 0, 0) | Background |
| 1 | Road | (128, 64, 128) | Infrastructure |
| 2 | Sidewalk | (244, 35, 232) | Infrastructure |
| 3 | Building | (70, 70, 70) | Infrastructure |
| 4 | Wall | (102, 102, 156) | Infrastructure |
| 5 | Fence | (190, 153, 153) | Infrastructure |
| 6 | Pole | (153, 153, 153) | Infrastructure |
| 7 | TrafficLight | (250, 170, 30) | Infrastructure |
| 8 | TrafficSign | (220, 220, 0) | Infrastructure |
| 9 | Vegetation | (107, 142, 35) | Nature |
| 10 | Terrain | (152, 251, 152) | Nature |
| 11 | Sky | (70, 130, 180) | Nature |
| 12 | Pedestrian | (220, 20, 60) | **Detection Target** |
| 13 | Rider | (255, 0, 0) | **Detection Target** |
| 14 | Car | (0, 0, 142) | **Detection Target** |
| 15 | Truck | (0, 0, 70) | **Detection Target** |
| 16 | Bus | (0, 60, 100) | **Detection Target** |
| 17 | Train | (0, 80, 100) | **Detection Target** |
| 18 | Motorcycle | (0, 0, 230) | **Detection Target** |
| 19 | Bicycle | (119, 11, 32) | **Detection Target** |
| 20 | Static | (110, 190, 160) | Special |
| 21 | Dynamic | (170, 120, 50) | Special |
| 22 | Other | (55, 90, 80) | Background |
| 23 | Water | (45, 60, 150) | Nature |
| 24 | RoadLine | (157, 234, 50) | Infrastructure |
| 25 | Ground | (81, 0, 81) | Infrastructure |
| 26 | Bridge | (150, 100, 100) | Infrastructure |
| 27 | RailTrack | (230, 150, 140) | Infrastructure |
| 28 | GuardRail | (180, 165, 180) | Infrastructure |
| 29 | Rock | (180, 130, 70) | Nature |

**Note:** Bounding box annotations in `annotations_voc/` are provided **only for classes 12-19** (vehicles and pedestrians). All other classes appear only in the semantic segmentation masks.

---

## Object Detection Classes

The following classes have bounding box annotations in Pascal VOC format:

| Class ID | Class Name | VOC Label |
|----------|------------|-----------|
| 12 | Pedestrian | `pedestrian` |
| 13 | Rider | `rider` |
| 14 | Car | `car` |
| 15 | Truck | `truck` |
| 16 | Bus | `bus` |
| 17 | Train | `train` |
| 18 | Motorcycle | `motorcycle` |
| 19 | Bicycle | `bicycle` |

---

## File Format Details

### RGB Images (`rgb/`)
- Standard PNG images (1280×720)
- Color format: BGR (OpenCV convention)

### Class ID Masks (`detmask/`)
- Grayscale PNG (8-bit, single channel)
- Pixel intensity = Class ID (0-29)
- Example: Pixel value 14 = car, 1 = road, 12 = pedestrian

### Color Segmentation (`seg/`)
- Color-coded PNG using CityScapes palette
- Each pixel colored according to its class (see table above)


### VOC Annotations (`annotations_voc/`)
- Pascal VOC XML format
- Contains bounding boxes only for classes 12-19
- All coordinates clipped to image bounds: 0 ≤ x < 1280, 0 ≤ y < 720
