import os
import xml.etree.ElementTree as ET

# Write bounding box annotations in Pascal VOC XML format
def write_voc_xml(path, boxes_xyxy_cls, W, H, image_path, CLASS_NAMES):
    # Create root annotation element
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.dirname(image_path)
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    
    # Add image size information
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text  = str(W)
    ET.SubElement(size, "height").text = str(H)
    ET.SubElement(size, "depth").text  = "3"
    
    # Add each bounding box as an object element
    for (xmin, ymin, xmax, ymax, cid) in boxes_xyxy_cls:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = CLASS_NAMES.get(cid, f"class_{cid}")
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)    
    
    # Write XML file to disk
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    del tree
