"""
Visualize semantic class pixels from exported CARLA dataset.
Shows each class (12-19) in a different color overlay on the original image.
"""

import cv2
import numpy as np
import os
import sys

# Class mapping
CLASS_INFO = {
    12: {"name": "pedestrian", "color": (0, 255, 255)},    # cyan
    13: {"name": "rider", "color": (255, 165, 0)},         # orange
    14: {"name": "car", "color": (0, 255, 0)},             # green
    15: {"name": "truck", "color": (0, 128, 255)},         # light orange
    16: {"name": "bus", "color": (128, 0, 128)},           # purple
    17: {"name": "train", "color": (75, 0, 130)},          # indigo
    18: {"name": "motorcycle", "color": (0, 0, 255)},      # red
    19: {"name": "bicycle", "color": (255, 0, 0)},         # blue
}

# CARLA export settings
FRAME_START = 32      # First exported frame number
FRAME_INTERVAL = 4    # Frames are numbered: 32, 36, 40, 44, ...

def visualize_frame(frame_id="000032", output_dir="out/visualizations"):
    """
    Visualize semantic classes for a single frame.
    
    Args:
        frame_id: Frame number (6-digit string)
        output_dir: Where to save visualization images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load RGB image
    rgb_path = f"out/images/rgb/frame_{frame_id}.png"
    if not os.path.exists(rgb_path):
        print(f"❌ RGB image not found: {rgb_path}")
        return
    
    img_bgr = cv2.imread(rgb_path)
    print(f"✅ Loaded RGB image: {rgb_path}")
    
    # Load class ID mask
    classid_path = f"out/images/detmask/frame_{frame_id}_classid.png"
    if not os.path.exists(classid_path):
        print(f"❌ Class ID mask not found: {classid_path}")
        return
    
    classid_mask = cv2.imread(classid_path, cv2.IMREAD_GRAYSCALE)
    print(f"✅ Loaded class ID mask: {classid_path}")
    
    # Get unique class IDs in this frame
    unique_ids = np.unique(classid_mask)
    target_classes = [cid for cid in unique_ids if cid in CLASS_INFO]
    
    print(f"\n📊 Frame {frame_id} statistics:")
    print(f"   Image size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
    print(f"   Total unique class IDs: {len(unique_ids)}")
    print(f"   Target classes (12-19) found: {target_classes}")
    
    # 1) Create individual class visualizations
    for cid in sorted(target_classes):
        info = CLASS_INFO[cid]
        mask = (classid_mask == cid)
        pixel_count = mask.sum()
        percentage = 100 * pixel_count / (img_bgr.shape[0] * img_bgr.shape[1])
        
        # Create overlay: original image + colored mask
        overlay = img_bgr.copy()
        overlay[mask] = info["color"]
        
        # Blend for semi-transparent effect
        blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
        
        # Add text label
        label = f"Class {cid}: {info['name']} ({pixel_count:,} pixels, {percentage:.2f}%)"
        cv2.putText(blended, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save individual class visualization
        out_path = os.path.join(output_dir, f"frame_{frame_id}_class{cid:02d}_{info['name']}.png")
        cv2.imwrite(out_path, blended)
        print(f"   ✅ Class {cid:2d} ({info['name']:12s}): {pixel_count:7,d} pixels ({percentage:5.2f}%) -> {out_path}")
    
    # 2) Create combined multi-class visualization
    combined = img_bgr.copy()
    for cid in sorted(target_classes):
        mask = (classid_mask == cid)
        combined[mask] = CLASS_INFO[cid]["color"]
    
    # Blend with original
    combined_blended = cv2.addWeighted(img_bgr, 0.5, combined, 0.5, 0)
    
    # Add legend
    legend_y = 30
    for cid in sorted(target_classes):
        info = CLASS_INFO[cid]
        cv2.rectangle(combined_blended, (10, legend_y), (30, legend_y + 20), info["color"], -1)
        cv2.putText(combined_blended, f"{cid}: {info['name']}", (40, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 30
    
    combined_path = os.path.join(output_dir, f"frame_{frame_id}_all_classes.png")
    cv2.imwrite(combined_path, combined_blended)
    print(f"\n✅ Combined visualization saved: {combined_path}")
    
    # 3) Create isolated pixels visualization (only class pixels, black background)
    isolated = np.zeros_like(img_bgr)
    for cid in sorted(target_classes):
        mask = (classid_mask == cid)
        isolated[mask] = CLASS_INFO[cid]["color"]
    
    isolated_path = os.path.join(output_dir, f"frame_{frame_id}_isolated.png")
    cv2.imwrite(isolated_path, isolated)
    print(f"✅ Isolated pixels saved: {isolated_path}")
    
    # 4) Create side-by-side comparison
    h, w = img_bgr.shape[:2]
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = img_bgr                    # Original
    comparison[:, w:2*w] = combined_blended        # Overlay
    comparison[:, 2*w:] = isolated                 # Isolated
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comparison, "Overlay", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(comparison, "Isolated", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    comparison_path = os.path.join(output_dir, f"frame_{frame_id}_comparison.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"✅ Comparison saved: {comparison_path}")

def visualize_multiple_frames(num_frames=5, output_dir="out/visualizations"):
    """Visualize multiple consecutive frames with +4 increment starting from 32."""
    print(f"\n{'='*60}")
    print(f"VISUALIZING {num_frames} FRAMES (32, 36, 40, 44, ...)")
    print(f"{'='*60}\n")
    
    for i in range(num_frames):
        frame_num = FRAME_START + (i * FRAME_INTERVAL)  # 32, 36, 40, 44, ...
        frame_id = f"{frame_num:06d}"
        print(f"\n--- Processing Frame {frame_id} (index {i+1}/{num_frames}) ---")
        visualize_frame(frame_id, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")

def create_class_statistics(num_frames=100):
    """Generate statistics across multiple frames (32, 36, 40, ...)."""
    print(f"\n{'='*60}")
    print(f"ANALYZING CLASS DISTRIBUTION ACROSS {num_frames} FRAMES")
    print(f"{'='*60}\n")
    
    class_pixel_counts = {cid: 0 for cid in CLASS_INFO.keys()}
    frames_with_class = {cid: 0 for cid in CLASS_INFO.keys()}
    total_pixels = 0
    valid_frames = 0
    
    for i in range(num_frames):
        frame_num = FRAME_START + (i * FRAME_INTERVAL)  # 32, 36, 40, 44, ...
        frame_id = f"{frame_num:06d}"
        classid_path = f"out/images/detmask/frame_{frame_id}_classid.png"
        
        if not os.path.exists(classid_path):
            print(f"⚠️  Skipping frame {frame_id} (not found)")
            continue
        
        classid_mask = cv2.imread(classid_path, cv2.IMREAD_GRAYSCALE)
        valid_frames += 1
        total_pixels += classid_mask.size
        
        for cid in CLASS_INFO.keys():
            mask = (classid_mask == cid)
            pixel_count = mask.sum()
            if pixel_count > 0:
                class_pixel_counts[cid] += pixel_count
                frames_with_class[cid] += 1
    
    print(f"\n📊 Statistics Summary:")
    print(f"   Analyzed frames: {valid_frames} / {num_frames}")
    print(f"   Total pixels analyzed: {total_pixels:,}")
    print(f"\n{'Class':<12} {'Pixels':<12} {'% Coverage':<12} {'Frames':<10} {'Avg px/frame':<15}")
    print("-" * 70)
    
    for cid in sorted(CLASS_INFO.keys()):
        info = CLASS_INFO[cid]
        pixels = class_pixel_counts[cid]
        percentage = 100 * pixels / total_pixels if total_pixels > 0 else 0
        frames = frames_with_class[cid]
        avg_px = pixels / valid_frames if valid_frames > 0 else 0
        
        print(f"{cid:2d} {info['name']:<9} {pixels:<12,} {percentage:<12.4f} {frames:<10} {avg_px:<15,.1f}")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "single":
            # Visualize a single frame
            frame_id = sys.argv[2] if len(sys.argv) > 2 else "000032"
            visualize_frame(frame_id)
        
        elif command == "multiple":
            # Visualize multiple frames (32, 36, 40, ...)
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            visualize_multiple_frames(count)
        
        elif command == "stats":
            # Generate statistics across frames
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            create_class_statistics(count)
        
        else:
            print("Unknown command. Use: single, multiple, or stats")
    
    else:
        # Default: visualize first frame
        print("Usage:")
        print("  python visualize.py single [frame_id]          # e.g., single 000032")
        print("  python visualize.py multiple [num_frames]      # e.g., multiple 10 (processes 32,36,40,...)")
        print("  python visualize.py stats [num_frames]         # e.g., stats 100")
        print("\nRunning default: visualizing frame 000032")
        visualize_frame("000032")