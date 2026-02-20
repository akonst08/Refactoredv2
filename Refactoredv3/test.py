"""
Validate that Class ID 14 in detmask corresponds to cars 
and has the correct color (0, 0, 142) in segmentation masks.
"""

import cv2
import numpy as np
import sys

def validate_car_class(frame_id="000032"):
    """
    Validate that class ID 14 is indeed cars with color (0, 0, 142).
    """
    print("\n" + "="*80)
    print(f"VALIDATING CLASS ID 14 (CAR) FOR FRAME {frame_id}")
    print("="*80)
    
    # Expected values
    EXPECTED_CLASS_ID = 14
    EXPECTED_RGB = (0, 0, 142)
    EXPECTED_BGR = (142, 0, 0)  # OpenCV uses BGR
    
    # Load class ID mask (detmask)
    detmask_path = f"out/images/detmask/frame_{frame_id}_classid.png"
    classid_mask = cv2.imread(detmask_path, cv2.IMREAD_GRAYSCALE)
    
    if classid_mask is None:
        print(f"❌ Could not load: {detmask_path}")
        return False
    
    print(f"\n✅ Loaded detmask: {detmask_path}")
    print(f"   Shape: {classid_mask.shape}")
    
    # Load color segmentation mask
    seg_path = f"out/images/seg/frame_{frame_id}_seg.png"
    seg_mask = cv2.imread(seg_path)
    
    if seg_mask is None:
        print(f"❌ Could not load: {seg_path}")
        return False
    
    print(f"✅ Loaded segmentation: {seg_path}")
    print(f"   Shape: {seg_mask.shape}")
    
    # Check if class 14 exists in detmask
    unique_classes = np.unique(classid_mask)
    print(f"\n📊 Unique class IDs in frame: {sorted(unique_classes)}")
    
    if EXPECTED_CLASS_ID not in unique_classes:
        print(f"\n⚠️  Class ID {EXPECTED_CLASS_ID} (car) NOT found in this frame!")
        print(f"   Try another frame that contains cars.")
        return False
    
    # Extract pixels with class ID 14
    car_mask = (classid_mask == EXPECTED_CLASS_ID)
    car_pixel_count = car_mask.sum()
    car_percentage = 100 * car_pixel_count / classid_mask.size
    
    print(f"\n✅ Found Class ID {EXPECTED_CLASS_ID} pixels:")
    print(f"   Pixel count: {car_pixel_count:,}")
    print(f"   Percentage: {car_percentage:.2f}%")
    
    # Extract corresponding colors from segmentation mask
    car_colors_bgr = seg_mask[car_mask]  # Nx3 array of BGR colors
    
    # Find the most common color (should be uniform)
    unique_colors, counts = np.unique(car_colors_bgr, axis=0, return_counts=True)
    most_common_idx = counts.argmax()
    most_common_color_bgr = tuple(unique_colors[most_common_idx])
    most_common_color_rgb = (most_common_color_bgr[2], most_common_color_bgr[1], most_common_color_bgr[0])
    most_common_count = counts[most_common_idx]
    most_common_percentage = 100 * most_common_count / car_pixel_count
    
    print(f"\n🎨 Color Analysis for Class ID {EXPECTED_CLASS_ID}:")
    print(f"   Most common color (BGR): {most_common_color_bgr}")
    print(f"   Most common color (RGB): {most_common_color_rgb}")
    print(f"   Frequency: {most_common_count:,} pixels ({most_common_percentage:.2f}%)")
    
    if len(unique_colors) > 1:
        print(f"\n   ⚠️  Found {len(unique_colors)} different colors (likely anti-aliasing at edges)")
        print(f"   Top 5 colors:")
        top_indices = counts.argsort()[-5:][::-1]
        for idx in top_indices:
            color_bgr = tuple(unique_colors[idx])
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            count = counts[idx]
            pct = 100 * count / car_pixel_count
            print(f"      RGB{color_rgb}: {count:,} pixels ({pct:.2f}%)")
    
    # Validate color
    print(f"\n🔍 Validation:")
    print(f"   Expected RGB: {EXPECTED_RGB}")
    print(f"   Actual RGB:   {most_common_color_rgb}")
    
    # Allow small tolerance for compression artifacts
    tolerance = 5
    color_match = all(abs(a - e) <= tolerance for a, e in zip(most_common_color_rgb, EXPECTED_RGB))
    
    if color_match:
        print(f"   ✅ COLOR MATCH! Class ID 14 has correct color (0, 0, 142)")
    else:
        print(f"   ❌ COLOR MISMATCH! Expected {EXPECTED_RGB} but got {most_common_color_rgb}")
        return False
    
    # Create visualizations
    print(f"\n🖼️  Creating validation visualizations...")
    
    import os
    os.makedirs("out/validation", exist_ok=True)
    
    # 1. Extracted detmask (only class 14 pixels, white on black)
    detmask_extracted = np.zeros_like(classid_mask)
    detmask_extracted[car_mask] = 255  # White pixels where cars are
    cv2.imwrite(f"out/validation/frame_{frame_id}_detmask_class14_extracted.png", detmask_extracted)
    print(f"   ✅ Saved: out/validation/frame_{frame_id}_detmask_class14_extracted.png")
    
    # 2. Extracted segmentation color (only class 14 pixels with their color)
    seg_extracted = np.zeros_like(seg_mask)
    seg_extracted[car_mask] = seg_mask[car_mask]  # Copy car pixels with their color
    cv2.imwrite(f"out/validation/frame_{frame_id}_seg_class14_extracted.png", seg_extracted)
    print(f"   ✅ Saved: out/validation/frame_{frame_id}_seg_class14_extracted.png")
    
    # 3. Original RGB with cars isolated
    rgb_path = f"out/images/rgb/frame_{frame_id}.png"
    rgb_img = cv2.imread(rgb_path)
    
    if rgb_img is not None:
        # Create isolated car image from RGB
        car_only = np.zeros_like(rgb_img)
        car_only[car_mask] = rgb_img[car_mask]
        cv2.imwrite(f"out/validation/frame_{frame_id}_rgb_class14_isolated.png", car_only)
        print(f"   ✅ Saved: out/validation/frame_{frame_id}_rgb_class14_isolated.png")
        
        # Create colored overlay on RGB
        overlay = rgb_img.copy()
        overlay[car_mask] = [0, 255, 0]  # Highlight cars in green
        blended = cv2.addWeighted(rgb_img, 0.7, overlay, 0.3, 0)
        
        # Add label
        label = f"Class 14 (Car): {car_pixel_count:,} pixels - Color RGB{most_common_color_rgb}"
        cv2.putText(blended, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imwrite(f"out/validation/frame_{frame_id}_rgb_class14_highlighted.png", blended)
        print(f"   ✅ Saved: out/validation/frame_{frame_id}_rgb_class14_highlighted.png")
    
    # 4. Create side-by-side comparison
    h, w = classid_mask.shape
    
    # Convert grayscale detmask to BGR for concatenation
    detmask_extracted_bgr = cv2.cvtColor(detmask_extracted, cv2.COLOR_GRAY2BGR)
    
    # Create comparison image: detmask | seg color | RGB original
    if rgb_img is not None:
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = detmask_extracted_bgr        # Detmask extraction
        comparison[:, w:2*w] = seg_extracted              # Seg color extraction
        comparison[:, 2*w:] = car_only                    # RGB isolated
        
        # Add labels
        cv2.putText(comparison, "Detmask (Class 14)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, "Seg Color (Class 14)", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, "RGB Original (Class 14)", (2*w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imwrite(f"out/validation/frame_{frame_id}_comparison_3way.png", comparison)
        print(f"   ✅ Saved: out/validation/frame_{frame_id}_comparison_3way.png")
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE - CLASS ID 14 IS CORRECTLY MAPPED TO CARS")
    print("="*80)
    print(f"\n📁 All visualizations saved to: out/validation/")
    
    return True

def find_frames_with_cars(num_frames=25, start_frame=32, interval=4):
    """
    Search through frames to find ones that contain class ID 14 (cars).
    """
    print("\n" + "="*80)
    print(f"SEARCHING FOR FRAMES WITH CLASS ID 14 (CARS)")
    print("="*80)
    
    frames_with_cars = []
    
    for i in range(num_frames):
        frame_num = start_frame + (i * interval)
        frame_id = f"{frame_num:06d}"
        detmask_path = f"out/images/detmask/frame_{frame_id}_classid.png"
        
        try:
            mask = cv2.imread(detmask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            unique_classes = np.unique(mask)
            if 14 in unique_classes:
                car_pixels = (mask == 14).sum()
                frames_with_cars.append((frame_id, car_pixels))
                print(f"   ✅ Frame {frame_id}: {car_pixels:,} car pixels")
        except:
            continue
    
    print(f"\n📊 Found {len(frames_with_cars)} frames with cars")
    
    if frames_with_cars:
        # Sort by number of car pixels
        frames_with_cars.sort(key=lambda x: x[1], reverse=True)
        print(f"\n💡 Frame with most cars: {frames_with_cars[0][0]} ({frames_with_cars[0][1]:,} pixels)")
        return frames_with_cars[0][0]
    
    return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "search":
            # Search for frames with cars
            best_frame = find_frames_with_cars()
            if best_frame:
                print(f"\n💡 Validating best frame: {best_frame}")
                validate_car_class(best_frame)
            else:
                print("\n❌ No frames with cars found!")
        
        elif command == "validate":
            # Validate specific frame
            frame_id = sys.argv[2] if len(sys.argv) > 2 else "000032"
            validate_car_class(frame_id)
        
        else:
            print("Unknown command. Use: search or validate <frame_id>")
    
    else:
        print("Usage:")
        print("  python test.py search              # Find frames with cars")
        print("  python test.py validate 000032     # Validate specific frame")
        print("\nRunning search mode...")
        best_frame = find_frames_with_cars()
        if best_frame:
            validate_car_class(best_frame)