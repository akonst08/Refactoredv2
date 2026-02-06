            # 1) bicycles / motorcycles - use erosion to separate touching objects
            kernel_erode = np.ones((2, 2), np.uint8)  # Small erosion to separate
            kernel_dilate = np.ones((3, 3), np.uint8)  # Slightly larger dilation to restore size
            for cid in (18, 19):
                m = (mask == cid).astype(np.uint8)
                # Erode to separate touching objects
                m = cv2.erode(m, kernel_erode, iterations=1)
                # Dilate to restore approximate original size
                m = cv2.dilate(m, kernel_dilate, iterations=1)
                # Light closing to reconnect slightly fragmented parts (not touching objects!)
                kernel_close = np.ones((3, 3), np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                mask[mask == cid] = 0  # Clear old pixels
                mask[m == 1] = cid     # Set new pixels

            # 2) cars / trucks / buses (conservative - just light cleanup)
            kernel_car = np.ones((3, 3), np.uint8)
            for cid in (13, 14, 15, 16, 17):  # rider, car, truck, bus, train
                m = (mask == cid).astype(np.uint8)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_car, iterations=1)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_car, iterations=1)
                mask[mask == cid] = 0
                mask[m == 1] = cid