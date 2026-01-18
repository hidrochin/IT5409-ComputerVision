import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# 1. UTILITIES
# ==========================================
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image(path, img, cmap=None):
    if cmap is None:
        cv2.imwrite(path, img)
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

# ==========================================
# 2. PIPELINE A: STANDARD (Pic 1 & Pic 2)
# Strategy: Top-Hat Transform -> Watershed
# ==========================================
def process_standard(img, img_out_dir):
    print("   [Mode] Standard Top-Hat + Watershed")

    # 1. Grayscale & Normalize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    save_image(os.path.join(img_out_dir, "01_grayscale.png"), norm_img)

    # 2. Background Removal (Top-Hat)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    tophat = cv2.morphologyEx(norm_img, cv2.MORPH_TOPHAT, morph_kernel)
    save_image(os.path.join(img_out_dir, "02_tophat.png"), tophat)
    
    # 3. Denoising
    blurred = cv2.medianBlur(tophat, 5)

    # 4. Otsu Thresholding
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_image(os.path.join(img_out_dir, "03_thresholded.png"), thresh)

    # 5. Watershed Markers
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    sure_bg = cv2.dilate(opening, kernel_clean, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 6. Apply Watershed
    img_color = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # 7. Count
    return finalize_count(img, markers, img_out_dir)

# ==========================================
# 3. PIPELINE B: WAVY IMAGE (Pic 3)
# Strategy: Vertical Kernel Opening -> Watershed
# ==========================================
def process_wavy(img, img_out_dir):
    print("   [Mode] Vertical Background Removal (Specific for Waves)")

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(os.path.join(img_out_dir, "01_grayscale.png"), gray)

    # 2. Vertical Background Estimation
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_vertical)
    save_image(os.path.join(img_out_dir, "02_background_waves.png"), background)
    
    # 3. Subtract Background
    clean = cv2.subtract(gray, background)
    
    # 4. Normalize
    clean_norm = cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX)
    save_image(os.path.join(img_out_dir, "03_clean_normalized.png"), clean_norm)

    # 5. Otsu Thresholding
    ret, thresh = cv2.threshold(clean_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean specks
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean, iterations=1)
    save_image(os.path.join(img_out_dir, "04_threshold.png"), thresh)

    # 6. Watershed Markers
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    sure_bg = cv2.dilate(thresh, kernel_clean, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 7. Apply Watershed
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 8. Count
    return finalize_count(img, markers, img_out_dir)

# ==========================================
# 4. PIPELINE C: DARK IMAGE (Pic 4)
# Strategy: Biased Otsu (Low Threshold)
# ==========================================
def process_dark(img, img_out_dir):
    print("   [Mode] Biased Otsu (Low Light)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_image(os.path.join(img_out_dir, "01_grayscale.png"), gray)
    
    # 1. Biased Otsu
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Lower threshold to 20% of Otsu to find dark grains
    biased_val = otsu_val * 0.2
    _, thresh = cv2.threshold(gray, biased_val, 255, cv2.THRESH_BINARY)
    save_image(os.path.join(img_out_dir, "02_biased_threshold.png"), thresh)
    
    # 2. Cleanup
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Simple Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
    
    final_viz = img.copy()
    count = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter noise
        if area > 5: 
            count += 1
            x = int(centroids[i][0])
            y = int(centroids[i][1])
            
            # --- FIX: USE RECTANGLE INSTEAD OF CONTOURS ---
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            cv2.rectangle(final_viz, (left, top), (left + w, top + h), (0, 255, 0), 1)
            cv2.putText(final_viz, str(count), (x-5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    save_image(os.path.join(img_out_dir, "result.png"), final_viz)
    return count

# ==========================================
# SHARED HELPER: FINAL COUNTING
# ==========================================
def finalize_count(original_img, markers, img_out_dir):
    labels = np.unique(markers)
    count = 0
    final_viz = original_img.copy()
    
    for label in labels:
        if label <= 1: continue # 0 is boundary, 1 is background

        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 10: # Minimum area filter
                count += 1
                cv2.drawContours(final_viz, [c], -1, (0, 255, 0), 1)
                
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(final_viz, str(count), (cX - 10, cY), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    save_image(os.path.join(img_out_dir, "result.png"), final_viz)
    return count

# ==========================================
# MAIN ROUTER
# ==========================================
def main():
    output_root = "results"
    ensure_dir(output_root)
    
    # 1. Search for images
    base_search = "input" if os.path.exists("input") else "."
    search_patterns = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    
    for pattern in search_patterns:
        files.extend(glob.glob(os.path.join(base_search, pattern)))
    
    if not files:
        print("No images found! Ensure they are in 'input' folder or current directory.")
        return

    files.sort()
    
    # 2. Process Loop
    for fpath in files:
        fname = os.path.basename(fpath)
        name_no_ext = os.path.splitext(fname)[0]
        
        # Create output folder per image
        img_out_dir = os.path.join(output_root, name_no_ext)
        ensure_dir(img_out_dir)
        
        print(f"Processing {fname}...")
        img = cv2.imread(fpath)
        if img is None: continue

        # --- INTELLIGENT ROUTING ---
        count = 0
        if "pic_3" in fname.lower():
            # Waves -> Vertical Removal Pipeline
            count = process_wavy(img, img_out_dir)
        elif "pic_4" in fname.lower():
            # Dark -> Biased Otsu Pipeline
            count = process_dark(img, img_out_dir)
        else:
            # Standard (Pic 1, 2, etc.) -> TopHat Pipeline
            count = process_standard(img, img_out_dir)
            
        print(f"   > Count: {count}")
        with open(os.path.join(img_out_dir, "result.txt"), "w") as f:
            f.write(f"Detected: {count}")

if __name__ == "__main__":
    main()