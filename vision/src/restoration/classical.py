"""
Classical (non-deep-learning) underwater image restoration baseline.

Pipeline order matters:
1. gray_world_white_balance      -> fixes color cast (water absorbs red/green first)
2. clahe_per_channel             -> fixes local contrast/brightness (applied only to lightness in LAB space, so color is untouched)
3. underwater_dark_channel_prior -> removes haze-like scattering caused by turbidity

Each function takes a BGR uint8 image (OpenCV's default) and returns one of the
same shape and dtype. No file I/O happens inside these functions on purpose —
that responsibility belongs to the calling script (evaluate_restoration.py).
"""

import cv2
import numpy as np

def gray_world_white_balance(img: np.ndarray) -> np.ndarray:

#

    result = img.copy().astype(np.float32)

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)

def clahe_per_channel(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:

#

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    l_equalized = clahe.apply(l_channel)

    lab_result = cv2.merge((l_equalized, a_channel, b_channel))

    return cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

def underwater_dark_channel_prior(
    img: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
) -> np.ndarray:

#

    img_float = img.astype(np.float64) / 255.0

    min_gb = np.minimum(img_float[:, :, 0], img_float[:, :, 1])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_gb, kernel)

#

    num_pixels = dark_channel.size
    num_brightest = max(int(num_pixels * 0.001), 1)
    
    flat_indices = np.argpartition(dark_channel.ravel(), -num_brightest)[-num_brightest:]
    
    brightest_pixels = img_float.reshape(-1, 3)[flat_indices]
    
    atmospheric_light = np.max(brightest_pixels, axis=0)

#

    norm_img = img_float / atmospheric_light
    
    min_gb_norm = np.minimum(norm_img[:, :, 0], norm_img[:, :, 1])
    
    transmission = 1 - omega * cv2.erode(min_gb_norm, kernel)
    
    transmission = np.clip(transmission, t0, 1.0)

#

    result = np.empty_like(img_float)
    
    for c in range(3):
        result[:, :, c] = (img_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]

    result = np.clip(result, 0, 1) * 255
    return result.astype(np.uint8)

def restore_classical_pipeline(img: np.ndarray) -> np.ndarray:
    step1 = gray_world_white_balance(img)
    step2 = clahe_per_channel(step1)
    step3 = underwater_dark_channel_prior(step2)
    return step3

#

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classical.py <path_to_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Could not read image at {input_path}")
        sys.exit(1)

    wb = gray_world_white_balance(img)
    wb_clahe = clahe_per_channel(wb)
    full = underwater_dark_channel_prior(wb_clahe)

    cv2.imwrite("step1_white_balance.jpg", wb)
    cv2.imwrite("step2_clahe.jpg", wb_clahe)
    cv2.imwrite("step3_udcp_final.jpg", full)
    
    print("Saved step1_white_balance.jpg, step2_clahe.jpg, step3_udcp_final.jpg")