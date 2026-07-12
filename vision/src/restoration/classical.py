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

import sys
import cv2
import numpy as np


def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    """Correct color cast caused by selective light absorption in water.

    Assumes the "gray world" hypothesis: on average, a natural scene is
    gray. Scales each channel (B, G, R) so its mean approaches the overall
    average, compensating for water absorbing red first, then green.
    """
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
    """Enhance local contrast using CLAHE on the L (lightness) channel only.

    Working in LAB space (instead of applying CLAHE per BGR channel) avoids
    introducing color artifacts, since contrast is adjusted without touching
    the A/B color channels.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    l_equalized = clahe.apply(l_channel)

    lab_result = cv2.merge((l_equalized, a_channel, b_channel))

    return cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 20, eps: float = 1e-3) -> np.ndarray:
    """Edge-preserving smoothing of `src`, guided by the structure of `guide`.

    Used to refine the dark-channel-prior transmission map: without this
    filter, the transmission map looks blocky due to the morphological
    kernel used earlier.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (radius, radius))  # type: ignore[call-overload]
    mean_b = cv2.boxFilter(b, -1, (radius, radius))  # type: ignore[call-overload]

    return mean_a * guide + mean_b


def underwater_dark_channel_prior(
    img: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
) -> np.ndarray:
    """Remove haze-like scattering caused by underwater turbidity.

    Adapted Dark Channel Prior (Fattal / He et al.) for underwater images:
    uses the min of the G and B channels only, since the red channel is
    already attenuated by absorption and is unreliable for scattering
    estimation.
    """
    img_float = img.astype(np.float64) / 255.0

    min_gb = np.minimum(img_float[:, :, 0], img_float[:, :, 1])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_gb, kernel)

    num_pixels = dark_channel.size
    num_brightest = max(int(num_pixels * 0.001), 1)
    flat_indices = np.argpartition(dark_channel.ravel(), -num_brightest)[-num_brightest:]
    brightest_pixels = img_float.reshape(-1, 3)[flat_indices]
    atmospheric_light = np.max(brightest_pixels, axis=0)

    # Estimate the transmission map t(x): how much scene radiance survives
    # to reach the camera. Refined with a guided filter to remove blocky artifacts.
    norm_img = img_float / atmospheric_light
    min_gb_norm = np.minimum(norm_img[:, :, 0], norm_img[:, :, 1])
    transmission = 1 - omega * cv2.erode(min_gb_norm, kernel)
    transmission = np.clip(transmission, t0, 1.0)

    gray_guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    transmission = _guided_filter(gray_guide, transmission, radius=40, eps=1e-3)
    transmission = np.clip(transmission, t0, 1.0)

    result = np.empty_like(img_float)
    
    for c in range(3):
        result[:, :, c] = (img_float[:, :, c] - atmospheric_light[c]) / transmission + atmospheric_light[c]

    result = np.clip(result, 0, 1) * 255
    return result.astype(np.uint8)


def restore_classical_pipeline(img: np.ndarray) -> np.ndarray:
    """Run the full pipeline: white balance -> CLAHE -> dehazing.

    Order matters: correcting color before contrast avoids amplifying the
    cast; correcting contrast before dehazing improves the atmospheric
    light estimate in step 3.
    """
    step1 = gray_world_white_balance(img)
    step2 = clahe_per_channel(step1)
    step3 = underwater_dark_channel_prior(step2)
    return step3


if __name__ == "__main__":
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