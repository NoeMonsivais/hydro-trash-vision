"""
No-reference underwater image quality metrics: UIQM and UCIQE.

Unlike PSNR/SSIM (which need a clean reference image and live in scikit-image),
these metrics work on a single degraded image with no reference — which is why
they're used on RUIE (real underwater photos with no "ground truth" clean version)
and why they had to be implemented by hand here.
"""

import sys

import cv2
import numpy as np


def _alpha_trimmed_mean_and_var(channel: np.ndarray, alpha: float = 0.1):
    """Compute mean and variance after trimming the extreme values.

    Sorts the channel values and discards the top/bottom `alpha` fraction
    before computing statistics, reducing sensitivity to outlier pixels.
    """
    values=np.sort(channel.ravel())
    n = values.size
    trim = int(alpha*n)

    trimmed = values[trim:n-trim] if trim > 0 else values

    mean = np.mean(trimmed)
    var = np.mean((trimmed - mean) ** 2)

    return mean, var


def _uicm(img_bgr: np.ndarray) -> float:
    """Estimate underwater image colorfulness measure (UICM).

    Based on the opponent color space (red-green and yellow-blue
    differences), using alpha-trimmed statistics to reduce sensitivity
    to outlier pixels.
    """
    img = img_bgr.astype(np.float64)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    rg = r - g
    yb = (r + g) / 2 - b
    mu_rg, var_rg = _alpha_trimmed_mean_and_var(rg)
    mu_yb, var_yb = _alpha_trimmed_mean_and_var(yb)

    uicm = (
        -0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2) \
        + 0.1586 * np.sqrt(var_rg + var_yb)
        )
    
    return uicm


def _block_eme(gray: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    """Compute the Enhancement Measure Estimation (EME) of a single-channel image.

    Splits the image into non-overlapping blocks and averages the log-ratio
    of max/min intensity per block.
    """
    h, w = gray.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size

    total = 0.0
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_max = np.max(block)
            block_min = np.min(block)
            total += np.log((block_max + eps) / (block_min + eps))

    n_blocks = max(n_blocks_h * n_blocks_w, 1)
    return (2.0 / n_blocks) * total


def _uism(img_bgr: np.ndarray) -> float:
    """Estimate underwater image sharpness measure (UISM).

    Applies a Sobel edge filter per channel, then measures local sharpness
    (EME) block by block on each edge map, combined via luminance weights.
    """
    b, g, r = (img_bgr[:, :, i].astype(np.float64) for i in range(3))

    sobel_b = np.hypot(cv2.Sobel(b, cv2.CV_64F, 1, 0), cv2.Sobel(b, cv2.CV_64F, 0, 1))
    sobel_g = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    sobel_r = np.hypot(cv2.Sobel(r, cv2.CV_64F, 1, 0), cv2.Sobel(r, cv2.CV_64F, 0, 1))

    eme_b = _block_eme(sobel_b)
    eme_g = _block_eme(sobel_g)
    eme_r = _block_eme(sobel_r)

    # Weights are the standard ITU-R BT.601 luminance coefficients.
    uism = 0.299 * eme_r + 0.587 * eme_g + 0.114 * eme_b
    return uism


def _uiconm(img_bgr: np.ndarray, block_size: int = 8, eps: float = 1e-6) -> float:
    """Estimate underwater image contrast measure (UIConM).

    Splits the grayscale image into blocks and averages a log-based
    contrast term per block.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    h, w = gray.shape
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size

    total = 0.0
    contrast = 0.0
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            block_max = np.max(block)
            block_min = np.min(block)
            contrast = (block_max - block_min) / (block_max + block_min + eps)
            total += contrast * np.log(contrast + eps)

    n_blocks = max(n_blocks_h * n_blocks_w, 1)
    return -1.0 * total / n_blocks


def calculate_uiqm(img_bgr: np.ndarray, c1: float = 0.0282, c2: float = 0.2953, c3: float = 3.5753) -> float:
    """Compute UIQM: Underwater Image Quality Measure.

    Weighted combination of colorfulness (UICM), sharpness (UISM), and
    contrast (UIConM). Default coefficients are the ones proposed in the
    original UIQM paper (Panetta et al., 2016).
    """
    uicm = _uicm(img_bgr)
    uicm = _uicm(img_bgr)
    uism = _uism(img_bgr)
    uiconm = _uiconm(img_bgr)

    return c1 * uicm + c2 * uism + c3 * uiconm


def calculate_uciqe(img_bgr: np.ndarray, c1: float = 0.4680, c2: float = 0.2745, c3: float = 0.2576) -> float:
    """Compute UCIQE: Underwater Color Image Quality Evaluation.

    Weighted combination of chroma std deviation, luminance contrast, and
    average saturation. Default coefficients (c1, c2, c3) are the ones
    proposed in the original UCIQE paper (Yang & Sowmya, 2015).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_channel, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    chroma = np.sqrt(a_channel**2 + b_channel**2)
    sigma_c = np.std(chroma)

    # Use top/bottom 1% to reduce sensitivity to outlier pixels.
    l_sorted = np.sort(l_channel.ravel())
    n = l_sorted.size
    top = l_sorted[int(0.99 * n):]
    bottom = l_sorted[:int(0.01 * n)]
    con_l = np.mean(top) - np.mean(bottom)

    with np.errstate(divide='ignore', invalid='ignore'):
        saturation = np.where(l_channel != 0, chroma / l_channel, 0)
        mu_s = np.mean(saturation)

    return float(c1 * sigma_c + c2 * con_l + c3 * mu_s)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python metrics.py <path_to_image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Error: Could not read image from {sys.argv[1]}")
        sys.exit(1)

    print(f"UIQM: {calculate_uiqm(img):.4f}")
    print(f"UCIQE: {calculate_uciqe(img):.4f}")