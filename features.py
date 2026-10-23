"""
DetectAI - Forensic Feature Extraction Module
Shared feature extractor used by both the Random Forest training pipeline
(train_rf.py) and the real-time inference server (app.py).

Extracts multidimensional forensic signals:
1. Error Level Analysis (ELA) - detects compression artifacts & resaving traces
2. 2D-FFT Frequency Analysis - detects GAN high-frequency spectral fingerprints
3. Laplacian Variance - measures edge sharpness vs smoothing/blurring
4. Local Binary Patterns (LBP) - captures micro-texture inconsistencies
5. Color-Channel Statistics (RGB + HSV) - identifies unnatural color distributions
6. Edge Density (Canny) - quantifies high-frequency boundary structure
7. Noise-Residual Variance - reveals synthetic smoothing & missing camera noise
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from scipy import stats

# Master list of all feature names in deterministic order
FEATURE_NAMES = [
    # ELA features (3)
    "ela_mean",
    "ela_std",
    "ela_max",
    # FFT frequency features (3)
    "fft_high_freq_ratio",
    "fft_spectral_mean",
    "fft_spectral_std",
    # Blur / Sharpness (1)
    "laplacian_variance",
    # LBP texture stats (10 histogram bins + summary) (12)
    "lbp_mean",
    "lbp_std",
    "lbp_bin_0",
    "lbp_bin_1",
    "lbp_bin_2",
    "lbp_bin_3",
    "lbp_bin_4",
    "lbp_bin_5",
    "lbp_bin_6",
    "lbp_bin_7",
    "lbp_bin_8",
    "lbp_bin_9",
    # Color Channel Stats RGB (9)
    "color_r_mean",
    "color_r_std",
    "color_r_skew",
    "color_g_mean",
    "color_g_std",
    "color_g_skew",
    "color_b_mean",
    "color_b_std",
    "color_b_skew",
    # Color Channel Stats HSV (9)
    "color_h_mean",
    "color_h_std",
    "color_h_skew",
    "color_s_mean",
    "color_s_std",
    "color_s_skew",
    "color_v_mean",
    "color_v_std",
    "color_v_skew",
    # Edge Density (1)
    "edge_density",
    # Noise Residual Variance (3)
    "noise_residual_var_b",
    "noise_residual_var_g",
    "noise_residual_var_r",
]


def compute_ela(image_bgr: np.ndarray, quality: int = 90, scale: int = 15):
    """
    Computes Error Level Analysis (ELA).
    Saves image at a known JPEG quality, computes absolute difference,
    and amplifies discrepancies to reveal localized compression differences.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_orig = Image.fromarray(image_rgb)

    buffer = io.BytesIO()
    pil_orig.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    pil_recomp = Image.open(buffer)

    diff = ImageChops.difference(pil_orig, pil_recomp)
    enhancer = ImageEnhance.Brightness(diff)
    ela_enhanced = enhancer.enhance(scale)
    ela_np = np.array(ela_enhanced)

    ela_diff_raw = np.array(diff, dtype=np.float32)
    ela_mean = float(np.mean(ela_diff_raw))
    ela_std = float(np.std(ela_diff_raw))
    ela_max = float(np.max(ela_diff_raw))

    ela_bgr = cv2.cvtColor(ela_np, cv2.COLOR_RGB2BGR)

    return {
        "ela_mean": ela_mean,
        "ela_std": ela_std,
        "ela_max": ela_max,
    }, ela_bgr


def compute_fft_features(gray: np.ndarray):
    """
    Calculates 2D Fast Fourier Transform frequency domain statistics.
    Deepfake generators frequently leave tell-tale concentric rings or
    depleted high-frequency power in the frequency spectrum.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx**2 + cy**2)

    high_freq_mask = r > (0.35 * max_r)
    total_energy = np.sum(np.abs(fshift) ** 2) + 1e-9
    high_freq_energy = np.sum(np.abs(fshift[high_freq_mask]) ** 2)
    high_freq_ratio = float(high_freq_energy / total_energy)

    spectral_mean = float(np.mean(magnitude_spectrum))
    spectral_std = float(np.std(magnitude_spectrum))

    return {
        "fft_high_freq_ratio": high_freq_ratio,
        "fft_spectral_mean": spectral_mean,
        "fft_spectral_std": spectral_std,
    }


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian operator.
    Standard metric for blur/sharpness detection. Synthetic blending often
    produces distinct boundary smoothing.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_lbp_features(gray: np.ndarray, num_bins: int = 10):
    """
    Computes standard 8-neighbor Local Binary Pattern (LBP) texture representation.
    Extracts micro-texture statistics without relying on external heavy dependencies.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return {"lbp_mean": 0.0, "lbp_std": 0.0, **{f"lbp_bin_{i}": 0.0 for i in range(num_bins)}}

    center = gray[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)

    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1)
    ]

    for bit, (dy, dx) in enumerate(shifts):
        neighbor = gray[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= ((neighbor >= center) << bit).astype(np.uint8)

    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, 256), density=True)

    results = {
        "lbp_mean": float(np.mean(lbp)),
        "lbp_std": float(np.std(lbp)),
    }
    for i in range(num_bins):
        results[f"lbp_bin_{i}"] = float(hist[i])

    return results


def compute_color_stats(image_bgr: np.ndarray):
    """
    Extracts mean, standard deviation, and skewness for RGB and HSV channels.
    Deepfake generators frequently introduce color saturation and chrominance shifts.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    feats = {}
    channel_data = {
        "color_r": image_rgb[:, :, 0],
        "color_g": image_rgb[:, :, 1],
        "color_b": image_rgb[:, :, 2],
        "color_h": image_hsv[:, :, 0],
        "color_s": image_hsv[:, :, 1],
        "color_v": image_hsv[:, :, 2],
    }

    for name, channel in channel_data.items():
        flat = channel.ravel().astype(np.float64)
        feats[f"{name}_mean"] = float(np.mean(flat))
        feats[f"{name}_std"] = float(np.std(flat))
        # Skewness calculation with zero-division safety
        std_val = feats[f"{name}_std"]
        if std_val > 1e-5:
            sk = float(stats.skew(flat))
            feats[f"{name}_skew"] = 0.0 if np.isnan(sk) else sk
        else:
            feats[f"{name}_skew"] = 0.0

    return feats


def compute_edge_density(gray: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> float:
    """
    Computes ratio of edge pixels using Canny edge detection.
    """
    edges = cv2.Canny(gray, threshold1, threshold2)
    return float(np.count_nonzero(edges) / (edges.size + 1e-9))


def compute_noise_residual_variance(image_bgr: np.ndarray):
    """
    Extracts high-frequency sensor noise residuals by subtracting a Gaussian-smoothed
    version from the input image. Camera sensors produce characteristic PRNU sensor
    noise, which is wiped out or distorted by AI upsamplers and GANs.
    """
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), sigmaX=1.0)
    residual = image_bgr.astype(np.float32) - blurred.astype(np.float32)

    var_b = float(np.var(residual[:, :, 0]))
    var_g = float(np.var(residual[:, :, 1]))
    var_r = float(np.var(residual[:, :, 2]))

    return {
        "noise_residual_var_b": var_b,
        "noise_residual_var_g": var_g,
        "noise_residual_var_r": var_r,
    }


def extract_features(image_bgr: np.ndarray):
    """
    Main feature extraction function.
    Given a cropped face image (typically 224x224 BGR), computes all 38 forensic
    features, produces the ELA visual image, and creates normalized radar metrics.

    Returns:
        feature_vector: np.ndarray shape (38,)
        feature_dict: dict of all extracted raw metrics
        radar_metrics: dict normalized [0.0, 1.0] for radar chart
        ela_bgr: visual ELA difference image (BGR)
    """
    if image_bgr.shape[0] != 224 or image_bgr.shape[1] != 224:
        image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    ela_feats, ela_bgr = compute_ela(image_bgr)
    fft_feats = compute_fft_features(gray)
    lap_var = compute_laplacian_variance(gray)
    lbp_feats = compute_lbp_features(gray)
    color_feats = compute_color_stats(image_bgr)
    edge_density = compute_edge_density(gray)
    noise_feats = compute_noise_residual_variance(image_bgr)

    # Consolidate into single dictionary
    feature_dict = {
        **ela_feats,
        **fft_feats,
        "laplacian_variance": lap_var,
        **lbp_feats,
        **color_feats,
        "edge_density": edge_density,
        **noise_feats,
    }

    # Build strict feature vector matching FEATURE_NAMES order
    feature_vector = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)

    # Normalize metrics to 0-1 range for the forensic radar chart
    # (Values calibrated against real vs deepfake facial distributions)
    radar_metrics = {
        "ela": float(np.clip(ela_feats["ela_mean"] / 8.0, 0.05, 0.98)),
        "frequency": float(np.clip(fft_feats["fft_high_freq_ratio"] * 3.5, 0.05, 0.98)),
        "sharpness": float(np.clip(lap_var / 500.0, 0.05, 0.98)),
        "texture": float(np.clip(lbp_feats["lbp_std"] / 45.0, 0.05, 0.98)),
        "noise": float(np.clip((noise_feats["noise_residual_var_g"]) / 30.0, 0.05, 0.98)),
        "edges": float(np.clip(edge_density * 8.0, 0.05, 0.98)),
    }

    return feature_vector, feature_dict, radar_metrics, ela_bgr
