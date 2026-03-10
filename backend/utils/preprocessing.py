"""Frame preprocessing pipeline for optimal model inference"""
import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple = (640, 640),
    normalize: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Resize with aspect ratio preservation
    2. CLAHE enhancement for low-light conditions  
    3. Noise reduction
    4. Normalization
    """
    if frame is None or frame.size == 0:
        return np.zeros((*target_size, 3), dtype=np.uint8)

    # Step 1: Resize maintaining aspect ratio (letterbox)
    h, w = frame.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    padded = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # Step 2: CLAHE for contrast enhancement
    lab = cv2.cvtColor(padded, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3: Gaussian denoising
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return denoised


def extract_texture_features(frame: np.ndarray) -> dict:
    """Extract texture statistics useful for quality assessment"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Local Binary Pattern approximation
    edges = cv2.Canny(gray, 50, 150)
    
    # Frequency domain analysis
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    return {
        "mean_brightness": float(np.mean(gray)),
        "std_brightness": float(np.std(gray)),
        "edge_density": float(np.mean(edges)),
        "texture_entropy": float(np.std(magnitude)),
        "uniformity": float(1.0 - np.std(gray) / (np.mean(gray) + 1e-6))
    }
