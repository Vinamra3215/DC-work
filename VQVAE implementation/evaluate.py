import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(gt, pred):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, data_range=1.0)
    return psnr, ssim
