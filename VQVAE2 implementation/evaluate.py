import sys
import types

try:
    import _bz2
except ImportError:
    dummy_bz2 = types.ModuleType('_bz2')
    class DummyBZ2Impl:
        def __init__(self, *args, **kwargs): pass
    dummy_bz2.BZ2Compressor = DummyBZ2Impl
    dummy_bz2.BZ2Decompressor = DummyBZ2Impl
    sys.modules['_bz2'] = dummy_bz2

try:
    import _lzma
except ImportError:
    dummy_lzma = types.ModuleType('_lzma')
    class DummyLZMAImpl:
        def __init__(self, *args, **kwargs): pass
    
    dummy_lzma.LZMACompressor = DummyLZMAImpl
    dummy_lzma.LZMADecompressor = DummyLZMAImpl
    dummy_lzma.LZMAError = Exception
    dummy_lzma.LZMAFile = DummyLZMAImpl
    dummy_lzma.FORMAT_AUTO = 0
    dummy_lzma.FORMAT_XZ = 1
    dummy_lzma.FORMAT_ALONE = 2
    dummy_lzma.FORMAT_RAW = 3
    dummy_lzma.CHECK_NONE = 0
    dummy_lzma.CHECK_CRC32 = 1
    dummy_lzma.CHECK_CRC64 = 2
    dummy_lzma.CHECK_SHA256 = 10
    dummy_lzma.FILTERS_MAX = 2**32
    
    dummy_lzma._encode_filter_properties = lambda *args: b''
    dummy_lzma._decode_filter_properties = lambda *args: {}
    dummy_lzma.is_check_supported = lambda *args: False
    
    sys.modules['_lzma'] = dummy_lzma

import os
import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset.mri_dataset import MRIDataset
from model.vqvae2 import VQVAE2

DATA_ROOT = "/scratch/b24cm1068/processed"
SPLIT = "test"   # Change to "val" for final results
CHECKPOINT_PATH = "./results_vqvae2_save/vqvae2_epoch_175.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = f"results/vqvae2_{SPLIT}_eval"

def save_comparison(low, high, recon, diff, idx, psnr, ssim):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(low, cmap="gray"); axs[0].set_title("Input (Low NSA)")
    axs[1].imshow(high, cmap="gray"); axs[1].set_title("Target (High NSA)")
    axs[2].imshow(recon, cmap="gray"); axs[2].set_title(f"VQ-VAE-2 (PSNR: {psnr:.2f})")
    im = axs[3].imshow(diff, cmap="inferno"); axs[3].set_title("Difference")
    plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
    for ax in axs: ax.axis("off")
    plt.suptitle(f"Sample {idx} | SSIM: {ssim:.4f}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"sample_{idx}.png"))
    plt.close()

def to_rgb(x):
    return x.repeat(1, 3, 1, 1)

def main():
    print(f"✅ Evaluating VQ-VAE-2 on {SPLIT} set")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataset = MRIDataset(DATA_ROOT, split=SPLIT)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = VQVAE2(in_ch=4, hidden_ch=128, num_embeddings=512, embedding_dim=64).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt)
    else:
        print("❌ Error: Checkpoint not found.")
        return

    model.eval()

    print("📏 Initializing Metrics (LPIPS, FID)...")
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    
    fid_metric = FrechetInceptionDistance(feature=64).to(DEVICE) 

    total_psnr, total_ssim, total_lpips, count = 0, 0, 0, 0

    print("🚀 Starting Evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            x_in = batch["input_stack"].to(DEVICE)
            low_orig = batch["low_original"].cpu().numpy()[0]
            high_orig = batch["high_original"].cpu().numpy()[0]
            
            x_out, _ = model(x_in)
            
            bands = x_out.cpu().numpy()[0]
            recon = pywt.idwt2((bands[0], (bands[1], bands[2], bands[3])), "haar")
            
            if recon.shape != high_orig.shape: 
                recon = recon[:high_orig.shape[0], :high_orig.shape[1]]
            recon = np.clip(recon, 0, 1)
            
            p = peak_signal_noise_ratio(high_orig, recon, data_range=1.0)
            s = structural_similarity(high_orig, recon, data_range=1.0)
            
            total_psnr += p
            total_ssim += s
            count += 1
            
            recon_tensor = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            high_tensor = torch.from_numpy(high_orig).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            
            recon_lpips = (recon_tensor * 2) - 1
            high_lpips = (high_tensor * 2) - 1
            l_score = lpips_fn(to_rgb(recon_lpips), to_rgb(high_lpips))
            total_lpips += l_score.item()
            
            recon_fid = (recon_tensor * 255).to(torch.uint8)
            high_fid = (high_tensor * 255).to(torch.uint8)
            
            fid_metric.update(to_rgb(high_fid), real=True)
            fid_metric.update(to_rgb(recon_fid), real=False)
            
            if i % 50 == 0:
                diff = np.abs(high_orig - recon)
                save_comparison(low_orig, high_orig, recon, diff, i, p, s)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count
    print("⏳ Computing FID (this may take a moment)...")
    fid_score = fid_metric.compute().item()

    print("\n" + "="*40)
    print(f"📢 FINAL VQ-VAE-2 RESULTS ({SPLIT})")
    print(f"   📈 Avg PSNR : {avg_psnr:.4f} dB  (Higher is better)")
    print(f"   📈 Avg SSIM : {avg_ssim:.4f}     (Higher is better)")
    print(f"   📉 Avg LPIPS: {avg_lpips:.4f}    (Lower is better)")
    print(f"   📉 FID Score: {fid_score:.4f}    (Lower is better)")
    print("="*40)

if __name__ == "__main__":
    main()
