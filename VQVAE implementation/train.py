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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pywt
import matplotlib.pyplot as plt

import lpips
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataset.mri_dataset import MRIDataset
from model.vqvae import VQVAE

DATA_ROOT = "/scratch/b24cm1068/processed"
SAVE_DIR = "./results_vqvae"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 200
LR = 1e-4
SAVE_EVERY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {DEVICE}")

print("📏 Initializing Training Metrics...")
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
lpips_metric = lpips.LPIPS(net='alex').to(DEVICE)

print("📂 Loading dataset...")
train_dataset = MRIDataset(DATA_ROOT, split="train")
val_dataset   = MRIDataset(DATA_ROOT, split="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

print("🏗️ Initializing VQ-VAE (Original)...")
model = VQVAE(in_ch=4, hidden_ch=128, num_embeddings=512, embedding_dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def to_rgb(x):
    return x.repeat(1, 3, 1, 1)

def wavelet_to_image_tensor(bands):
    b, c, h, w = bands.shape
    recons = []
    bands_np = bands.detach().cpu().numpy()
    for i in range(b):
        rec = pywt.idwt2((bands_np[i,0], (bands_np[i,1], bands_np[i,2], bands_np[i,3])), "haar")
        recons.append(rec)
    recons = np.array(recons)
    return torch.from_numpy(recons).unsqueeze(1).float().to(DEVICE)

print("🚀 Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    
    psnr_metric.reset()
    ssim_metric.reset()
    epoch_loss = 0.0
    epoch_lpips = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for batch in pbar:
        x_in = batch["input_stack"].to(DEVICE)
        x_gt = batch["target_stack"].to(DEVICE)

        optimizer.zero_grad()
        
        x_out, vq_loss = model(x_in)

        loss_ll = F.l1_loss(x_out[:, 0, ...], x_gt[:, 0, ...])
        loss_hf = F.mse_loss(x_out[:, 1:, ...], x_gt[:, 1:, ...])
        
        recon_loss = loss_ll + (10.0 * loss_hf)
        loss = recon_loss + 0.25 * vq_loss

        if torch.isnan(loss):
            print("❌ Error: Loss is NaN!")
            exit()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        with torch.no_grad():
            img_pred = wavelet_to_image_tensor(x_out).clamp(0, 1)
            img_gt = wavelet_to_image_tensor(x_gt).clamp(0, 1)

            psnr_metric.update(img_pred, img_gt)
            ssim_metric.update(img_pred, img_gt)
            
            img_pred_norm = (img_pred * 2) - 1
            img_gt_norm = (img_gt * 2) - 1
            lpips_val = lpips_metric(to_rgb(img_pred_norm), to_rgb(img_gt_norm))
            epoch_lpips += lpips_val.mean().item()

    avg_loss = epoch_loss / len(train_loader)
    avg_lpips = epoch_lpips / len(train_loader)
    avg_psnr = psnr_metric.compute().item()
    avg_ssim = ssim_metric.compute().item()

    print(f"Epoch {epoch} Results:")
    print(f"   📉 Loss : {avg_loss:.4f}")
    print(f"   📈 PSNR : {avg_psnr:.2f} dB")
    print(f"   📈 SSIM : {avg_ssim:.4f}")
    print(f"   📉 LPIPS: {avg_lpips:.4f}")

    if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
        save_path = os.path.join(SAVE_DIR, f"vqvae_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved checkpoint: {save_path}")

        model.eval()
        with torch.no_grad():
            try:
                batch = next(iter(val_loader))
                x_in = batch["input_stack"].to(DEVICE)
                high_orig = batch["high_original"].to(DEVICE)

                x_out, _ = model(x_in)

                input_recon_tensor = wavelet_to_image_tensor(x_in).clamp(0, 1)
                input_img_np = input_recon_tensor[0, 0].cpu().numpy()

                target_tensor = high_orig.unsqueeze(1).clamp(0, 1)
                target_img_np = high_orig[0].cpu().numpy()

                recon_tensor = wavelet_to_image_tensor(x_out).clamp(0, 1)
                recon_img_np = recon_tensor[0, 0].cpu().numpy()

                psnr_metric.reset()
                ssim_metric.reset()
                val_psnr = psnr_metric(recon_tensor, target_tensor).item()
                val_ssim = ssim_metric(recon_tensor, target_tensor).item()
                val_lpips = lpips_metric(to_rgb((recon_tensor*2)-1), to_rgb((target_tensor*2)-1)).item()

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(input_img_np, cmap='gray'); axs[0].set_title("Input (NSA 1)"); axs[0].axis('off')
                axs[1].imshow(target_img_np, cmap='gray'); axs[1].set_title("Target (NSA 3)"); axs[1].axis('off')
                axs[2].imshow(recon_img_np, cmap='gray'); axs[2].set_title("Generated (VQVAE)"); axs[2].axis('off')

                plt.suptitle(f"Epoch {epoch} | PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}", fontsize=14)
                plt.tight_layout()
                save_path = os.path.join(SAVE_DIR, f"visual_{epoch}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"🖼️ Saved visualization: {save_path}")

            except Exception as e:
                print(f"⚠️ Visual failed: {e}")
