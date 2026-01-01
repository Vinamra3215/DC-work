import os
import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --- SPECIFIC IMPORTS ---
from dataset.mri_dataset import MRIDataset
from model.vqvae2 import VQVAE2

# --- CONFIGURATION ---
DATA_ROOT = "/scratch/b24cm1068/processed"
SPLIT = "val"  # Change to "test" for final results
CHECKPOINT_PATH = "./results_vqvae2/vqvae2_epoch_175.pth" # Update with your best VQVAE2 model path
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

def main():
    print(f"✅ Evaluating VQ-VAE-2 on {SPLIT} set")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset
    dataset = MRIDataset(DATA_ROOT, split=SPLIT)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 2. Model
    model = VQVAE2(in_ch=4, hidden_ch=128, num_embeddings=512, embedding_dim=64).to(DEVICE)
    
    # 3. Load Weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt)
    else:
        print("❌ Error: Checkpoint not found.")
        return

    model.eval()
    total_psnr, total_ssim, count = 0, 0, 0

    print("🚀 Starting...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            x_in = batch["input_stack"].to(DEVICE)
            low = batch["low_original"].cpu().numpy()[0]
            high = batch["high_original"].cpu().numpy()[0]
            
            # Predict
            x_out, _ = model(x_in)
            
            # Reconstruct Wavelets
            bands = x_out.cpu().numpy()[0]
            recon = pywt.idwt2((bands[0], (bands[1], bands[2], bands[3])), "haar")
            
            # Fix Dimensions & Clip
            if recon.shape != high.shape: recon = recon[:high.shape[0], :high.shape[1]]
            recon = np.clip(recon, 0, 1)
            
            # Metrics
            p = peak_signal_noise_ratio(high, recon, data_range=1.0)
            s = structural_similarity(high, recon, data_range=1.0)
            
            total_psnr += p
            total_ssim += s
            count += 1
            
            # Save visual every 50 images
            if i % 50 == 0:
                diff = np.abs(high - recon)
                save_comparison(low, high, recon, diff, i, p, s)

    print("\n" + "="*30)
    print(f"📢 VQ-VAE-2 RESULTS ({SPLIT})")
    print(f"Avg PSNR: {total_psnr/count:.4f}")
    print(f"Avg SSIM: {total_ssim/count:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()