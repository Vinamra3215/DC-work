import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset.mri_dataset import MRIDataset
from model.vqvae import VQVAE

DATA_ROOT = "/scratch/b24cm1068/processed"
SAVE_DIR = "./results"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 100
LR = 1e-4
SAVE_EVERY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {DEVICE}")

train_dataset = MRIDataset(DATA_ROOT, split="train")
val_dataset = MRIDataset(DATA_ROOT, split="val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=1,
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)

model = VQVAE(in_ch=4, hidden_ch=128, num_embeddings=512, embedding_dim=64).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

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
            print("❌ Error: Loss is NaN! Stopping training.")
            exit()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader):.4f}")

    if epoch % SAVE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            try:
                batch = next(iter(val_loader))

                x_in = batch["input_stack"].to(DEVICE)
                low_orig = batch["low_original"].cpu().numpy()[0]
                high_orig = batch["high_original"].cpu().numpy()[0]

                x_out, _ = model(x_in)

                bands = x_out.cpu().numpy()[0]
                LL_pred = bands[0]
                LH_pred = bands[1]
                HL_pred = bands[2]
                HH_pred = bands[3]

                recon = pywt.idwt2(
                    (LL_pred, (LH_pred, HL_pred, HH_pred)),
                    "haar"
                )

                if recon.shape != high_orig.shape:
                    recon = recon[:high_orig.shape[0], :high_orig.shape[1]]

                recon = np.clip(recon, 0, 1)

                psnr = peak_signal_noise_ratio(high_orig, recon, data_range=1.0)
                ssim = structural_similarity(high_orig, recon, data_range=1.0)

                diff_map = np.abs(high_orig - recon)

                fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                axs[0].imshow(low_orig, cmap="gray")
                axs[1].imshow(high_orig, cmap="gray")
                axs[2].imshow(recon, cmap="gray")
                im_diff = axs[3].imshow(diff_map, cmap="inferno")

                for ax in axs:
                    ax.axis("off")

                plt.colorbar(im_diff, ax=axs[3], fraction=0.046, pad=0.04)
                plt.savefig(os.path.join(SAVE_DIR, f"epoch_{epoch}.png"))
                plt.close()

                print(f"📊 Validation: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

            except Exception as e:
                print(f"⚠️ Validation plotting failed: {e}")
