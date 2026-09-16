"""
Generate images from a trained DCGAN Generator.

Usage:
    python generate_images.py --generator_path generator.pth

    # Optional: add --discriminator_path to display the average
    # D(G(z)) probability (estimated realism) of the generated images:
    python generate_images.py --generator_path generator.pth --discriminator_path discriminator.pth

Note: this script is designed for a Generator producing 64x64 images.
The Discriminator is entirely optional — if provided, the script simply
prints the average D(G(z)) probability over the generated images,
without influencing the generation itself.
"""

import argparse
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Architectures (identical to the ones used during training)
# ---------------------------------------------------------------------------

class MyGenerator(nn.Module):
    """
    DCGAN Generator producing 64x64 output (5 upsampling steps: 4->8->16->32->64).

    To train/generate at a different resolution (e.g. 128x128), just add
    one more layer (ConvTranspose2d + BatchNorm2d + ReLU) before the final
    layer, which doubles the resolution each time:
        4 -> 8 -> 16 -> 32 -> 64 -> 128 (add 1 layer for 128x128)
        ... -> 128 -> 256 (add another layer for 256x256)
    You then also need to adjust kernel_num*32, etc. in the new layer and
    retrain the model: a generator trained at 64x64 cannot load a
    state_dict meant for a different resolution (size mismatch).
    """
    def __init__(self, noises_size=100, channels=3, kernel_num=32):
        super(MyGenerator, self).__init__()
        self.main = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(noises_size, kernel_num * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(kernel_num * 16),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(kernel_num * 16, kernel_num * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(kernel_num * 8),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(kernel_num * 8, kernel_num * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(kernel_num * 4),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(kernel_num * 4, kernel_num * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(kernel_num * 2),
            nn.ReLU(True),

            # 32x32 -> 64x64 (final layer)
            nn.ConvTranspose2d(kernel_num * 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class MyDiscriminator(nn.Module):
    """
    DCGAN Discriminator for a 64x64 input (5 downsampling steps:
    64->32->16->8->4->1). Can use a different kernel_num than the
    Generator (they are two independently trained networks).
    """
    def __init__(self, channels=3, kernel_num=64):
        super(MyDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            nn.utils.spectral_norm(nn.Conv2d(channels, kernel_num, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.utils.spectral_norm(nn.Conv2d(kernel_num, kernel_num * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.utils.spectral_norm(nn.Conv2d(kernel_num * 2, kernel_num * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.utils.spectral_norm(nn.Conv2d(kernel_num * 4, kernel_num * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 (final)
            nn.utils.spectral_norm(nn.Conv2d(kernel_num * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate images from a trained DCGAN Generator.")
    parser.add_argument("--generator_path", type=str, required=True,
                         help="Path to the Generator's .pth file.")
    parser.add_argument("--discriminator_path", type=str, default=None,
                         help="Path to the Discriminator's .pth file (optional, used only to score the generated images).")
    parser.add_argument("--output_dir", type=str, default="./generated_images",
                         help="Output directory for the generated images.")
    parser.add_argument("--num_images", type=int, default=16,
                         help="Number of images to generate.")
    parser.add_argument("--noises_size", type=int, default=100,
                         help="Size of the latent noise vector (must match training).")
    parser.add_argument("--kernel_num", type=int, default=32,
                         help="Base number of filters for the Generator (must EXACTLY match training).")
    parser.add_argument("--disc_kernel_num", type=int, default=64,
                         help="Base number of filters for the Discriminator (may differ from the Generator's).")
    parser.add_argument("--grid", action="store_true",
                         help="Also save a grid image combining all generated images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load the Generator ---
    generator = MyGenerator(noises_size=args.noises_size, channels=3, kernel_num=args.kernel_num).to(device)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    generator.eval()
    print(f"Generator loaded from: {args.generator_path}")

    # --- Optionally load the Discriminator (to score the images) ---
    discriminator = None
    if args.discriminator_path:
        try:
            discriminator = MyDiscriminator(channels=3, kernel_num=args.disc_kernel_num).to(device)
            discriminator.load_state_dict(torch.load(args.discriminator_path, map_location=device))
            discriminator.eval()
            print(f"Discriminator loaded from: {args.discriminator_path}")
        except RuntimeError as e:
            print("Warning: the Discriminator could not be loaded (incompatible architecture).")
            print("Image generation will continue normally, without the D(G(z)) score.")
            print(f"Details: {e}")
            discriminator = None

    # --- Generation ---
    with torch.no_grad():
        noise = torch.randn(args.num_images, args.noises_size, 1, 1, device=device)
        fake_images = generator(noise)  # output in [-1, 1] (Tanh)
        images_01 = (fake_images + 1) / 2.0  # rescale to [0, 1] for saving

        scores = None
        if discriminator is not None:
            scores = discriminator(fake_images).cpu()

    to_pil = transforms.ToPILImage()
    for i in range(args.num_images):
        img = to_pil(images_01[i].cpu())
        filename = f"image_{i:03d}.png"
        img.save(os.path.join(args.output_dir, filename))

    print(f"{args.num_images} images saved to: {args.output_dir}")

    if scores is not None:
        print(f"Average D(G(z)) probability over the {args.num_images} generated images: {scores.mean().item():.4f} / target: 0.5")

    # --- Optional summary grid ---
    if args.grid:
        grid = make_grid(images_01.cpu(), nrow=int(args.num_images ** 0.5) or 1, padding=2)
        grid_path = os.path.join(args.output_dir, "summary_grid.png")
        save_image(grid, grid_path)
        print(f"Summary grid saved: {grid_path}")


if __name__ == "__main__":
    main()