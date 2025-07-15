import os
from os.path import join as opj
from omegaconf import OmegaConf
import argparse

import cv2
import numpy as np
import torch

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img
import subprocess

def imread(p, h, w, is_mask=False, in_inverse_mask=False, cloth_mask_check=False):
    """Load and preprocess image"""
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {p}")

    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
        # Ensure we have 3 channels for RGB images
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w, h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:, :, None]  # Add channel dimension
        if cloth_mask_check and img.sum() < 30720*4:
            img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
    return img

def load_single_sample(data_root_dir, img_name, cloth_name, img_H, img_W, data_type="test"):
    """Load a single image-cloth pair from the dataset"""

    # Load agnostic image (person without clothes)
    agn_path = opj(data_root_dir, data_type, "agnostic-v3.2", img_name)
    agn = imread(agn_path, img_H, img_W)

    # Load agnostic mask
    agn_mask_path = opj(data_root_dir, data_type, "agnostic-mask", img_name.replace(".jpg", "_mask.png"))
    agn_mask = imread(agn_mask_path, img_H, img_W, is_mask=True, in_inverse_mask=True)

    # Load cloth image
    cloth_path = opj(data_root_dir, data_type, "cloth", cloth_name)
    cloth = imread(cloth_path, img_H, img_W)

    # Load cloth mask
    cloth_mask_path = opj(data_root_dir, data_type, "cloth-mask", cloth_name)
    cloth_mask = imread(cloth_mask_path, img_H, img_W, is_mask=True, cloth_mask_check=True)

    # Load original image
    image_path = opj(data_root_dir, data_type, "image", img_name)
    image = imread(image_path, img_H, img_W)

    # Load densepose
    densepose_path = opj(data_root_dir, data_type, "image-densepose", img_name)
    image_densepose = imread(densepose_path, img_H, img_W)

    # For test data, we don't have ground truth warped cloth mask
    gt_cloth_warped_mask = np.zeros_like(agn_mask)

    # Convert to tensors with the format the model expects: [batch, height, width, channels]
    batch = {
        'agn': torch.from_numpy(agn).unsqueeze(0).float(),
        'agn_mask': torch.from_numpy(agn_mask).unsqueeze(0).float(),
        'cloth': torch.from_numpy(cloth).unsqueeze(0).float(),
        'cloth_mask': torch.from_numpy(cloth_mask).unsqueeze(0).float(),
        'image': torch.from_numpy(image).unsqueeze(0).float(),
        'image_densepose': torch.from_numpy(image_densepose).unsqueeze(0).float(),
        'gt_cloth_warped_mask': torch.from_numpy(gt_cloth_warped_mask).unsqueeze(0).float(),
        'txt': "",
        'img_fn': img_name,
        'cloth_fn': cloth_name
    }
    # Debug: Print tensor shapes to verify correctness
    print("Tensor shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    return batch

def build_args():
    parser = argparse.ArgumentParser(description="Single image inference for StableVITON")
    parser.add_argument("--config_path", type=str, help="Path to config file",
                        default=r"D:\phuong\StableVITON\configs\VITONHD.yaml")
    parser.add_argument("--model_load_path", type=str, help="Path to model checkpoint",
                        default=r"D:\phuong\StableVITON\VITONHD.ckpt")
    parser.add_argument("--data_root_dir", type=str, default="./zalando-hd-resized",
                       help="Root directory of the dataset")
    parser.add_argument("--img_name", type=str, required=True,
                       help="Name of the person image (e.g., '000001_0.jpg')")
    parser.add_argument("--cloth_name", type=str, required=True,
                       help="Name of the cloth image (e.g., '000001_1.jpg')")
    parser.add_argument("--save_dir", type=str, default="./single_samples",
                       help="Directory to save the output")
    parser.add_argument("--output_name", type=str, default="output_image.jpg",
                       help="Custom output filename (default: img_name_cloth_name.jpg)")
    parser.add_argument("--data_type", type=str, default="test", choices=["train", "test"],
                       help="Dataset split to use")
    parser.add_argument("--repaint", action="store_true",
                       help="Use repainting to preserve original person regions")

    # Model parameters
    parser.add_argument("--denoise_steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--img_H", type=int, default=512, help="Image height")
    parser.add_argument("--img_W", type=int, default=384, help="Image width")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter")

    args = parser.parse_args()
    return args

@torch.no_grad()
def main(args):
    # Load configuration
    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    # Load model
    print(f"Loading model from {args.model_load_path}")
    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp, strict=False)
    model = model.cuda()
    model.eval()

    # Initialize sampler
    sampler = PLMSSampler(model)

    # Load single sample
    print(f"Loading sample: person={args.img_name}, cloth={args.cloth_name}")
    try:
        batch = load_single_sample(
            args.data_root_dir,
            args.img_name,
            args.cloth_name,
            args.img_H,
            args.img_W,
            args.data_type
        )
    except FileNotFoundError as e:
        print(f"Error loading sample: {e}")
        print("Please check that the image and cloth names exist in the dataset.")
        return

    # Move batch tensors to GPU
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()

    # Prepare for inference
    z, c = model.get_input(batch, params.first_stage_key)
    bs = z.shape[0]
    c_crossattn = c["c_crossattn"][0][:bs]

    if c_crossattn.ndim == 4:
        c_crossattn = model.get_learned_conditioning(c_crossattn)
        c["c_crossattn"] = [c_crossattn]

    uc_cross = model.get_unconditional_conditioning(bs)
    uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
    uc_full["first_stage_cond"] = c["first_stage_cond"]

    sampler.model.batch = batch

    # Sample
    print("Starting inference...")
    shape = (4, args.img_H//8, args.img_W//8)
    ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
    start_code = model.q_sample(z, ts)

    samples, _, _ = sampler.sample(
        args.denoise_steps,
        bs,
        shape,
        c,
        x_T=start_code,
        verbose=False,
        eta=args.eta,
        unconditional_conditioning=uc_full,
    )

    # Decode and save result
    x_samples = model.decode_first_stage(samples)
    x_sample = x_samples[0]
    x_sample_img = tensor2img(x_sample)  # [0, 255]

    # Apply repainting if requested
    if args.repaint:
        print("Applying repainting...")
        repaint_agn_img = np.uint8((batch["image"][0].cpu().numpy() + 1) / 2 * 255)  # [0,255]
        repaint_agn_mask_img = batch["agn_mask"][0].cpu().numpy()  # 0 or 1
        x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1 - repaint_agn_mask_img)
        x_sample_img = np.uint8(x_sample_img)

    # Save result
    os.makedirs(args.save_dir, exist_ok=True)
    if args.output_name:
        output_filename = args.output_name
        if not output_filename.endswith(('.jpg', '.png')):
            output_filename += '.jpg'
    else:
        img_base = args.img_name.split('.')[0]
        cloth_base = args.cloth_name.split('.')[0]
        output_filename = f"{img_base}_{cloth_base}.jpg"

    output_path = opj(args.save_dir, output_filename)
    cv2.imwrite(output_path, x_sample_img[:, :, ::-1])  # Convert RGB to BGR for cv2

    print(f"Result saved to: {output_path}")

# def inference_streamlit(args, input_img, input_cloth, output_inference):
#     input_img = args.img_name
#     input_cloth = args.cloth_name
#     output_name = args.output_name
#     result = subprocess.run([
#         "python", "single_inference.py",
#         "--img_name", input_img,
#         "--cloth_name", input_cloth,
#         "--repaint"
#     ])
#     return input_img, input_cloth, output_name


if __name__ == "__main__":
    args = build_args()
    main(args)
    # inference_streamlit(input_img, input_cloth, output_inference)