import os
import argparse
import random
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from diffusers import StableDiffusionPipeline, DDIMScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from modules.clip_vit import StyleEncoder


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_style_injected(ldm_model):
    count = 0
    for name, m in ldm_model.named_modules():
        if m.__class__.__name__ == "CrossAttention":
            if hasattr(m, "to_k_injected"):
                for p in m.to_k_injected.parameters():
                    p.requires_grad = True
                count += 1
            if hasattr(m, "to_v_injected"):
                for p in m.to_v_injected.parameters():
                    p.requires_grad = True
                count += 1
    print(f"[unfreeze_style_injected] enabled grad on ~{count} submodules (to_k_injected/to_v_injected)")


def collect_style_injected_params(ldm_model):
    params = []
    names = []
    for name, m in ldm_model.named_modules():
        if m.__class__.__name__ == "CrossAttention":
            if hasattr(m, "to_k_injected"):
                params += list(m.to_k_injected.parameters())
                names.append(f"{name}.to_k_injected")
            if hasattr(m, "to_v_injected"):
                params += list(m.to_v_injected.parameters())
                names.append(f"{name}.to_v_injected")
    return params, names


def pil_to_latents_batch(imgs, pipe, device):
    if isinstance(imgs, list):
        tfm = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        batch = torch.stack([tfm(img) for img in imgs], dim=0).to(device)
    elif isinstance(imgs, torch.Tensor):
        batch = imgs.to(device)
    else:
        raise ValueError("imgs must be list[PIL] or torch.Tensor batch")

    with torch.no_grad():
        encoded = pipe.vae.encode(batch).latent_dist
        latents = encoded.sample()
        latents = latents * 0.18215
    return latents


def get_text_conditioning(ldm_model, prompts, device):
    if hasattr(ldm_model, "get_learned_conditioning"):
        try:
            return ldm_model.get_learned_conditioning(prompts)
        except Exception as e:
            print("warning - couldn't get learned conditioning via ldm_model.get_learned_conditioning:", e)
            return None
    return None


# def print_trainable_params_summary(model, extra_modules=None, top_n=100):
#     total_params = 0
#     trainable = []
#     trainable_params = 0

#     for name, p in model.named_parameters():
#         total_params += p.numel()
#         if p.requires_grad:
#             trainable.append((name, tuple(p.shape), p.numel()))
#             trainable_params += p.numel()

#     if extra_modules is not None:
#         for nm, mod in extra_modules:
#             if mod is model:
#                 continue
#             for name, p in mod.named_parameters():
#                 total_params += p.numel()
#                 full_name = f"{nm}.{name}"
#                 if p.requires_grad:
#                     trainable.append((full_name, tuple(p.shape), p.numel()))
#                     trainable_params += p.numel()

#     print("=== Trainable parameters summary ===")
#     print(f"Total parameters (model + extras): {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,} ({100.0 * trainable_params / max(1,total_params):.4f}%)")
#     print(f"Number of trainable tensors: {len(trainable)}")
#     print("-----------------------------------")

#     for i, (n, shape, cnt) in enumerate(trainable):
#         if i >= top_n:
#             print(f"... (and {len(trainable)-top_n} more trainable tensors)")
#             break
#         print(f"{i+1:03d}. {n:70s} | shape={str(shape):20s} | params={cnt:,}")
#     print("===================================\n")


# def list_modules_with_trainable_params(model):
#     """
#     Prints names of modules that contain at least one trainable parameter.
#     """
#     modules_with_trainable = []
#     for m_name, module in model.named_modules():
#         any_trainable = False
#         for _, p in module.named_parameters(recurse=False):
#             if p.requires_grad:
#                 any_trainable = True
#                 break
#         if any_trainable:
#             modules_with_trainable.append(m_name)
#     print("Modules that have trainable params (top-level module names):")
#     for nm in modules_with_trainable:
#         print(" -", nm)
#     print("Total modules with trainable params:", len(modules_with_trainable))
#     print()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print(f"========== device: {device} ==========")

    # stable diffusion v1.5 parameters
    CONFIG_PATH = args.ldm_config_path
    MODEL_PATH = args.ldm_model_path

    # load model (for DDIM Inversion - diffusion process)
    print("========== [DDIM Inversion] Loading Hugging Face diffusers pipeline for VAE and Inversion... ==========")
    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        original_config_file=CONFIG_PATH,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.eval()
    freeze_module(pipe.vae)

    # load model (for DDIM Sampling - reverse diffusion process (noise prediction))
    print(f"========== [DDIM Sampling] Loading Latent Diffusion model... ==========")
    config = OmegaConf.load(CONFIG_PATH)
    ldm_model = instantiate_from_config(config.model)
    if MODEL_PATH.endswith(".safetensors"):
        sd = load_file(MODEL_PATH, device="cpu")                        # parameters type: ".safetensors" (ONLY)
    else:
        sd = torch.load(MODEL_PATH, map_location="cpu")["state_dict"]   # parameters type: ".ckpt"
    ldm_model.load_state_dict(sd, strict=False)
    ldm_model = ldm_model.to(device)
    ldm_model.eval()
    freeze_module(ldm_model)

    # unfreeze only style attention layer (to_k_injected & to_v_injected)
    unfreeze_style_injected(ldm_model)
    style_injected_params, style_injected_names = collect_style_injected_params(ldm_model)

    # print_trainable_params_summary(ldm_model, extra_modules=[])
    # list_modules_with_trainable_params(ldm_model)

    # load style encoder (CLIP-based) module
    print("========== Loading Style Encoder... ==========")
    style_encoder = StyleEncoder(device=device, sty_alpha=args.sty_alpha, freeze_clip=True)
    style_encoder.to(device)
    freeze_module(style_encoder.clip) 

    # dataset (wikiart) - ImageFolder style
    transform = T.Compose([
        T.Resize((512, 512)),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(args.train_data_dir, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            drop_last=True)

    # optimizer
    trainable = []
    if style_injected_params:
        trainable += style_injected_params
    if len(trainable) == 0:
        raise RuntimeError("No parameters to train! Check that CrossAttention.to_k_injected / to_v_injected exist in your LDM.")
    optimizer = optim.AdamW(trainable, lr = args.lr, weight_decay=args.weight_decay)

    # scheduler alphas (use pipe.scheduler.alphas_cumprod)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
    T_count = alphas_cumprod.shape[0]

    global_step = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(dataloader, desc="train", leave=False)

        for batch_idx, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # 1) encode to latents
            with torch.no_grad():
                latents = pil_to_latents_batch(imgs, pipe, device)

            # 2) sample random timesteps for each sample
            t = torch.randint(low=0, high=T_count, size=(B,), device=device).long()

            # 3) sample noise
            noise = torch.randn_like(latents)

            # 4) compute z_t = sqrt(alpha_cumprod[t]) * latents + sqrt(1 - alpha_cumprod[t]) * noise
            a_t = alphas_cumprod[t].view(B, 1, 1, 1)
            z_t = torch.sqrt(a_t) * latents + torch.sqrt(1 - a_t) * noise
            z_t = z_t.detach().requires_grad_(True)

            # 5) get conditioning (empty prompt) - try get_learned_conditioning if available
            cond = get_text_conditioning(ldm_model, [""] * B, device)

            # 6) get CLIP patch tokens & sty_alpha
            with torch.no_grad():
                injected_features = style_encoder(imgs)

            if isinstance(injected_features, dict):
                for key in ('k', 'v'):
                    if key in injected_features and torch.is_tensor(injected_features[key]):
                        injected_features[key] = injected_features[key].detach().to(device).float().requires_grad_(True)
                injected_features['sty_alpha'] = injected_features['sty_alpha'].detach().to(device).float().requires_grad_(True)

            # 7) forward through UNet / LDM to predict noise
            optimizer.zero_grad()
            ldm_model.train() # enable grad for selected params
            pred = ldm_model.apply_model(z_t, t, cond, injected_features=injected_features)

            loss = F.mse_loss(pred, noise)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                pbar.set_postfix({'loss': float(loss.detach().cpu())})

            if global_step % args.save_interval == 0:
                save_path = os.path.join(args.save_dir, f"model_style{global_step}.safetensors")
                flat_dict = {}
                flat_dict["meta.step"] = torch.tensor(global_step, dtype=torch.long).cpu()
                for name, module in ldm_model.named_modules():
                    if module.__class__.__name__ == "CrossAttention":
                        if hasattr(module, "to_k_injected"):
                            sd = module.to_k_injected.state_dict()
                            for k, v in sd.items():
                                key = f"{name}.to_k_injected.{k}"
                                flat_dict[key] = v.cpu()
                        if hasattr(module, "to_v_injected"):
                            sd = module.to_v_injected.state_dict()
                            for k, v in sd.items():
                                key = f"{name}.to_v_injected.{k}"
                                flat_dict[key] = v.cpu()
                save_file(flat_dict, save_path)
                print("Saved checkpoint (safetensors):", save_path)

            if global_step >= args.max_steps:
                print("Reached max steps. Exiting.")
                return

    print("✅ Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ldm_config_path", type=str, default="models/ldm/stable-diffusion-v1-5/v1-inference.yaml")
    parser.add_argument("--ldm_model_path", type=str, default="models/ldm/stable-diffusion-v1-5/model.safetensors")
    parser.add_argument("--train_data_dir", type=str, required=True) # /data/lfs/sekwang/wikiart
    parser.add_argument("--save_dir", type=str, default="models/ldm/stable-diffusion-v1-5-style/")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--sty_alpha", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    train(args)


"""
CUDA_VISIBLE_DEVICES=1 python train/train_style.py --train_data_dir /data/lfs/sekwang/wikiart
CUDA_VISIBLE_DEVICES=1 nohup python train/train_style.py --train_data_dir /data/lfs/sekwang/wikiart &
"""