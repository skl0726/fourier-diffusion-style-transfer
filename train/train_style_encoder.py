import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from safetensors.torch import save_file, load_file 

from diffusers import StableDiffusionPipeline, DDIMScheduler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from modules.clip_vit import clip_vit_encoder, StyleSelfAttention


def pil_batch_to_latents(imgs_tensor, pipe, device):
    # imgs_tensor: (B,3,H,W) tensor in [-1,1] (same as inference transform)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(imgs_tensor).latent_dist
        z0 = latent_dist.sample()
        z0 = z0 * 0.18215
    return z0


def train(args):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"===== device: {device} =====")

    # load model (for DDIM Inversion)
    print("===== Loading Hugging Face diffusers pipeline for VAE and Inversion... =====")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # load model (for DDIM Sampling)
    model_path = args.ldm_model_path
    print(f"===== Loading Latent Diffusion model from: {model_path} =====")
    config = OmegaConf.load(args.ldm_config_path)
    ldm_model = instantiate_from_config(config.model)
    if model_path.endswith(".safetensors"): # parameters type: ".safetensors"
        from safetensors.torch import load_file
        sd = load_file(model_path, device="cpu")
    else:                                   # parameters type: ".ckpt"
        sd = torch.load(model_path, map_location="cpu")["state_dict"]
    ldm_model.load_state_dict(sd, strict=False)
    ldm_model = ldm_model.to(device).eval()
    sampler = DDIMSampler(ldm_model)

    # instantiate style encoder (trainable)
    style_encoder = StyleSelfAttention(dim=args.dim,
                                       nhead=args.nhead,
                                       nlayers=args.nlayers).to(device)
    style_encoder.train()

    # dataset + dataloader
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # scheduler / alphas
    scheduler = pipe.scheduler # DDIMScheduler
    alphas_cumprod = torch.tensor(scheduler.alphas_cumprod, device=device, dtype=torch.float32)
    T_max = alphas_cumprod.shape[0]

    optimizer = torch.optim.AdamW(style_encoder.parameters(), lr=args.lr)

    global_step = 0
    pbar = tqdm(total=args.max_steps)
    while global_step < args.max_steps:
        for imgs, _ in loader:
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]

            # 1) encode to latent z0
            z0 = pil_batch_to_latents(imgs, pipe, device=device)

            # 2) sample timesteps uniformly
            timesteps = torch.randint(0, T_max, (batch_size,), device=device).long()

            # 3) sample noise and form z_t using alphas_cumprod
            noise = torch.randn_like(z0)
            alpha_t = alphas_cumprod[timesteps].view(batch_size, 1, 1, 1)
            z_t = torch.sqrt(alpha_t) * z0 + torch.sqrt(1.0 - alpha_t) * noise

            # 4) get raw tokens from frozen clip vit encoder and pass through trainable StyleSelfAttention
            with torch.no_grad():
                raw_tokens = clip_vit_encoder(imgs,
                                              device=device,
                                              out_dim=args.dim,
                                              max_tokens=args.max_tokens)
            style_tokens = style_encoder(raw_tokens)

            # 5) normalize tokens
            # style_tokens = style_tokens / (style_tokens.std(dim=-1, keepdim=True) + 1e-6)
            with torch.no_grad():
                sample_text = ldm_model.get_learned_conditioning([""])
                text_norm = sample_text.norm(dim=-1).mean()
                style_norm = style_tokens.norm(dim=-1).mean()
            scale = (text_norm / (style_norm + 1e-6)).detach()
            style_tokens = style_tokens * scale

            num_layers = 25
            cond = {"c_crossattn": [style_tokens] * num_layers}

            # 6) predict noise
            pred = sampler.model.apply_model(z_t, timesteps, cond)
                
            # 7) compute MSE loss between pred and true noise
            loss = torch.nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            for p in sampler.model.parameters():
                p.grad = None
            # grad_style = torch.autograd.grad(loss, style_tokens, retain_graph=False, allow_unused=False)[0]
            # style_tokens.backward(grad_style)
            torch.nn.utils.clip_grad_norm_(style_encoder.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            pbar.update(1)
            pbar.set_description(f"step {global_step} loss {loss.item():.6f}")

            if global_step % args.save_every == 0:
                # torch.save(style_encoder.state_dict(), os.path.join(args.save_dir, f"style_encoder_{global_step}.pt"))
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"style_encoder_{global_step}.safetensors")
                save_file(style_encoder.state_dict(), save_path)

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break

    pbar.close()
    # torch.save(style_encoder.state_dict(), os.path.join(args.save_dir, "style_encoder_final.pt"))
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"style_encoder_{global_step}.safetensors")
    save_file(style_encoder.state_dict(), save_path)
    print("===== ✅ Training done. =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_config_path", type=str, default="models/ldm/stable-diffusion-v1-5/v1-inference.yaml")
    parser.add_argument("--ldm_model_path", type=str, default="models/ldm/stable-diffusion-v1-5/v1-5-pruned.safetensors")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--max_tokens", type=int, default=77)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="models/attention/style-encoder")
    parser.add_argument("--data_root", type=str, required=True) # /data/lfs/sekwang/wikiart


    args = parser.parse_args()
    train(args)


"""
python train/train_style_encoder.py --data_root dataset/wikiart
CUDA_VISIBLE_DEVICES=5 python train/train_style_encoder.py --data_root /data/lfs/sekwang/wikiart
CUDA_VISIBLE_DEVICES=5 nohup python train/train_style_encoder.py --data_root /data/lfs/sekwang/wikiart &

## stage 1 (training style encoder) ##
"ldm의 u-net이 이해할 수 있는 style condition representation을 학습 (StyleSelfAttention 학습)"

- 모델 파라미터 pt / ckpt 말고 safetensors로 만들 수 있는지?
- 기존의 학습된 파라미터를 불러와서 다시 학습할 수 있는지?
- overfitting 방지 어떻게 하는지?
- 이 코드가 실제로 diffusion training에 해당하는지 검증?
- denoising u-net의 파라미터를 학습하지 않으면 애초에 효과가 있는 건지 확인?
- (img2img_style에서 현재는 StyleSelfAttention에 파라미터 적용이 안 되었는데 파라미터 불러올 수 있도록 하기)
"""