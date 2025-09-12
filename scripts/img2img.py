import argparse, os, torch, numpy as np
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_ldm(config_path, ckpt_path, device):
    print(f"[load] checkpoint: {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]

    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model


def pil_to_latent(model, pil, device):
    tfm = T.Compose([
        T.Resize((512, 512), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3) # -> [-1,1]
    ])
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        posterior = model.encode_first_stage(x)
        z = posterior.sample() * 0.18215
    return z


def latent_to_pil(model, z):
    z = z / 0.18215
    with torch.no_grad():
        img = model.decode_first_stage(z)
        img = torch.clamp((img + 1) / 2, 0, 1).cpu()[0]
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def ddim_invert_latents(model, sampler, latents_0, prompt, ddim_steps, device):
    """
    DDIM Inversion: original latent -> noise (forward process)
    """
    print(f"[inversion] Starting DDIM inversion for {ddim_steps} steps...")
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0)
    
    # text encoding
    cond = model.get_learned_conditioning([prompt])
    
    latents = latents_0.clone()
    intermediates = [latents.clone()]
    
    # forward inversion process
    for i in tqdm(range(ddim_steps), desc="DDIM Inversion"):
        t_curr = sampler.ddim_timesteps[i]
        
        # float 값을 tensor로 변환
        alpha_curr = sampler.ddim_alphas[i]
        if i < ddim_steps - 1:
            alpha_next = sampler.ddim_alphas[i + 1]
        else:
            alpha_next = 0.0
        
        # device와 dtype 일치시키기
        alpha_curr = torch.tensor(alpha_curr, device=device, dtype=latents.dtype)
        alpha_next = torch.tensor(alpha_next, device=device, dtype=latents.dtype)
        
        t_tensor = torch.tensor([t_curr], device=device, dtype=torch.long)
        noise_pred = model.apply_model(latents, t_tensor, cond)
        
        # DDIM inversion step
        latents = (
            torch.sqrt(alpha_next) * (latents - torch.sqrt(1 - alpha_curr) * noise_pred) / torch.sqrt(alpha_curr) +
            torch.sqrt(1 - alpha_next) * noise_pred
        )
        
        intermediates.append(latents.clone())
    
    return intermediates

@torch.no_grad()
def ddim_sample_from_inverted(model, sampler, start_latents, start_step, prompt, ddim_steps, ddim_eta, device):
    """
    DDIM sampling: inverted latent (noise) -> denoising
    """
    print(f"[sampling] Starting DDIM sampling from step {start_step}...")
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)
    
    # text encoding
    cond = model.get_learned_conditioning([prompt])
    
    # DDIM sampling
    samples, _ = sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        batch_size=1,
        shape=start_latents.shape[1:],
        verbose=True,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        eta=ddim_eta,
        x_T=start_latents,
        start_step=start_step
    )
    
    return samples


def main(opt):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"========== device: {device} ==========")
    
    model = load_ldm(
        "models/ldm/stable-diffusion-v1/v1-inference.yaml",
        "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        device
    )

    sampler = PLMSSampler(model) if opt.plms else DDIMSampler(model)
    
    # 1) Encode input PNG -> latent z0
    init_pil = Image.open(opt.input).convert("RGB")
    z0 = pil_to_latent(model, init_pil, device)
    
    # 2) DDIM Inversion
    prompt = opt.prompt if hasattr(opt, 'prompt') and opt.prompt else "a photograph"
    
    inverted_latents = ddim_invert_latents(
        model=model,
        sampler=sampler,
        latents_0=z0,
        prompt=prompt,
        ddim_steps=opt.ddim_steps,
        device=device
    )
    
    total_steps = len(inverted_latents) - 1
    start_step = int(opt.strength * total_steps)
    start_step = max(0, min(total_steps-1, start_step))
    
    print(f"[reconstruction] strength={opt.strength} -> start_step={start_step}/{total_steps}")
    
    # 3) DDIM sampling
    start_latents = inverted_latents[start_step]
    
    reconstructed = ddim_sample_from_inverted(
        model=model,
        sampler=sampler,
        start_latents=start_latents,
        start_step=start_step,
        prompt=prompt,
        ddim_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        device=device
    )
    
    out_pil = latent_to_pil(model, reconstructed)
    os.makedirs(opt.outdir, exist_ok=True)
    save_path = os.path.join(opt.outdir, "img2img_with_inversion.png")
    out_pil.save(save_path)
    
    print(f"[done] saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="input PNG/JPG (ideally 512x512)")
    parser.add_argument("--prompt", type=str, default="a photograph",
                        help="text prompt for inversion and reconstruction")
    parser.add_argument("--strength", type=float, default=0.3,
                        help="0=no change, 1=full inversion (lower values preserve more original content)")
    parser.add_argument("--ddim_steps", type=int, default=100,
                        help="total DDIM timesteps (more steps = better quality)")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta (0=deterministic)")
    parser.add_argument("--plms", action="store_true",
                        help="use PLMS sampler instead of DDIM")
    parser.add_argument("--outdir", default="outputs/img2img_inversion")
    
    opt = parser.parse_args()
    main(opt)


"""
사용 예시:
python scripts/img2img.py \
    --input images/inputs/input.png \
    --prompt "a photograph" \
    --strength 0.2 \
    --ddim_steps 100 \
    --ddim_eta 0.0 \
    --outdir images/outputs/img2img
"""
# 'strength'만큼 DDIM forward noising(x_0 -> x_t)