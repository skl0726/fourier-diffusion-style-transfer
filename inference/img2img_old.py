import argparse, os, torch, numpy as np
from PIL import Image
from torchvision import transforms as T
from omegaconf import OmegaConf
from tqdm import tqdm
from safetensors.torch import load_file

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_ldm(config_path, ckpt_path, device):
    print(f"[load] checkpoint: {ckpt_path}")
    # pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # sd = pl_sd["state_dict"]

    if ckpt_path.endswith(".safetensors"):
        sd = load_file(ckpt_path)
    else:
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = obj["state_dict"] if "state_dict" in obj else obj

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
def ddim_invert_latents(model, 
                        sampler, 
                        latents_0, 
                        prompt, 
                        negative_prompt, 
                        guidance_scale, 
                        ddim_steps, 
                        device):
    """
    Deterministic DDIM inversion with CFG alignment:
    x_t -> x_{t+1} using t_next for epsilon prediction and matching (alpha_t, alpha_{t+1}).
    """
    print(f"[inversion] DDIM inversion (eta=0) with CFG w={guidance_scale}")
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0)

    # CFG embeddings
    cond = model.get_learned_conditioning([prompt])
    uncond = model.get_learned_conditioning([negative_prompt if negative_prompt is not None else ""])

    latents = latents_0.clone()
    intermediates = [latents.clone()]
    a_bar = torch.tensor(sampler.ddim_alphas, device=device, dtype=latents.dtype)

    for i in range(ddim_steps - 1): # stop at steps-2 to have i+1 valid
        t_next = int(sampler.ddim_timesteps[i + 1])
        alpha_t, alpha_next = a_bar[i], a_bar[i + 1]

        # predict epsilon at NEXT step (matches test_diffusion and DDIM inversion derivation)
        t_tensor = torch.tensor([t_next], device=device, dtype=torch.long)

        # CFG: predict uncond and cond noise and combine
        eps_uncond = model.apply_model(latents, t_tensor, uncond)
        eps_cond   = model.apply_model(latents, t_tensor, cond)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        latents = (
            torch.sqrt(alpha_next) * (latents - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            + torch.sqrt(1 - alpha_next) * eps
        )

        intermediates.append(latents.clone())

    return intermediates


@torch.no_grad()
def ddim_sample_from_inverted(model,
                              sampler,
                              start_latents,
                              start_step,
                              prompt, 
                              negative_prompt, 
                              guidance_scale, 
                              ddim_steps, 
                              ddim_eta, 
                              device):
    """
    Deterministic DDIM sampling from a given start_step with CFG aligned to inversion.
    """
    print(f"[sampling] DDIM sampling (eta={ddim_eta}) from step {start_step} with CFG w={guidance_scale}")
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)
    a_bar = torch.tensor(sampler.ddim_alphas, device=device, dtype=start_latents.dtype)
    timesteps = sampler.ddim_timesteps

    cond = model.get_learned_conditioning([prompt])
    uncond = model.get_learned_conditioning([negative_prompt if negative_prompt is not None else ""])

    # initialize latents from start_latents
    latents = start_latents.clone()

    # go from i=start_step down to 0 with deterministic DDIM update
    for i in range(start_step, -1, -1):
        t_i = int(timesteps[i])
        alpha_t = a_bar[i]
        alpha_prev = a_bar[i-1] if i > 0 else torch.tensor(1.0, device=device, dtype=latents.dtype)

        t_tensor = torch.tensor([t_i], device=device, dtype=torch.long)
        eps_u = model.apply_model(latents, t_tensor, uncond)
        eps_c = model.apply_model(latents, t_tensor, cond)
        eps   = eps_u + guidance_scale * (eps_c - eps_u)

        latents = (
            torch.sqrt(alpha_prev) * (latents - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            + torch.sqrt(1 - alpha_prev) * eps
        )

    return latents

    # # pass both cond/uncond and the same guidance_scale used in inversion
    # samples, _ = sampler.sample(
    #     S=ddim_steps,
    #     conditioning=cond,
    #     batch_size=1,
    #     shape=start_latents.shape[1:],
    #     verbose=True,
    #     unconditional_guidance_scale=guidance_scale,
    #     unconditional_conditioning=uncond,
    #     eta=ddim_eta,            # 0.0 for deterministic path matching inversion
    #     x_T=start_latents,       # start latent at the chosen step
    #     start_step=start_step    # requires sampler to support resuming from start_step
    # )

    # return samples


def main(opt):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"========== device: {device} ==========")
    
    model = load_ldm(
        # "models/ldm/stable-diffusion-v1-4/v1-inference.yaml",
        # "models/ldm/stable-diffusion-v1-4/sd-v1-4.ckpt",
        "models/ldm/stable-diffusion-v1-5/v1-inference.yaml",
        "models/ldm/stable-diffusion-v1-5/v1-5-pruned.safetensors",
        device
    )

    sampler = PLMSSampler(model) if opt.plms else DDIMSampler(model)
    
    # 1) Encode input PNG -> latent z0
    init_pil = Image.open(opt.input).convert("RGB")
    z0 = pil_to_latent(model, init_pil, device)

    prompt = opt.prompt if hasattr(opt, 'prompt') and opt.prompt else "a photograph"

    # 2) DDIM Inversion    
    inverted_latents = ddim_invert_latents(
        model=model,
        sampler=sampler,
        latents_0=z0,
        prompt=prompt,
        negative_prompt=opt.negative_prompt,
        guidance_scale=opt.guidance_scale,
        ddim_steps=opt.ddim_steps,
        device=device
    )
    
    total_steps = len(inverted_latents) - 1
    start_step = int(opt.strength * total_steps)
    start_step = max(0, min(total_steps-1, start_step))
    
    print(f"[reconstruction] strength={opt.strength} -> start_step={start_step}/{total_steps}")
    
    start_latents = inverted_latents[start_step]
    
    # 3) DDIM sampling
    reconstructed = ddim_sample_from_inverted(
        model=model,
        sampler=sampler,
        start_latents=start_latents,
        start_step=start_step,
        prompt=prompt,
        negative_prompt=opt.negative_prompt,
        guidance_scale=opt.guidance_scale,
        ddim_steps=opt.ddim_steps,
        ddim_eta=opt.ddim_eta,
        device=device
    )
    
    out_pil = latent_to_pil(model, reconstructed)
    os.makedirs(opt.outdir, exist_ok=True)
    save_path = os.path.join(opt.outdir, "input_restored.png")
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
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="negative prompt for CFG")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="CFG scale")

    opt = parser.parse_args()
    main(opt)


"""
사용 예시:
python scripts/img2img_old.py \
  --input images/inputs/input.png \
  --prompt "a photograph" \
  --negative_prompt "" \
  --guidance_scale 1 \
  --strength 0.2 \
  --ddim_steps 100 \
  --ddim_eta 0.0 \
  --outdir images/outputs/img2img
"""
# 'strength'만큼 DDIM forward noising(x_0 -> x_t)