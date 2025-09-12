"""
- Forward noising to a chosen timestep t* and denoise back
- DDIM Inversion to recover the original image from an intermediate step
"""


import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

from diffusers import StableDiffusionPipeline, DDIMScheduler

import os

import torch.nn.functional as F


def describe_noise_level(
    scheduler,
    num_inference_steps,
    start_step,
    init_latents=None,
    noise=None,
    device=torch.device("cpu"),
):
    """
    Prints the alpha (alphas_cumprod) at the scheduler timestep corresponding to start_step,
    signal/noise weights, and optionally compares the actual latents_t (if init_latents and noise are provided)
    to the analytical mixture.
    
    - scheduler: pipe.scheduler
    - num_inference_steps: how many steps the scheduler is set to (must call set_timesteps first or inside)
    - start_step: integer index into scheduler.timesteps (0 <= start_step < num_inference_steps)
    - init_latents: (optional) the x0 latent tensor used for building latents_t
    - noise: (optional) the noise tensor used in add_noise
    """
    # ensure timesteps are set the same way you will run the process
    scheduler.set_timesteps(num_inference_steps, device=device)
    assert 0 <= start_step < len(scheduler.timesteps), "start_step out of range"

    t = scheduler.timesteps[start_step]          # this is a tensor like tensor(980)
    # alpha (bar alpha) at this timestep
    try:
        alpha = scheduler.alphas_cumprod[t].item()
    except Exception:
        # some schedulers store as numpy; fallback
        alpha = float(scheduler.alphas_cumprod[int(t.item())])

    signal_weight = float(np.sqrt(alpha))
    noise_weight = float(np.sqrt(1.0 - alpha))
    noise_percent = (1.0 - alpha) * 100.0

    print(f"=== Noise stats for start_step={start_step} (t={int(t.item())}) ===")
    print(f"alphas_cumprod[t] = {alpha:.8f}")
    print(f"signal weight (sqrt(alpha)) = {signal_weight:.6f}")
    print(f"noise  weight (sqrt(1-alpha)) = {noise_weight:.6f}")
    print(f"approx. noise energy = {noise_percent:.4f}%")
    print("---------------------------------------------------------")

    # If user supplied init_latents and noise, compute the expected latents_t and compare
    if init_latents is not None and noise is not None:
        # expected latent from formula: sqrt(alpha)*x0 + sqrt(1-alpha)*noise
        alpha_t = torch.tensor(alpha, device=init_latents.device, dtype=init_latents.dtype)
        exp_latents_t = alpha_t.sqrt() * init_latents + (1 - alpha_t).sqrt() * noise

        # compute MSE between expected and actual if user passed an actual latents_t (we can also compute actual via scheduler.add_noise)
        actual_latents_t = scheduler.add_noise(init_latents, noise, t)

        mse = F.mse_loss(actual_latents_t, exp_latents_t).item()
        print(f"MSE(actual add_noise vs analytic mixture) = {mse:.6e}")

        # Flatten and compute cosine similarity with original and with noise to show "how much original remains"
        a = actual_latents_t.flatten(1)  # (B, N)
        x0 = init_latents.flatten(1)
        n = noise.flatten(1)

        # use cosine similarity averaged across batch
        cos_with_x0 = F.cosine_similarity(a, x0, dim=1).mean().item()
        cos_with_noise = F.cosine_similarity(a, n, dim=1).mean().item()

        print(f"cosine_similarity(latents_t, x0)   = {cos_with_x0:.6f}")
        print(f"cosine_similarity(latents_t, noise)= {cos_with_noise:.6f}")
        print("---------------------------------------------------------")

    # return numbers programmatically too
    return {
        "t": int(t.item()),
        "alpha": alpha,
        "signal_weight": signal_weight,
        "noise_weight": noise_weight,
        "noise_percent": noise_percent,
    }


# ---------------------------
# 0) Setup: load pipeline on CPU and use DDIM scheduler
# ---------------------------
device = torch.device("cpu")
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Replace scheduler with DDIM (deterministic if eta=0)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


# ---------------------------
# 1) Utilities
# ---------------------------

# Preprocess PIL image to VAE latent
# - Use 512x512 for SD v1.x
# - Latent scaling factor 0.18215 (as used in img2img)
def pil_to_latents(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB").resize((512, 512), Image.LANCZOS)
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    x = tfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(x).latent_dist
        latents = latent_dist.sample()
        latents = latents * 0.18215  # VAE scaling used by SD v1.x
    return latents

# Decode latent to PIL
def latents_to_pil(latents):
    with torch.no_grad():
        images = pipe.decode_latents(latents)
    pil_list = pipe.numpy_to_pil(images)
    return pil_list

# Get text embeddings (classifier-free guidance ready)
def _encode_prompt(pipe, prompt, negative_prompt="", num_images_per_prompt=1, do_cfg=True):
    # Use the same internal helper as Hugging Face course (works across diffusers versions)
    return pipe._encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )


# ---------------------------
# 3) DDIM sampling from an arbitrary start step (deterministic, eta=0)
# ---------------------------
@torch.no_grad()
def ddim_sample_from(
    prompt,
    start_latents,
    start_step,
    num_inference_steps=50,
    guidance_scale=1.0,
    negative_prompt="",
    eta=0.0,
    save_intermediates=False,
    save_prefix="ddim_step"
):
    do_cfg = guidance_scale > 1.0
    text_embeds = _encode_prompt(pipe, prompt, negative_prompt, 1, do_cfg)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    if save_intermediates:
        os.makedirs("ddim_image", exist_ok=True)

    latents = start_latents.clone()
    for i in range(start_step, num_inference_steps):
        t_i = pipe.scheduler.timesteps[i]
        latent_model_input = torch.cat([latents, latents]) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t_i)

        noise_pred = pipe.unet(latent_model_input, t_i, encoder_hidden_states=text_embeds).sample
        if do_cfg:
            n_uncond, n_text = noise_pred.chunk(2)
            noise_pred = n_uncond + guidance_scale * (n_text - n_uncond)

        step_out = pipe.scheduler.step(noise_pred, t_i, latents, eta=eta)
        latents = step_out.prev_sample

        if save_intermediates:
            imgs = latents_to_pil(latents)
            imgs[0].save(f"ddim_image/{save_prefix}_{i}.png")

    return latents


# ---------------------------
# 4) DDIM Inversion: push latents “forward” in time (t: 0 -> high)
#    Then re-start denoising from an intermediate inverted latent
# ---------------------------
@torch.no_grad()
def ddim_invert_latents(
    latents_0,
    prompt,
    num_inference_steps=80,
    guidance_scale=3.5,
    negative_prompt="",
):
    # Encode prompt with CFG
    do_cfg = guidance_scale > 1.0
    text_embeds = _encode_prompt(pipe, prompt, negative_prompt, 1, do_cfg)

    # Prepare scheduler timesteps (forward pass uses reversed order indexing)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # We'll index timesteps in reverse to go "forward" in time
    ts = list(reversed(pipe.scheduler.timesteps))

    latents = latents_0.clone()
    intermediates = []

    for i in range(1, num_inference_steps):
        t_next = ts[i]              # next (higher) timestep
        t_curr = max(0, t_next.item() - (1000 // num_inference_steps))  # approx previous in DDIM grid

        alpha_t = pipe.scheduler.alphas_cumprod[t_curr]
        alpha_next = pipe.scheduler.alphas_cumprod[t_next]

        latent_in = torch.cat([latents, latents]) if do_cfg else latents
        latent_in = pipe.scheduler.scale_model_input(latent_in, t_next)

        noise_pred = pipe.unet(latent_in, t_next, efncoder_hidden_states=text_embeds).sample
        if do_cfg:
            n_uncond, n_text = noise_pred.chunk(2)
            noise_pred = n_uncond + guidance_scale * (n_text - n_uncond)

        # Inversion update (rearranged DDIM step for x_t -> x_{t+1})
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_next.sqrt() / alpha_t.sqrt()) \
                  + (1 - alpha_next).sqrt() * noise_pred

        intermediates.append(latents.clone())

    return intermediates  # list of latents at increasing noise levels


# ---------------------------
# 5) Demo
# ---------------------------
if __name__ == "__main__":
    # 5-1) Load an input image
    img_path = "input.png"  # put your image here (512x512 recommended)
    image = Image.open(img_path)

    # 5-2) Encode to latents
    latents0 = pil_to_latents(image)

    # ---------- DDIM inversion -> restart from intermediate ----------
    inv_steps = 100
    inv_prompt = "A photograph"  # brief description helps CFG during inversion
    inverted_seq = ddim_invert_latents(
        latents_0=latents0,
        prompt=inv_prompt,
        num_inference_steps=inv_steps,
        guidance_scale=3.5,
        negative_prompt="",
    )

    # pick a mid inversion step to start reverse (closer -> better fidelity)
    start_step_inv = 20  # out of inv_steps
    start_latents = inverted_seq[-(start_step_inv + 1)]
    recon_latents = ddim_sample_from(
        prompt=inv_prompt,               # use same prompt for “identity edit”
        start_latents=start_latents,
        start_step=start_step_inv,
        num_inference_steps=inv_steps,
        guidance_scale=3.5,
        negative_prompt="",
        eta=0.0,
        save_intermediates=True,
        save_prefix="recon_progress"
    )

    recon_pils = latents_to_pil(recon_latents)

    if not os.path.exists("ddim_image"):
        os.makedirs("ddim_image", exist_ok=True)
    recon_pils[0].save("ddim_image/restore_ddim_inversion.png")

    print("Saved: restore_forward_then_denoise.png and restore_ddim_inversion.png")