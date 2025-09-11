import argparse, os, math, torch, numpy as np
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
    sd    = pl_sd["state_dict"]

    cfg   = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model


def pil_to_latent(model, pil, device):
    tfm = T.Compose([
        T.Resize((512,512), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5]*3,[0.5]*3)       # → [-1,1]
    ])
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        posterior = model.encode_first_stage(x)
        z         = posterior.sample() * 0.18215
    return z


def latent_to_pil(model, z):
    z = z / 0.18215
    with torch.no_grad():
        img = model.decode_first_stage(z)
    img = torch.clamp((img+1)/2, 0, 1).cpu()[0]
    img = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
    return Image.fromarray(img)


def main(opt):
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"========== device: {device} ==========")

    model = load_ldm(
        "models/ldm/stable-diffusion-v1/v1-inference.yaml",
        "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        device
    )

    sampler = PLMSSampler(model) if opt.plms else DDIMSampler(model)

    # 1) Encode input PNG → latent z0
    init_pil  = Image.open(opt.input).convert("RGB")
    z0        = pil_to_latent(model, init_pil, device)

    # 2) Forward-noise z0 to timestep t_enc (strength ratio)
    n_steps = opt.ddim_steps
    t_enc   = int(opt.strength * n_steps)
    t_enc   = max(0, min(n_steps-1, t_enc))
    print(f"[noise] strength={opt.strength} -> t_enc={t_enc}/{n_steps}")

    sampler.make_schedule(ddim_num_steps=n_steps, ddim_eta=opt.ddim_eta)
    alpha  = sampler.ddim_alphas_prev[t_enc]
    # alpha = torch.tensor(alpha, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float32, device=device) # MPS framework doesn't support float64
    noise  = torch.randn_like(z0)
    z_t    = alpha.sqrt() * z0 + (1-alpha).sqrt() * noise # DDIM forward equation

    # 3) Reverse-diffusion (deterministic, eta given)
    empty_cond = model.get_learned_conditioning([""]) # (1, 77, 768) tensor
    samples, _ = sampler.sample(
        S=n_steps,
        conditioning=empty_cond, # no text (여기서 style 주입할 수 있는지 확인!!!)
        batch_size=1,
        shape=z0.shape[1:],
        verbose=True,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        eta=opt.ddim_eta,
        x_T=z_t,
        start_step=t_enc
    )

    out_pil = latent_to_pil(model, samples)
    os.makedirs(opt.outdir, exist_ok=True)
    save_path = os.path.join(opt.outdir, "input_reconstructed.png")
    out_pil.save(save_path)
    print(f"[done] saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True,
                        help="input PNG/JPG (ideally 512x512)")
    parser.add_argument("--strength", type=float, default=0.75,
                        help="0=noise-free, 1=full noise")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="total DDIM timesteps")
    parser.add_argument("--ddim_eta",   type=float, default=0.0,
                        help="DDIM eta (0→deterministic)")
    parser.add_argument("--plms",    action="store_true",
                        help="use PLMS sampler instead of DDIM")
    parser.add_argument("--outdir",  default="outputs/img2img")
    opt = parser.parse_args()

    main(opt)


"""
python scripts/img2img.py \
         --input images/inputs/input.png \
         --strength 0.3 \
         --ddim_steps 100 \
         --outdir images/outputs/img2img
"""
# 'strength'만큼 DDIM forward noising(x_0 -> x_t)