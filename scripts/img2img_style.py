import argparse, os
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms as T

from diffusers import StableDiffusionPipeline, DDIMScheduler

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from modules.vit_encoder import ViTStyleEncoder


def pil_to_latents(pil_img: Image.Image, pipe, device):
    """
    Encoder: convert (encode) image -> latent
    """
    pil_img = pil_img.convert("RGB").resize((512, 512), Image.LANCZOS)
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = tfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(x).latent_dist
        latents = latent_dist.sample()
        latents = latents * 0.18215 # VAE scaling used by SD v1.x
    return latents


def latents_to_pil(latents, pipe):
    """
    Decoder: convert (decode) latent -> image
    """
    latents = latents / 0.18215
    with torch.no_grad():
        images = pipe.vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    return pipe.numpy_to_pil(images)


def register_kv_hook(unet, style_kv, replace_ratio):

    def _hook(module, args):
        """
        args -> tuple of positional inputs (may be 1 or 2 elements)
        For SD-1.x:
            module(hidden_states, context)
        For some patched versions:
            module(hidden_states, context=None)  # context may be kwarg
        """
        # positional
        hidden_states = args[0]
        if len(args) == 2:                     # 일반적인 SD-1.x
            context = args[1]
        else:                                  # len==1 → context는 None이거나 kwarg로 전달
            context = getattr(module, 'context', None)
            if context is None:
                # 마지막 보루: 이전 forward_pre_hook에서 module에 저장
                context = hidden_states

        style_320 = project_to_320(style_kv)

        # 새 KV 생성
        if replace_ratio == 1.0:
            ctx_new = style_kv
        elif replace_ratio == 0.0:
            ctx_new = context
        else:
            ctx_new = torch.cat([context*(1-replace_ratio), style_kv*replace_ratio], dim=1)

        return (hidden_states, ctx_new)

    for m in unet.modules():
        if m.__class__.__name__ == "CrossAttention":
            m.register_forward_pre_hook(_hook, with_kwargs=False)


@torch.no_grad()
def ddim_invert_latents(device,
                        pipe,
                        latents_0,
                        prompt,
                        num_inference_steps=100,
                        guidance_scale=1.0,
                        negative_prompt=""):
    """
    DDIM Inversion: push latents forward (t: 0 -> high)
    """
    print("===== Starting DDIM Inversion using diffusers pipeline... =====")
    
    # encode prompt with CFG
    do_cfg = guidance_scale > 1.0
    prompt_embeds_tuple = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt
    )
    text_embeds = prompt_embeds_tuple[0]
    
    # prepare scheduler timesteps (forward pass uses reversed order indexing)
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = list(reversed(pipe.scheduler.timesteps))
    
    latents = latents_0.clone()
    intermediates = [latents.clone()]

    for i in range(len(timesteps) - 1):
        t_curr, t_next = timesteps[i], timesteps[i+1]
        alpha_t, alpha_next = pipe.scheduler.alphas_cumprod[t_curr], pipe.scheduler.alphas_cumprod[t_next]
        
        latent_in = torch.cat([latents, latents]) if do_cfg else latents
        latent_in = pipe.scheduler.scale_model_input(latent_in, t_next)

        noise_pred = pipe.unet(latent_in, t_next, encoder_hidden_states=text_embeds).sample
        if do_cfg:
            n_uncond, n_text = noise_pred.chunk(2)
            noise_pred = n_uncond + guidance_scale * (n_text - n_uncond)

        # inversion update (rearranged DDIM step for x_t -> x_{t+1})
        latents = (
            (alpha_next.sqrt() / alpha_t.sqrt()) * (latents - (1 - alpha_t).sqrt() * noise_pred) + 
            (1 - alpha_next).sqrt() * noise_pred
        )
        intermediates.append(latents.clone())

    print("===== DDIM Inversion complete. =====")
    return intermediates # list of latents at increasing noise levels


@torch.no_grad()
def ddim_sample_from_inverted(device,
                              sampler,
                              start_latents,
                              start_step,
                              cond,
                              uncond,
                              guidance_scale,
                              ddim_steps,
                              ddim_eta):
    """
    DDIM Sampling: sampling from an arbitrary start step (deterministic, eta=0)
    """
    print(f"===== Starting Reverse Diffusion from step {start_step} using Latent Diffusion Model... =====")

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)

    # cond = model.get_learned_conditioning([prompt])
    # uncond = model.get_learned_conditioning([negative_prompt if negative_prompt is not None else ""])
    latents = start_latents.clone()

    batch_size = start_latents.shape[0]

    for index in range(start_step, -1, -1):
        t_numpy = sampler.ddim_timesteps[index]
        t = torch.full((batch_size,), t_numpy, device=device, dtype=torch.long)
        
        latents, _ = sampler.p_sample_ddim(
            x=latents,
            c=cond,
            t=t,
            index=index,
            use_original_steps=False,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond
        )

    print("===== Reverse Diffusion complete. =====")
    return latents


def main(args):
    # set device
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
    CONFIG_PATH = "models/ldm/stable-diffusion-v1-5/v1-inference.yaml"      # TODO: set config path
    CKPT_PATH = "models/ldm/stable-diffusion-v1-5/v1-5-pruned.safetensors"  # TODO: set model checkpoint path

    print(f"===== Loading Latent Diffusion model from: {CKPT_PATH} =====")
    config = OmegaConf.load(CONFIG_PATH)
    ldm_model = instantiate_from_config(config.model)

    if CKPT_PATH.endswith(".safetensors"):
        # parameters type: ".safetensors"
        from safetensors.torch import load_file
        sd = load_file(CKPT_PATH, device="cpu")
    else:
        # parameters type: ".ckpt"
        sd = torch.load(CKPT_PATH, map_location="cpu")["state_dict"]
    
    ldm_model.load_state_dict(sd, strict=False)
    ldm_model = ldm_model.to(device).eval()
    sampler = DDIMSampler(ldm_model)

    # 1) initial VAE encoding
    content_img = Image.open(args.cnt_img)
    latents_0 = pil_to_latents(content_img, pipe, device)

    # 2) DDIM Inversion
    inverted_latents_seq = ddim_invert_latents(
        device=device,
        pipe=pipe,
        latents_0=latents_0,
        prompt="",
        num_inference_steps=args.ddim_steps,
        guidance_scale=1.0, # fix
        negative_prompt="",
    )

    total_steps = len(inverted_latents_seq) - 1
    start_step = max(0, min(total_steps - 1, int(args.strength * total_steps)))
    start_latents = inverted_latents_seq[start_step]


    # ***** style injection test *****
    style_encoder = ViTStyleEncoder()
    style_feature = style_encoder(Image.open(opt.sty_img), device=device) # (1, N, 768)
    
    B = start_latents.shape[0]
    style_feature = style_feature.expand(B, -1, -1)

    first_ca = next(m for m in ldm_model.modules()
                if m.__class__.__name__ == "CrossAttention")

    W_k = first_ca.to_k.weight       # [320, 768] 또는 [320,320]
    b_k = first_ca.to_k.bias         # [320]

    def project_to_320(x: torch.Tensor):
        """
        x : (B, N, 768 또는 320)
        반환: (B, N, 320) - 모델의 to_k/to_v 입력 크기와 동일
        """
        if x.shape[-1] == 320:
            return x
        return torch.matmul(x, W_k.T.to(x.dtype)) + b_k

    register_kv_hook(ldm_model.model.diffusion_model,
                     style_feature,
                     opt.sty_alpha)

    empty = ldm_model.get_learned_conditioning([""])
    cond = {"c_crossattn": [empty]*25}
    uncond = {"c_crossattn": [empty]*25}
    # ********************************

    
    # 3) DDIM Sampling
    reconstructed_latents = ddim_sample_from_inverted(
        device=device,
        sampler=sampler,
        start_latents=start_latents,
        start_step=start_step,
        cond=cond,
        uncond=uncond,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
    )

    # 4) final VAE decoding
    output_pil = latents_to_pil(reconstructed_latents, pipe)

    # save image
    outdir = os.path.dirname(args.output_img)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    output_pil[0].save(args.output_img)

    print(f"✅ Saved stylized image to: {args.output_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cnt_img", type=str, default="images/inputs/input.png")
    parser.add_argument("--sty_img", type=str, default="images/inputs/sty_1.png")
    parser.add_argument("--sty_alpha", type=float, default=1.0)
    parser.add_argument("--output_img", type=str, default="images/outputs/img2img_style/stylized.png")
    parser.add_argument("--prompt", type=str, default="a photograph of stylized image")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0) # > 1.0인 경우 에러
    
    opt = parser.parse_args()
    main(opt)


"""
python scripts/img2img_style.py \
  --cnt_img images/inputs/cnt/cnt_1.png \
  --sty_img images/inputs/sty/sty_1.png \
  --output_img images/outputs/img2img_style/stylized_new.png \
  --strength 0.6 \
  --ddim_steps 100

# 'strength'만큼 DDIM forward noising(x_0 -> x_t)
"""