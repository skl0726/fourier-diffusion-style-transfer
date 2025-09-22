import argparse, os, random
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
from torchvision import transforms as T
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline, DDIMScheduler

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from modules.clip_vit import clip_vit_encoder, StyleSelfAttention


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

    # cond = model.get_learned_conditioning([prompt if prompt is not None else ""])
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
    CONFIG_PATH = args.ldm_config_path
    MODEL_PATH = args.ldm_model_path

    print(f"===== Loading Latent Diffusion model from: {MODEL_PATH} =====")
    config = OmegaConf.load(CONFIG_PATH)
    ldm_model = instantiate_from_config(config.model)

    if MODEL_PATH.endswith(".safetensors"):
        # parameters type: ".safetensors"
        sd = load_file(MODEL_PATH, device="cpu")
    else:
        # parameters type: ".ckpt"
        sd = torch.load(MODEL_PATH, map_location="cpu")["state_dict"]
    
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
        negative_prompt="",
        num_inference_steps=args.ddim_steps,
        guidance_scale=1.0, # fix
    )

    total_steps = len(inverted_latents_seq) - 1
    start_step = max(0, min(total_steps - 1, int(args.strength * total_steps)))
    start_latents = inverted_latents_seq[start_step]


    # ***** style injection test *****
    style_img = Image.open(args.sty_img).convert("RGB")

    # clip vit encoder
    tokens = clip_vit_encoder(
        style_img,
        device=device,
        out_dim=768,
        max_tokens=77
    ) # (1, N_style, 768)
    # tokens = tokens / tokens.std(dim=-1, keepdim=True) # normalize

    # self-attention network (after clip vit encoder)
    style_encoder = StyleSelfAttention(dim=768, nhead=8, nlayers=2)
    if args.selfattn_model_path:
        sd_selfattn = load_file(args.selfattn_model_path, device="cpu")
        style_encoder.load_state_dict(sd_selfattn, strict=True)
    style_encoder = style_encoder.to(device).eval()
    
    style_tokens = style_encoder(tokens)

    uncond_text = ldm_model.get_learned_conditioning([args.negative_prompt]) # (1, N_txt=77, 768)

    def match_batch(x, B):
        return x if x.shape[0] == B else x.expand(B, -1, -1)

    B = start_latents.shape[0]
    uncond_text = uncond_text.expand(B, 77, -1)
    style_tokens = match_batch(style_tokens.to(device, dtype=uncond_text.dtype), B)

    print("********cond shape:", style_tokens.shape, "uncond shape:", uncond_text.shape)

    num_layers = 25 # number of SD-1.x UNet cross-attention layer
    cond   = {"c_crossattn": [args.sty_alpha * style_tokens]  * num_layers}
    uncond = {"c_crossattn": [uncond_text] * num_layers}

    def _extract_tensor_from_cond(c):
        if isinstance(c, dict):
            c2 = c.get("c_crossattn", c)
            if isinstance(c2, list):
                return c2[0]
            return c2
        return c

    cond = _extract_tensor_from_cond(cond)
    uncond = _extract_tensor_from_cond(uncond)
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
    SEED = 1234
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("--ldm_config_path", type=str, default="models/ldm/stable-diffusion-v1-5/v1-inference.yaml")
    parser.add_argument("--ldm_model_path", type=str, default="models/ldm/stable-diffusion-v1-5/v1-5-pruned.safetensors")
    parser.add_argument("--selfattn_model_path", type=str, default="models/attention/style-encoder/style_encoder_20000.safetensors")
    
    parser.add_argument("--cnt_img", type=str, default="images/inputs/cnt/cnt_1.png")
    parser.add_argument("--sty_img", type=str, default="images/inputs/sty/sty_1.png")
    parser.add_argument("--sty_alpha", type=float, default=1.0)
    parser.add_argument("--output_img", type=str, default="images/outputs/img2img_style/stylized.png")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    
    opt = parser.parse_args()
    main(opt)


"""
CUDA_VISIBLE_DEVICES=1 python inference/img2img_style.py \
  --cnt_img images/inputs/cnt/cnt_1.png \
  --sty_img images/inputs/sty/sty_1.png \
  --output_img images/outputs/img2img_style/stylized_1.png \
  --strength 0.4 \
  --ddim_steps 100
"""


"""
[MEMO]

problem:
- style 이미지를 condition으로 넣었을 때 제대로 동작하지 않음 (기존 style transfer 모델의 전형적인 output과는 다른 이미지가 출력)
- content와 style 이미지를 모두 동일한 이미지로 사용했을 때에도, style 이미지를 다른 것을 사용했을 때와 비슷한 결과 출력

idea:
i)  stage 1 (training style encoder): ldm의 u-net이 이해할 수 있는 style condition representation을 학습 (StyleSelfAttention 학습)
ii) stage 2 (fine-tuning denoising u-net): fourier transform 적용 (병렬 diffsuion)

todo:
- 체크해야 할 사항: cnt + sty 잘 반영되는지 / cnt + cnt일 때 원본 이미지 그대로 복구하는지
"""