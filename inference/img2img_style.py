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
from modules.clip_vit import StyleEncoder


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
                        negative_prompt,
                        num_inference_steps=100,
                        guidance_scale=1.0):
    """
    DDIM Inversion: push latents forward (t: 0 -> high)
    """
    print("========== Starting DDIM Inversion... ==========")
    
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
        idx_t, idx_next = int(t_curr), int(t_next)
        alphas = pipe.scheduler.alphas_cumprod.to(device)
        alpha_t, alpha_next = alphas[idx_t], alphas[idx_next]
        # alpha_t, alpha_next = pipe.scheduler.alphas_cumprod[t_curr], pipe.scheduler.alphas_cumprod[t_next]
        
        latent_in = torch.cat([latents, latents]) if do_cfg else latents
        latent_in = pipe.scheduler.scale_model_input(latent_in, t_curr)

        noise_pred = pipe.unet(latent_in, t_curr, encoder_hidden_states=text_embeds).sample
        if do_cfg:
            n_uncond, n_text = noise_pred.chunk(2)
            noise_pred = n_uncond + guidance_scale * (n_text - n_uncond)

        # inversion update (rearranged DDIM step for x_t -> x_{t+1})
        latents = (
            (alpha_next.sqrt() / alpha_t.sqrt()) * (latents - (1 - alpha_t).sqrt() * noise_pred) + 
            (1 - alpha_next).sqrt() * noise_pred
        )
        intermediates.append(latents.clone())

    print("========== DDIM Inversion complete. ==========")
    return intermediates # list of latents at increasing noise levels


@torch.no_grad()
def ddim_sample_from_inverted(device,
                              model,
                              sampler,
                              start_latents,
                              start_step,
                              prompt,
                              negative_prompt,
                              injected_features,
                              guidance_scale,
                              ddim_steps,
                              ddim_eta):
    """
    DDIM Sampling: sampling from an arbitrary start step (deterministic, eta=0)
    """
    print(f"========== Starting DDIM Sampling... ==========")

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta)

    cond = model.get_learned_conditioning([prompt])
    uncond = model.get_learned_conditioning([negative_prompt])
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
            injected_features=injected_features,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uncond
        )

    print("========== Reverse Diffusion complete. ==========")
    return latents


def main(args):
    # set device
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"========== device: {device} ==========")

    # stable diffusion v1.5 parameters
    CONFIG_PATH = args.ldm_config_path
    MODEL_PATH = args.ldm_model_path

    # load model (for DDIM Inversion - diffusion process)
    print("========== [DDIM Inversion] Loading Hugging Face diffusers pipeline for VAE and Inversion... ==========")
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = StableDiffusionPipeline.from_single_file(
        args.ldm_model_path,
        original_config_file=args.ldm_config_path,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # load model (for DDIM Sampling - reverse diffusion process)
    print(f"========== [DDIM Sampling] Loading Latent Diffusion model... ==========")
    config = OmegaConf.load(CONFIG_PATH)
    ldm_model = instantiate_from_config(config.model)
    if MODEL_PATH.endswith(".safetensors"):
        sd = load_file(MODEL_PATH, device="cpu")                        # parameters type: ".safetensors"
    else:
        sd = torch.load(MODEL_PATH, map_location="cpu")["state_dict"]   # parameters type: ".ckpt"
    ldm_model.load_state_dict(sd, strict=False)
    ldm_model = ldm_model.to(device).eval()
    sampler = DDIMSampler(ldm_model)

    # load style encoder module
    style_encoder = StyleEncoder(device=device, sty_alpha=args.sty_alpha)

    # load content and style images
    content_img = Image.open(args.cnt_img).convert("RGB")
    if args.sty_img:
        style_img = Image.open(args.sty_img).convert("RGB")

    # 1) initial VAE encoding
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
    start_step = max(0, min(total_steps, int(args.strength * total_steps)))
    start_latents = inverted_latents_seq[start_step]

    # 3) get style image features
    if args.sty_img:
        injected_features = style_encoder(style_img) # dict{'k': k, 'v': v, 'sty_alpha': alpha}
    else:
        injected_features = None
    
    # 4) DDIM Sampling + style injection
    reconstructed_latents = ddim_sample_from_inverted(
        device=device,
        model=ldm_model,
        sampler=sampler,
        start_latents=start_latents,
        start_step=start_step,
        prompt="",
        negative_prompt="",
        injected_features=injected_features, # style injection
        guidance_scale=1.0, # fix
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
    )

    # 5) final VAE decoding
    output_pil = latents_to_pil(reconstructed_latents, pipe)

    # save image
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if args.sty_img:
        cnt_filename = os.path.basename(args.cnt_img)
        sty_filename = os.path.basename(args.sty_img)
        cnt_name, _ = os.path.splitext(cnt_filename)
        sty_name, _ = os.path.splitext(sty_filename)
        out_path = os.path.join(args.output_dir, f"{cnt_name}_stylized_{sty_name}.png")
        output_pil[0].save(out_path)
    else:
        cnt_filename = os.path.basename(args.cnt_img)
        cnt_name, _ = os.path.splitext(cnt_filename)
        out_path = os.path.join(args.output_dir, f"{cnt_name}_reconstructed.png")
        output_pil[0].save(out_path)

    print(f"✅ Saved image to: {out_path}")


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
    
    parser.add_argument("--cnt_img", type=str, required=True)   # _data/cnt/<cnt_image_name>.png 
    parser.add_argument("--sty_img", type=str, default="")      # _data/sty/<sty_image_name>.png 
    parser.add_argument("--sty_alpha", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="_outputs/")
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    
    opt = parser.parse_args()
    main(opt)


"""
CUDA_VISIBLE_DEVICES=1 python inference/img2img_style.py \
  --cnt_img _data/cnt/cnt1.png \
  --sty_img _data/sty/sty4.png \
  --sty_alpha 0.5 \
  --strength 0.4 \
  --ddim_steps 100

CUDA_VISIBLE_DEVICES=1 python inference/img2img_style.py \
  --cnt_img _data/cnt/cnt1.png \
  --strength 0.4 \
  --ddim_steps 100
"""


"""
[MEMO]

idea:
i)  stage 1 (training style encoder): ldm의 u-net이 이해할 수 있는 style condition representation을 학습 (StyleSelfAttention 학습)
ii) stage 2 (fine-tuning denoising u-net): fourier transform 적용 (병렬 diffsuion)

고려해야 할 사항:
- StyleID의 코드대로 style feature 삽입 로직 구현하기 (ddim.py, ddpm.py, openaimodel.py, attention.py)

- ldm 학습 과정에서, ddim inversion의 매 step에서 주입된 noise와 denoising u-net의 매 step에서 예측한 noise를 MSE loss로 학습시키는 방식을 활용할 텐데,
    ddim inversion의 매 step에서 어떻게 noise를 추출하는지? (지금은 diffusers 라이브러리 쓰고 있는데...)

"""