import argparse, os
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms as T

from diffusers import StableDiffusionPipeline, DDIMScheduler

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from modules.clip_vit import get_style_tokens


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


    # ***** style injection test (fixed) *****
    style_img = Image.open(args.sty_img).convert("RGB")
    style_tokens = get_style_tokens(style_img, device=device)  # (1, N_style, 768)
    style_tokens = style_tokens / style_tokens.std(dim=-1, keepdim=True) # normalize

    cond_text = ldm_model.get_learned_conditioning([args.prompt])  # (1, N_txt, 768)
    uncond_text = ldm_model.get_learned_conditioning([""])  # (1, N_txt, 768)

    def match_batch(x, B):
        return x if x.shape[0] == B else x.expand(B, -1, -1)

    B = start_latents.shape[0]
    cond_txt, uncond_txt = map(lambda t: match_batch(t.to(device), B),
                               (cond_text, uncond_text))
    style_tokens = match_batch(style_tokens.to(device, dtype=cond_txt.dtype), B)

    num_layers=25 # number of SD-1.x UNet cross-attention layer
    cross  = torch.cat([cond_txt,  args.sty_alpha * style_tokens], dim=1)
    ucross = torch.cat([uncond_txt, torch.zeros_like(style_tokens)], dim=1)
    cond   = {"c_crossattn": [cross]  * num_layers}   # <-- (1)
    uncond = {"c_crossattn": [ucross] * num_layers}

    # alpha = args.sty_alpha

    # # 정수 배치 크기
    # B = start_latents.shape[0]

    # # dtype/device 정렬
    # style_tokens = style_tokens.to(device=device, dtype=cond_text.dtype)
    # cond_text   = cond_text.to(device=device)
    # uncond_text = uncond_text.to(device=device)

    # def match_batch(x: torch.Tensor, B: int):
    #     # 보장: x는 (batch, seq_len, dim) 형태여야 함. 만약 (seq_len, dim)이라면 batch 차원 추가
    #     if x.dim() == 2:
    #         x = x.unsqueeze(0)

    #     b0 = x.shape[0]  # 실제 배치 크기 (int)

    #     if b0 == B:
    #         return x
    #     if b0 == 1 and B > 1:
    #         return x.expand(B, -1, -1)
    #     # 만약 B가 b0의 배수가 아니라면 단순히 앞에서 자름 (혹은 원하는 동작으로 바꿔도 됨)
    #     if B % b0 != 0:
    #         return x[:B]
    #     reps = B // b0
    #     return x.repeat(reps, 1, 1)
    
    # cond_text_b    = match_batch(cond_text, B)      # (B, N_txt, 768)
    # uncond_text_b  = match_batch(uncond_text, B)    # (B, N_txt, 768)
    # style_tokens_b = match_batch(style_tokens, B)   # (B, N_style, 768)

    # cond   = {"c_crossattn": [torch.cat([cond_text_b,   alpha * style_tokens_b], dim=1)]}
    # uncond = {"c_crossattn": [torch.cat([uncond_text_b, torch.zeros_like(style_tokens_b)], dim=1)]}
    # ********************************

    ##########    
    # cond_tensor = cond["c_crossattn"][0].to(device=device)
    # uncond_tensor = uncond["c_crossattn"][0].to(device=device)

    # # ensure batch dimension matches start_latents batch size
    # latent_batch = start_latents.shape[0]
    # if cond_tensor.shape[0] != latent_batch:
    #     # if cond_tensor was (1, seq, dim), expand to match batch
    #     if cond_tensor.shape[0] == 1:
    #         cond_tensor = cond_tensor.expand(latent_batch, -1, -1)
    #         uncond_tensor = uncond_tensor.expand(latent_batch, -1, -1)
    #     else:
    #         # if smaller but divides, repeat; otherwise trim/expand first row
    #         if latent_batch % cond_tensor.shape[0] == 0:
    #             reps = latent_batch // cond_tensor.shape[0]
    #             cond_tensor = cond_tensor.repeat(reps, 1, 1)
    #             uncond_tensor = uncond_tensor.repeat(reps, 1, 1)
    #         else:
    #             cond_tensor = cond_tensor[:latent_batch].expand(latent_batch, -1, -1)
    #             uncond_tensor = uncond_tensor[:latent_batch].expand(latent_batch, -1, -1)
    ##########

    
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
  --cnt_img images/inputs/input.png \
  --sty_img images/inputs/sty_1.png \
  --output_img images/outputs/img2img_style/stylized.png \
  --prompt "a photograph of stylized image" \
  --strength 0.6 \
  --ddim_steps 100 \
  --ddim_eta 0.0 \
  --guidance_scale 1.0

# 'strength'만큼 DDIM forward noising(x_0 -> x_t)
"""