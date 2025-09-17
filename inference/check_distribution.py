import torch, numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image

from modules.clip_vit import clip_vit_encoder, StyleSelfAttention
from ldm.util import instantiate_from_config


@torch.no_grad()
def get_text_tokens(ldm_model, prompts, device, dtype=torch.float32):
    # ldm_model.get_learned_conditioning returns (B, 77, 768) for SD v1.x
    out = []
    bs = 32
    for i in range(0, len(prompts), bs):
        toks = ldm_model.get_learned_conditioning(prompts[i:i+bs]).to(device, dtype=dtype)
        out.append(toks)
    return torch.cat(out, dim=0)  # (N, 77, 768)


@torch.no_grad()
def get_style_tokens(style_imgs, clip_vit_encoder, style_encoder, device, dtype=torch.float32):
    outs = []
    for img in style_imgs:
        t = clip_vit_encoder(img, device=device, out_dim=768, max_tokens=77)
        t = style_encoder(t)  # same dims
        outs.append(t.to(device, dtype=dtype))
    return torch.cat(outs, dim=0)  # (N, 77, 768)


def feat_stats(x):  # x: (N, T, D)
    x2 = x.reshape(-1, x.shape[-1])
    mu = x2.mean(0)
    cov = torch.from_numpy(np.cov(x2.cpu().numpy(), rowvar=False)).to(x.device, x.dtype)
    std = x2.std(0)
    return mu, cov, std


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    # FD between two Gaussians ~ used in FID
    diff = (mu1 - mu2).unsqueeze(0)
    # matrix sqrt via eigen
    def sqrtm(mat):
        w, v = torch.linalg.eigh(mat + eps*torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype))
        return (v * torch.clamp(w, min=0).sqrt().unsqueeze(0)) @ v.transpose(-1, -2)
    covmean = sqrtm(cov1 @ cov2)
    fd = (diff @ diff.transpose(-1, -2)).squeeze() + torch.trace(cov1 + cov2 - 2*covmean)
    return fd.item()


def cosine_stats(A, B):
    # pairwise cosine between per-token means
    A_m = A.mean(1)  # (N, D)
    B_m = B.mean(1)  # (N, D)
    A_m = A_m / (A_m.norm(dim=-1, keepdim=True)+1e-8)
    B_m = B_m / (B_m.norm(dim=-1, keepdim=True)+1e-8)
    cs = (A_m * B_m).sum(-1)
    return cs.mean().item(), cs.std().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    style_encoder = StyleSelfAttention(dim=768, nhead=8, nlayers=2).to(device).eval()

    style_img = Image.open("images/inputs/sty/sty_4.png").convert("RGB")
    prompts = ["style image of pencil draw of woman"]*100

    text_tok = get_text_tokens(ldm_model, prompts, device, dtype=torch.float32)   # (N,77,768)
    style_tok = get_style_tokens([style_img]*len(prompts), clip_vit_encoder, style_encoder, device, torch.float32)  # (N,77,768)

    t_mu, t_cov, t_std = feat_stats(text_tok)
    s_mu, s_cov, s_std = feat_stats(style_tok)
    fd = frechet_distance(t_mu, t_cov, s_mu, s_cov)
    cos_m, cos_s = cosine_stats(text_tok, style_tok)

    print("Per-dim mean Δ (L2):", torch.linalg.norm(t_mu - s_mu).item())
    print("Per-dim std Δ (L2):", torch.linalg.norm(t_std - s_std).item())
    print("Fréchet-like distance:", fd)
    print("Mean cosine(text, style):", cos_m, "+/-", cos_s)


"""
CUDA_VISIBLE_DEVICES=7 python inference/check_distribution.py

[지표별 해석]
Per-dim mean Δ (L2) = 24.51:
- 두 분포의 중심 차이가 크다는 뜻으로, 이미지·텍스트 임베딩 중심의 이격을 모달리티 갭으로 정의해 정량화하는 기존 관찰과 일치한다.

Per-dim std Δ (L2) = 12.94:
- 분산 구조(스케일)가 다르다는 의미로, 동일 어텐션 투영을 거칠 때 가중치 분포가 달라져 cross-attention 안정성이 떨어질 수 있다.

Fréchet-like distance = 766.37:
- FID 계열 거리의 해석과 동일하게 값이 클수록 두 분포가 멀다는 의미이며,
    절대값보다는 동일 표본 규모에서의 상대 비교가 중요하나 이 값은 상당히 큰 편으로 볼 수 있다.

Mean cosine ≈ 0.0226:
- 고차원에서 무작위 단위 벡터 쌍의 기대 코사인은 0이고 분산은 1/n로 수렴하므로,
    거의 0에 가까운 값은 두 임베딩이 방향 정렬을 거의 이루지 못했다는 의미다.

* Stable Diffusion v1.x의 cross-attention은 텍스트 임베딩 분포에 맞춰 학습되어 있어,
    통계가 상이한 이미지 기반 토큰을 K,V로 대체하면 모델이 신호를 무시하거나 불안정한 주의 맵을 형성하기 쉽다.
"""