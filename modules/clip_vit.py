import torch, torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor


_clip_model = None
_clip_proc = None
_proj = None


def clip_vit_encoder(img, # PIL image or torch.Tensor
                     model_name="openai/clip-vit-large-patch14",
                     out_dim=768,
                     max_tokens=64,
                     device="cuda"):
    global _clip_model, _clip_proc, _proj
    device = torch.device(device)

    if _clip_model is None:
        _clip_model = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
        _clip_proc = CLIPImageProcessor.from_pretrained(model_name)
        hidden = _clip_model.config.hidden_size  # e.g., 1024 for ViT-L/14
        _proj = nn.Identity() if hidden == out_dim else nn.Linear(hidden, out_dim, bias=False).to(device)
        for p in list(_clip_model.parameters()) + list(_proj.parameters()):
            p.requires_grad_(False)

    if isinstance(img, torch.Tensor):
        if img.min() < 0:
            img = (img + 1.0) / 2.0
        img = img.clamp(0.0, 1.0)
        
        if img.dim() == 4:
            img_list = []
            for i in range(img.shape[0]):
                img_np = img[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype('uint8')
                pil_img = Image.fromarray(img_np)
                img_list.append(pil_img)
            img = img_list
        elif img.dim() == 3:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype('uint8')
            img = Image.fromarray(img_np)

    inputs = _clip_proc(images=img, return_tensors="pt")
    clip_input = inputs["pixel_values"].to(device)

    with torch.no_grad():
        out = _clip_model(pixel_values=clip_input)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
        B, N, C = tokens.shape
        if max_tokens is not None and N > max_tokens:
            factor = N // max_tokens
            tokens = tokens[:, :factor * max_tokens, :]
            tokens = tokens.reshape(B, max_tokens, factor, C).mean(dim=2)
        tokens = _proj(tokens)  # (1, N_style, out_dim)

    return tokens


class StyleSelfAttention(nn.Module):
    def __init__(self, dim=768, nhead=8, nlayers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, tokens):
        return self.encoder(tokens)


"""
# 내부에 있는 모듈(예: attention 클래스)의 멤버 정의 예
self.num_heads = num_heads
self.head_dim = head_dim
self.dim = num_heads * head_dim

# injected features -> k/v projection (사용자가 만든 모듈로 정의)
# 만약 injected_features가 (B, C, H, W)라면 1x1 conv + flatten을 사용해도 됨.
self.to_k_inj = nn.Linear(in_feat_dim, self.dim)  # 만약 feat flattened된 벡터라면 Linear
self.to_v_inj = nn.Linear(in_feat_dim, self.dim)

# --- 변환 함수 ---
def project_injected_to_kv(injected_features):
    # injected_features: [B, C, H, W] 또는 [B, seq_s, feat_dim]
    if injected_features.dim() == 4:
        B, C, H, W = injected_features.shape
        feat = injected_features.view(B, C, H*W).permute(0, 2, 1)  # [B, seq_s, feat_dim=C]
    else:
        # already [B, seq_s, feat_dim]
        feat = injected_features  # [B, seq_s, feat_dim]
    B, seq_s, feat_dim = feat.shape

    k_proj = self.to_k_inj(feat)  # [B, seq_s, dim]
    v_proj = self.to_v_inj(feat)  # [B, seq_s, dim]

    # reshape to multi-head: [B, seq_s, num_heads, head_dim] -> [B, num_heads, seq_s, head_dim]
    k_proj = k_proj.view(B, seq_s, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()
    v_proj = v_proj.view(B, seq_s, self.num_heads, self.head_dim).permute(0,2,1,3).contiguous()

    return {'k': k_proj, 'v': v_proj}  # shapes: [B, num_heads, seq_s, head_dim]
"""


if __name__ == "__main__":
    attn = StyleSelfAttention(dim=768, nhead=8, nlayers=2, dim_feedforward=2048, dropout=0.1)

    total_params = sum(p.numel() for p in attn.parameters())
    trainable_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)


"""
Total params: 11027968
Trainable params: 11027968
"""