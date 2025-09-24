import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPConfig, CLIPFeatureExtractor


class StyleEncoder(nn.Module):
    def __init__(self,
                 heads=8,
                 dim_head=64,
                 sty_alpha: float = 0.1,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 freeze_clip: bool = True,
                 device=None):
        super().__init__()
        self.sty_alpha = sty_alpha
        self.device = device

        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip.eval()
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        clip_feat_dim = self.clip.config.hidden_size
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(clip_feat_dim) # optional
        self.to_k_injected = nn.Linear(clip_feat_dim, inner_dim, bias=False)
        self.to_v_injected = nn.Linear(clip_feat_dim, inner_dim, bias=False)

        self.preprocess = transforms.Compose([
            transforms.Resize(self.clip.config.image_size if hasattr(self.clip.config, "image_size") else 224),
            transforms.CenterCrop(self.clip.config.image_size if hasattr(self.clip.config, "image_size") else 224),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, x):
        feat = x # clip vit 추가

        if feat.dim() == 4: # [B, C, H, W]
            B, C, H, W = feat.shape
            feat = feat.view(B, C, H*W).permute(0, 2, 1) # [B, seq_s, feat_dim=C]
        B, seq_s, feat_dim = feat.shape

        k_proj = self.to_k_injected(feat)
        v_proj = self.to_v_injected(feat)
        k_proj = k_proj.view(B, seq_s, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() # [B, seq_s, heads, dim_head] -> [B, heads, seq_s, head_dim]
        v_proj = v_proj.view(B, seq_s, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() # [B, seq_s, heads, dim_head] -> [B, heads, seq_s, head_dim]

        return {'k': k_proj, 'v': v_proj, 'sty_alpha': self.sty_alpha}


# import torch, torch.nn as nn
# from PIL import Image
# from transformers import CLIPVisionModel, CLIPImageProcessor


# _clip_model = None
# _clip_proc = None
# _proj = None


# def clip_vit_encoder(img, # PIL image or torch.Tensor
#                      model_name="openai/clip-vit-large-patch14",
#                      out_dim=768,
#                      max_tokens=64,
#                      device="cuda"):
#     global _clip_model, _clip_proc, _proj
#     device = torch.device(device)

#     if _clip_model is None:
#         _clip_model = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
#         _clip_proc = CLIPImageProcessor.from_pretrained(model_name)
#         hidden = _clip_model.config.hidden_size  # e.g., 1024 for ViT-L/14
#         _proj = nn.Identity() if hidden == out_dim else nn.Linear(hidden, out_dim, bias=False).to(device)
#         for p in list(_clip_model.parameters()) + list(_proj.parameters()):
#             p.requires_grad_(False)

#     if isinstance(img, torch.Tensor):
#         if img.min() < 0:
#             img = (img + 1.0) / 2.0
#         img = img.clamp(0.0, 1.0)
        
#         if img.dim() == 4:
#             img_list = []
#             for i in range(img.shape[0]):
#                 img_np = img[i].permute(1, 2, 0).cpu().numpy()
#                 img_np = (img_np * 255).astype('uint8')
#                 pil_img = Image.fromarray(img_np)
#                 img_list.append(pil_img)
#             img = img_list
#         elif img.dim() == 3:
#             img_np = img.permute(1, 2, 0).cpu().numpy()
#             img_np = (img_np * 255).astype('uint8')
#             img = Image.fromarray(img_np)

#     inputs = _clip_proc(images=img, return_tensors="pt")
#     clip_input = inputs["pixel_values"].to(device)

#     with torch.no_grad():
#         out = _clip_model(pixel_values=clip_input)
#         tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
#         B, N, C = tokens.shape
#         if max_tokens is not None and N > max_tokens:
#             factor = N // max_tokens
#             tokens = tokens[:, :factor * max_tokens, :]
#             tokens = tokens.reshape(B, max_tokens, factor, C).mean(dim=2)
#         tokens = _proj(tokens)  # (1, N_style, out_dim)

#     return tokens


# class StyleSelfAttention(nn.Module):
#     def __init__(self, dim=768, nhead=8, nlayers=2, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
#                                                    nhead=nhead,
#                                                    dim_feedforward=dim_feedforward,
#                                                    dropout=dropout,
#                                                    batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

#     def forward(self, tokens):
#         return self.encoder(tokens)


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