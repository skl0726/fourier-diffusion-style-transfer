import torch, torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor


_clip_model = None
_clip_proc = None
_proj = None


def get_style_tokens(img: Image.Image,
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

    inputs = _clip_proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = _clip_model(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
        B, N, C = tokens.shape
        if max_tokens is not None and N > max_tokens:
            factor = N // max_tokens
            tokens = tokens[:, :factor * max_tokens, :]
            tokens = tokens.reshape(B, max_tokens, factor, C).mean(dim=2)
        style_tokens = _proj(tokens)  # (1, N_style, out_dim)

    return style_tokens
