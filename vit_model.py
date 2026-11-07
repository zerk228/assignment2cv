import torch
import torch.nn as nn

IMG_SIZE = 64
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
EMBED_DIM = 64
DEPTH = 4
HEADS = 4
MLP_DIM = 128


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, HEADS, batch_first=True)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, MLP_DIM),
            nn.GELU(),
            nn.Linear(MLP_DIM, EMBED_DIM),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed, need_weights=True, average_attn_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, EMBED_DIM))
        self.blocks = nn.ModuleList([AttentionBlock() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, num_classes)

    def forward(self, x, return_attn=False):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)

        x = self.norm(x)
        logits = self.head(x[:, 0])

        if return_attn:
            return logits, attn_maps
        return logits


def attention_rollout(attn_maps):
    result = torch.eye(attn_maps[0].shape[-1]).unsqueeze(0).to(attn_maps[0].device)
    for attn in attn_maps:
        attn = attn + torch.eye(attn.shape[-1]).to(attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn, result)
    cls_to_patches = result[:, 0, 1:]
    size = int(cls_to_patches.shape[-1] ** 0.5)
    return cls_to_patches.reshape(-1, size, size)
