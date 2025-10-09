from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn.functional import softmax


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, num_patches, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, num_patches, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, num_patches, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # Snapshot 16: Feed-forward input (inspect x)
        linear1_out = self.linear_1(x)
        # Snapshot 17: Feed-forward hidden layer output (inspect linear1_out)
        relu_out = torch.relu(linear1_out)
        drop_out = self.dropout(relu_out)
        linear2_out = self.linear_2(drop_out)
        # Snapshot 18: Feed-forward output after second linear (inspect linear2_out)
        return linear2_out


class PatchEmbedding(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size * in_channels, d_model)

    def forward(self, x):
        # x: (batch, channels, height, width)
        # Snapshot 1: Raw input image tensor (inspect x)
        # Unfold into patches: (batch, channels, num_patches_height, num_patches_width, patch_size, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # Snapshot 2: Image divided into patches (inspect x)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H/p, W/p, C, p, p)
        x = x.view(x.shape[0], self.num_patches, -1)  # (B, num_patches, p*p*C)
        # Snapshot 3: Flattened patches (inspect x)
        x = self.projection(x)  # (batch, num_patches, d_model)
        # Snapshot 4: Patch embeddings after linear projection (inspect x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, num_patches: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, num_patches + 1, d_model))  # +1 for class token

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        # Snapshot 7: Embeddings after adding positional encoding (inspect x)
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        norm_out = self.norm(x)
        sub_out = sublayer(norm_out)
        drop_out = self.dropout(sub_out)
        res_out = x + drop_out
        # Snapshot 15: Residual connection + normalization (post-attention) (inspect res_out)
        # Snapshot 19: Residual connection + normalization (post-MLP) (inspect res_out)
        return res_out


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # Snapshot 12: Attention scores before softmax (inspect attention_scores)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        # Snapshot 13: Attention scores after softmax (inspect attention_scores)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        attn_out = attention_scores @ value
        return attn_out, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        # Snapshot 9: Multi-head attention queries (Q) (inspect query)
        key = self.w_k(k)
        # Snapshot 10: Multi-head attention keys (K) (inspect key)
        value = self.w_v(v)
        # Snapshot 11: Multi-head attention values (V) (inspect value)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # Snapshot 14: Multi-head attention output (after concatenation) (inspect x)
        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        # Snapshot 8: Encoder block input tensor (inspect x)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        # Snapshot 20: Encoder block final output (inspect x)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            # Snapshot 21: Encoder block 2 output (inspect x) - after second layer
            if i == 1:
                pass  # Snapshot point
            # Snapshot 22: Encoder block N (last block) output (inspect x) - after last layer
            if i == len(self.layers) - 1:
                pass  # Snapshot point
        x = self.norm(x)
        # Snapshot 23: Final sequence output (including class token) (inspect x)
        return x


class ClassificationHead(nn.Module):

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x[:, 0, :]  # (batch, d_model)
        # Snapshot 24: Class token extracted (final representation) (inspect x)
        x = self.proj(x)
        # Snapshot 25: Classification head logits (inspect x)
        return x


class VisionTransformer(nn.Module):

    def __init__(self, encoder: Encoder, patch_embed: PatchEmbedding,
                 pos_embed: PositionalEncoding, classification_head: ClassificationHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.classification_head = classification_head
        self.class_token = nn.Parameter(torch.randn(1, 1, patch_embed.projection.out_features))

    def forward(self, x):
        # Embed patches
        patches = self.patch_embed(x)  # (batch, num_patches, d_model)
        B = patches.shape[0]
        class_tokens = self.class_token.expand(B, -1, -1)  # (batch, 1, d_model)
        # Snapshot 5: Class token before concatenation (inspect class_tokens)
        x = torch.cat((class_tokens, patches), dim=1)  # (batch, num_patches + 1, d_model)
        # Snapshot 6: Embeddings after adding the class token (inspect x)
        x = self.pos_embed(x)
        encoder_output = self.encoder(x, mask=None)
        logits = self.classification_head(encoder_output)
        # Snapshot 26: Softmax probabilities (example slice) (inspect torch.softmax(logits, dim=-1)[:5])
        logits = logits.softmax(dim=-1)
        return logits


def build_vit(img_size: int = 224, patch_size: int = 16, in_channels: int = 3, num_classes: int = 1000,
              d_model: int = 768, N: int = 2, h: int = 12, dropout: float = 0.1, d_ff: int = 3072) -> VisionTransformer:
    patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
    pos_embed = PositionalEncoding(d_model, patch_embed.num_patches, dropout)
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    classification_head = ClassificationHead(d_model, num_classes)
    vit = VisionTransformer(encoder, patch_embed, pos_embed, classification_head)
    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return vit


if __name__ == "__main__":
    img = Image.open('IMG.png').convert('RGB')
    img_array = torch.Tensor(np.array(img)).unsqueeze(0)
    model = build_vit()
    img_array = img_array.transpose(1, 3)
    logits = model(img_array)
    print("End")