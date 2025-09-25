import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # Snapshot 16: Feed-forward input (inspect x) [for encoder]
        # Snapshot 37: Decoder feed-forward input (inspect x) [for decoder]
        linear1_out = self.linear_1(x)
        # Snapshot 17: Feed-forward first linear layer output (inspect linear1_out) [for encoder]
        # Snapshot 38: Feed-forward first linear layer output (inspect linear1_out) [for decoder]
        relu_out = torch.relu(linear1_out)
        drop_out = self.dropout(relu_out)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        linear2_out = self.linear_2(drop_out)
        # Snapshot 18: Feed-forward second linear layer output (inspect linear2_out) [for encoder]
        # Snapshot 39: Feed-forward second linear layer output (inspect linear2_out) [for decoder]
        return linear2_out


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        norm_out = self.norm(x)
        # Snapshot 15: Layer normalization output (inspect norm_out) [for encoder FF residual]
        # Also applicable for other residuals, but per snapshot list
        sub_out = sublayer(norm_out)
        drop_out = self.dropout(sub_out)
        res_out = x + drop_out
        # Snapshot 14: Residual connection tensors (inspect x and sub_out) [for encoder self-attn residual]
        # Snapshot 29: Residual + normalization after masked self-attention (inspect res_out) [for decoder self-attn]
        # Snapshot 36: Residual + normalization after cross-attention (inspect res_out) [for decoder cross-attn]
        return res_out


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # Snapshot 10: Attention score matrix before softmax (inspect attention_scores) [for encoder self-attn]
        # Snapshot 24: Masked attention scores before mask (inspect attention_scores) [for decoder masked self-attn]
        # Snapshot 33: Cross-attention score matrix before softmax (inspect attention_scores) [for decoder cross-attn]
        if mask is not None:
            # Snapshot 25: Mask tensor (inspect mask) [for decoder masked self-attn]
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        # Snapshot 11: Attention score matrix after softmax (inspect attention_scores) [for encoder self-attn]
        # Snapshot 26: Masked attention scores after mask + softmax (inspect attention_scores) [for decoder masked self-attn]
        # Snapshot 34: Cross-attention score matrix after softmax (inspect attention_scores) [for decoder cross-attn]
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        attn_out = attention_scores @ value
        return attn_out, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # Snapshot 7: Self-attention queries (Q) (inspect query) [for encoder self-attn]
        # Snapshot 21: Masked self-attention queries (Q) (inspect query) [for decoder self-attn]
        # Snapshot 30: Cross-attention queries (from decoder) (inspect query) [for decoder cross-attn]
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # Snapshot 8: Self-attention keys (K) (inspect key) [for encoder self-attn]
        # Snapshot 22: Masked self-attention keys (K) (inspect key) [for decoder self-attn]
        # Snapshot 31: Cross-attention keys (from encoder) (inspect key) [for decoder cross-attn]
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        # Snapshot 9: Self-attention values (V) (inspect value) [for encoder self-attn]
        # Snapshot 23: Masked self-attention values (V) (inspect value) [for decoder self-attn]
        # Snapshot 32: Cross-attention values (from encoder) (inspect value) [for decoder cross-attn]

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # Snapshot 12: Multi-head split (Q/K/V split) (inspect query, key, value) [for encoder self-attn]
        # Snapshot 27: Masked self-attention multi-head split (inspect query, key, value) [for decoder self-attn]
        # For cross-attn, similar split but no specific snapshot number

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        # Snapshot 13: Multi-head attention output after concatenation (inspect x) [for encoder self-attn]
        # Snapshot 28: Masked self-attention multi-head concatenated output (inspect x) [for decoder self-attn]
        # Snapshot 35: Cross-attention output after concatenation (inspect x) [for decoder cross-attn]

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Snapshot 6: Encoder block input tensor (inspect x)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        # Snapshot 19: Encoder block final output tensor (inspect x)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Snapshot 20: Decoder block input tensor (inspect x)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        # Snapshot 40: Decoder block final output tensor (inspect x)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        # Snapshot 4: Input embeddings after lookup (inspect src)
        src = self.src_pos(src)
        # Snapshot 5: Embeddings after adding positional encoding (inspect src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        # Snapshot 41: Decoder final sequence output (before projection) (inspect decoder_output)
        logits = self.project(decoder_output)
        # Snapshot 42: Logits after final linear projection (inspect logits)
        # Snapshot 43: Logits slice (first few values for one token) (inspect e.g., logits[0, 0, :5])
        return logits


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def get_tgt_mask(size) -> torch.tensor:
    # Generates a lower triangular matrix for causal masking
    mask = torch.tril(torch.ones((size, size), dtype=torch.bool)).view(1, 1, size, size)
    return mask


if __name__ == "__main__":
    # Reduced model as per requirements: 2 layers, 4 heads, d_model=128
    d_model = 128
    N = 2
    h = 4
    d_ff = 512  # Example, can adjust
    dropout = 0.1
    vocab_size = 10000  # Dummy vocab size
    seq_len = 8  # 5-12 tokens

    model = build_transformer(vocab_size, vocab_size, seq_len, seq_len, d_model=d_model, N=N, h=h, d_ff=d_ff, dropout=dropout)

    # Dummy input: unique pair, e.g., programming domain: token IDs for "print('Hello world')" as input, target as next or translation.
    # For simplicity, use arange for token IDs.
    batch_size = 1
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Raw input tokens (IDs)
    # Snapshot 1: Raw input tokens (IDs or text) (inspect src)
    tgt = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])  # Target tokens (shifted or whatever)
    # Snapshot 2: Target tokens (IDs or text) (inspect tgt)

    # Snapshot 3: Embedding weight matrix (slice, e.g., 5x5) (inspect model.src_embed.embedding.weight[:5, :5])

    # Masks: src_mask None (no padding), tgt_mask causal
    src_mask = None  # Assume no padding
    tgt_mask = get_tgt_mask(seq_len)

    # Call forward
    logits = model(src, tgt, src_mask, tgt_mask)

    print("End")