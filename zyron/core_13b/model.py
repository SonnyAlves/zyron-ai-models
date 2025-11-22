import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention for Visual Brain integration.
    Allows the model to selectively attend to visual features.
    """
    def __init__(self, d_model, visual_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(visual_dim, d_model)
        self.value = nn.Linear(visual_dim, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.zeros(1)) # Initialize gate to 0
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, visual_features):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Expand visual features for attention
        v_feat = visual_features.unsqueeze(1) # [batch, 1, visual_dim]
        k = self.key(v_feat).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(v_feat).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection with gate
        return self.gate * self.out_proj(out)

class ZyronCore13BModel(nn.Module):
    """
    Zyron Core 13B Model Implementation
    General reasoning model with Visual Brain integration.
    """
    def __init__(self, config: Dict[str, Any], mode: str = "prod"):
        super().__init__()
        
        # Select configuration based on mode
        if mode == "dev":
            model_config = config.get("dev", {})
            print(f"   🔧 Initializing Core 13B in DEV mode: {model_config}")
        else:
            model_config = config.get("model", {})
            print(f"   🏭 Initializing Core 13B in PROD mode: {model_config}")

        # Extract hyperparameters
        self.d_model = model_config.get("d_model", 5120)
        self.n_layers = model_config.get("n_layers", 40)
        self.n_heads = model_config.get("n_heads", 40)
        self.vocab_size = model_config.get("vocab_size", 32000)
        self.max_seq_len = model_config.get("max_seq_len", 8192)
        
        # Visual Brain Config (Hardcoded for now as it's architectural)
        self.visual_dim = 2048
        self.use_visual_brain = True

        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=4 * self.d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        
        # Visual Brain Integration
        self.visual_attention = GatedCrossAttention(
            d_model=self.d_model,
            visual_dim=self.visual_dim,
            n_heads=self.n_heads // 4
        )
        
        # Final Projection
        self.norm_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, attention_mask=None, visual_features=None):
        """
        Forward pass with optional visual features
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Positions
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Visual Integration (Pre-Transformer)
        if visual_features is not None:
            visual_context = self.visual_attention(x, visual_features)
            x = x + visual_context
            
        # Causal Mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # Transformer
        x = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits
