import torch
import torch.nn as nn
from typing import Dict, Any

class ZyronFinance7BModel(nn.Module):
    """
    Zyron Finance 7B Model Implementation
    Specialized for financial operations (VAT, Invoicing).
    """
    def __init__(self, config: Dict[str, Any], mode: str = "prod"):
        super().__init__()
        
        # Select configuration based on mode
        if mode == "dev":
            model_config = config.get("dev", {})
            print(f"   🔧 Initializing in DEV mode: {model_config}")
        else:
            model_config = config.get("model", {})
            print(f"   🏭 Initializing in PROD mode: {model_config}")

        # Extract hyperparameters
        self.d_model = model_config.get("d_model", 4096)
        self.n_layers = model_config.get("n_layers", 32)
        self.n_heads = model_config.get("n_heads", 32)
        self.vocab_size = model_config.get("vocab_size", 32000)
        self.max_seq_len = model_config.get("max_seq_len", 8192)
        
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
            norm_first=True # Pre-LN is standard for LLMs
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
        
        # Final Projection
        self.norm_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Weight tying (optional but common)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Causal Mask
        # Generate square subsequent mask for causal attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # Pass through Transformer
        x = self.transformer(
            tgt=x,
            memory=x, # Self-attention only
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits
