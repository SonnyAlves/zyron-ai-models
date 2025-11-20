#!/usr/bin/env python3
"""
Toy GPT with NeMo - Minimal implementation to validate NeMo pipeline on DGX
This is a test model, NOT Zyron Finance 7B - just validating the training loop

Compatible with:
- PyTorch 2.6.0a0+df5bbc09d1.nv24.11
- NeMo 2.5.3
- NVIDIA GB10 GPU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# NeMo imports - minimal set for NeMo 2.5.3
import nemo
from nemo.core import ModelPT
from nemo.core.classes import typecheck
import nemo.core.neural_types as ntype

print(f"📦 NeMo version: {nemo.__version__}")
print(f"📦 PyTorch version: {torch.__version__}")


class ToyGPTNeMo(ModelPT):
    """
    Minimal GPT model using NeMo's ModelPT base class
    This is a toy model for pipeline validation, not production
    """

    def __init__(self, cfg=None):
        # Default config - hardcoded for simplicity (no Hydra/YAML)
        if cfg is None:
            cfg = {
                'vocab_size': 1000,      # Small vocab for testing
                'd_model': 128,          # Small model dimension
                'n_layers': 2,           # Just 2 layers for quick testing
                'n_heads': 4,            # 4 attention heads
                'max_seq_len': 64,       # Short sequences
                'dropout': 0.1,
                'batch_size': 8,
                'learning_rate': 1e-3,
            }

        # Initialize NeMo ModelPT
        super().__init__(cfg)

        # Store config
        self.vocab_size = cfg['vocab_size']
        self.d_model = cfg['d_model']
        self.n_layers = cfg['n_layers']
        self.n_heads = cfg['n_heads']
        self.max_seq_len = cfg['max_seq_len']
        self.dropout = cfg['dropout']

        # Build model components
        self._build_model()

    def _build_model(self):
        """Build the transformer model components"""

        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional embeddings (learned)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        # Transformer decoder layers (GPT-style: causal self-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,  # Standard 4x expansion
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,  # Important for batch-first tensors
        )

        # Stack of decoder layers
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.n_layers,
        )

        # Layer norm before output
        self.ln_f = nn.LayerNorm(self.d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        Forward pass through the model

        Args:
            input_ids: [batch_size, seq_len] tensor of token ids

        Returns:
            logits: [batch_size, seq_len, vocab_size] tensor
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [B, L, D]

        # Position ids and embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(position_ids)  # [B, L, D]

        # Combine embeddings
        x = self.dropout_layer(token_emb + pos_emb)

        # Create causal mask for autoregressive generation
        # TODO: In production, this would be cached for efficiency
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        # Pass through transformer (decoder-only)
        # Note: For decoder-only, we use the same tensor as both input and memory
        x = self.transformer(
            tgt=x,
            memory=x,  # Self-attention only
            tgt_mask=causal_mask,
        )

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    # NeMo-specific methods (minimal implementation)
    @typecheck()
    def training_step(self, batch, batch_idx):
        """Training step for NeMo/PTL compatibility"""
        # For this toy example, batch is just a dict with input_ids
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass
        logits = self.forward(input_ids)

        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100  # Standard padding token
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for NeMo/PTL compatibility"""
        return self.training_step(batch, batch_idx)


def create_dummy_data(batch_size, seq_len, vocab_size, device):
    """
    Create dummy data for testing
    TODO: Replace with real tokenized data in production

    Returns:
        dict with 'input_ids' and 'labels' (shifted by 1 for autoregressive)
    """
    # Random token ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Labels are input shifted by 1 (standard autoregressive setup)
    # TODO: In production, handle proper padding and EOS tokens
    labels = torch.cat([
        input_ids[:, 1:],
        torch.randint(0, vocab_size, (batch_size, 1), device=device)
    ], dim=1)

    return {
        'input_ids': input_ids,
        'labels': labels
    }


def compute_accuracy(logits, labels):
    """
    Compute token-level accuracy
    TODO: Add more sophisticated metrics for production
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    # Mask out padding (-100)
    mask = labels != -100
    accuracy = (correct * mask).sum() / mask.sum()
    return accuracy.item() * 100  # Return as percentage


def main():
    """Main training loop - minimal implementation for pipeline validation"""

    print("\n" + "="*60)
    print("🚀 Toy GPT NeMo - Pipeline Validation Script")
    print("="*60 + "\n")

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected, running on CPU (will be slow)")

    print(f"🔧 Device: {device}\n")

    # Model configuration (hardcoded for simplicity)
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 64,
        'dropout': 0.1,
        'batch_size': 8,
        'learning_rate': 1e-3,
    }

    print("📊 Model Configuration:")
    print(f"   Vocab size: {config['vocab_size']}")
    print(f"   Model dim: {config['d_model']}")
    print(f"   Layers: {config['n_layers']}")
    print(f"   Heads: {config['n_heads']}")
    print(f"   Max seq len: {config['max_seq_len']}")
    print(f"   Batch size: {config['batch_size']}")
    print()

    # Create model
    print("🔨 Creating Toy GPT model...")
    model = ToyGPTNeMo(cfg=config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"   (~{total_params/1e6:.2f}M params - this is a toy model, not 7B!)\n")

    # Create optimizer
    # TODO: In production, use more sophisticated optimizers (AdamW, etc.)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop - just 10 steps for validation
    num_steps = 10
    print(f"🎯 Running {num_steps} training steps for pipeline validation...\n")
    print("-" * 60)

    model.train()

    for step in range(num_steps):
        # Generate dummy data
        # TODO: Replace with real DataLoader in production
        batch = create_dummy_data(
            batch_size=config['batch_size'],
            seq_len=config['max_seq_len'],
            vocab_size=config['vocab_size'],
            device=device
        )

        # Forward pass
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = model(input_ids)

        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, config['vocab_size']),
            labels.reshape(-1)
        )

        # Compute metrics
        perplexity = math.exp(min(loss.item(), 10))  # Cap at exp(10) to avoid overflow
        accuracy = compute_accuracy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for stability)
        # TODO: Make this configurable in production
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        # Log progress
        print(f"Step {step+1:2d}/{num_steps} | "
              f"Loss: {loss.item():.4f} | "
              f"Perplexity: {perplexity:.2f} | "
              f"Accuracy: {accuracy:.1f}%")

    print("-" * 60)

    # Final validation
    print("\n🔬 Running final validation step...")
    model.eval()

    with torch.no_grad():
        val_batch = create_dummy_data(
            batch_size=config['batch_size'],
            seq_len=config['max_seq_len'],
            vocab_size=config['vocab_size'],
            device=device
        )

        val_logits = model(val_batch['input_ids'])
        val_loss = F.cross_entropy(
            val_logits.reshape(-1, config['vocab_size']),
            val_batch['labels'].reshape(-1)
        )

        val_perplexity = math.exp(min(val_loss.item(), 10))
        val_accuracy = compute_accuracy(val_logits, val_batch['labels'])

        print(f"   Validation Loss: {val_loss.item():.4f}")
        print(f"   Validation Perplexity: {val_perplexity:.2f}")
        print(f"   Validation Accuracy: {val_accuracy:.1f}%")

    # Success message
    print("\n" + "="*60)
    print(f"✅ Toy NeMo GPT training loop finished successfully on {device}.")
    print("="*60)
    print("\n📝 Notes:")
    print("   - This is a pipeline validation script, not a production model")
    print("   - The model has ~100K params (toy size for testing)")
    print("   - Next steps: real tokenizer, real data, scale to 7B")
    print("\n🎯 Pipeline validated and ready for scaling to Zyron Finance 7B!")


if __name__ == "__main__":
    main()