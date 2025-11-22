#!/usr/bin/env python3
"""
Zyron Training Orchestrator
Central script for training Zyron Finance 7B and Core 13B on DGX Spark.
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zyron.finance_7b.model import ZyronFinance7BModel
from zyron.core_13b.model import ZyronCore13BModel
from zyron.common.data import ZyronDataset

class ZyronOrchestrator:
    """Main orchestrator for Zyron models training"""

    def __init__(self, model_name: str, mode: str = "dev"):
        self.model_name = model_name
        self.mode = mode
        
        # Load Config
        self.config = self._load_config()
        
        # Setup Device
        self.device = self._setup_device()
        
        print(f"\n{'='*60}")
        print(f"🎯 Zyron Training Orchestrator")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Mode: {mode}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Load Model
        self.model = self._load_model()
        self.model.to(self.device)

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        config_path = Path(f"configs/training/{self.model_name.replace('zyron-', '').replace('-', '_')}.yaml")
        # Handle case where user passes "finance_7b" but file is "finance_7b.yaml"
        if not config_path.exists():
             # Try direct mapping if user passed "finance_7b"
             config_path = Path(f"configs/training/{self.model_name}.yaml")
             
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_device(self) -> str:
        """Setup compute device"""
        runtime_config = self.config.get("runtime", {})
        device_req = runtime_config.get("device", "auto")
        
        if device_req == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device_req

    def _load_model(self) -> torch.nn.Module:
        """Instantiate the correct model class"""
        if "finance-7b" in self.model_name or "finance_7b" in self.model_name:
            return ZyronFinance7BModel(self.config, mode=self.mode)
        elif "core-13b" in self.model_name or "core_13b" in self.model_name:
            return ZyronCore13BModel(self.config, mode=self.mode)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def train(self):
        """Execute training loop"""
        
        # Determine params based on mode
        if self.mode == "dev":
            params = self.config.get("dev", {})
            print("🚀 Running in DEV mode (Fast Check)")
        else:
            params = self.config.get("training", {})
            print("🚀 Running in PROD mode (Full Training)")

        max_steps = params.get("max_steps", 10)
        batch_size = params.get("global_batch_size", 1)
        seq_len = params.get("max_seq_len", 128)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(params.get("lr", 1e-3)))
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        
        print(f"\nStarting training for {max_steps} steps...")
        
        for step in range(max_steps):
            # Dummy data for v0.1 dev loop
            # In real implementation, this comes from ZyronDataset
            input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
            labels = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
            
            optimizer.zero_grad()
            
            # Forward
            logits = self.model(input_ids)
            
            # Loss (Flatten for CrossEntropy)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 1 == 0: # Log every step in dev
                perplexity = torch.exp(loss).item()
                print(f"Step {step+1}/{max_steps} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f}")

        print("\n✅ Training loop completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Zyron Training Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="finance_7b or core_13b")
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "prod"])
    
    args = parser.parse_args()
    
    orchestrator = ZyronOrchestrator(args.model, args.mode)
    orchestrator.train()

if __name__ == "__main__":
    main()
