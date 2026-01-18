import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HarmaAdapter(nn.Module):
    """
    Harma Adapter for efficient fine-tuning
    Implements parameter-efficient adaptation layer
    """
    
    def __init__(
        self,
        in_features: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        init_scale: float = 1e-3
    ):
        """
        Args:
            in_features: Input feature dimension
            adapter_dim: Bottleneck dimension for adapter
            dropout: Dropout rate
            init_scale: Initialization scale for adapter weights
        """
        super(HarmaAdapter, self).__init__()
        
        self.in_features = in_features
        self.adapter_dim = adapter_dim
        
        # Down-projection
        self.down_proj = nn.Linear(in_features, adapter_dim)
        
        # Non-linearity
        self.activation = nn.GELU()
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_dim, in_features)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stability
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor with same shape as input
        """
        residual = x
        
        # Adapter transformation
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        
        # Residual connection
        return residual + x


class MultiHeadHarmaAdapter(nn.Module):
    """
    Multi-head Harma Adapter for richer feature adaptation
    """
    
    def __init__(
        self,
        in_features: int,
        num_heads: int = 4,
        adapter_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            in_features: Input feature dimension
            num_heads: Number of adapter heads
            adapter_dim: Bottleneck dimension per head
            dropout: Dropout rate
        """
        super(MultiHeadHarmaAdapter, self).__init__()
        
        self.num_heads = num_heads
        self.adapters = nn.ModuleList([
            HarmaAdapter(in_features, adapter_dim, dropout)
            for _ in range(num_heads)
        ])
        
        # Gating mechanism to combine heads
        self.gate = nn.Linear(in_features, num_heads)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-head adaptation
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor with same shape as input
        """
        # Compute gating weights
        gate_weights = F.softmax(self.gate(x), dim=-1)  # (batch, seq_len, num_heads)
        
        # Apply each adapter head
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_outputs.append(adapter(x))
        
        # Stack and weight by gates
        adapter_outputs = torch.stack(adapter_outputs, dim=-1)  # (batch, seq_len, in_features, num_heads)
        gate_weights = gate_weights.unsqueeze(-2)  # (batch, seq_len, 1, num_heads)
        
        # Weighted combination
        output = (adapter_outputs * gate_weights).sum(dim=-1)
        
        return output


class HarmaAdapterLayer(nn.Module):
    """
    Complete layer with Harma Adapter integration
    Can be inserted into transformer blocks
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_type: str = 'single',
        adapter_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: Hidden dimension size
            adapter_type: 'single' or 'multi' for adapter type
            adapter_dim: Adapter bottleneck dimension
            num_heads: Number of heads for multi-head adapter
            dropout: Dropout rate
        """
        super(HarmaAdapterLayer, self).__init__()
        
        if adapter_type == 'single':
            self.adapter = HarmaAdapter(hidden_size, adapter_dim, dropout)
        elif adapter_type == 'multi':
            self.adapter = MultiHeadHarmaAdapter(hidden_size, num_heads, adapter_dim, dropout)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with layer normalization
        
        Args:
            x: Input tensor
            
        Returns:
            Adapted output tensor
        """
        x = self.layer_norm(x)
        x = self.adapter(x)
        return x


def freeze_model_except_adapters(model: nn.Module):
    """
    Freeze all parameters except Harma adapters
    
    Args:
        model: PyTorch model containing Harma adapters
    """
    for name, param in model.named_parameters():
        if 'adapter' in name.lower() or 'harma' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    return trainable_params, total_params
