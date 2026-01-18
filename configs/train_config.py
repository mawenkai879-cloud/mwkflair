"""
Training Configuration for MWK-FLAIR
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training configuration dataclass"""
    
    # Dataset
    data_root: str = './data'
    num_classes: int = 9
    
    # Model
    vision_backbone: str = 'resnet50'
    embed_dim: int = 512
    adapter_dim: int = 64
    use_bert: bool = False
    freeze_backbone: bool = True
    
    # HarMA
    use_harma: bool = True
    harma_reduction_dim: int = 128
    harma_num_heads: int = 8
    harma_dropout: float = 0.0
    
    # Training
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    margin: float = 0.2
    
    # Learning rate schedule
    lr_scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    cls_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.5
    
    # Checkpoints and logging
    save_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_freq: int = 5
    print_freq: int = 10
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Device
    device: str = 'cuda'
    
    # Random seed
    seed: int = 42


# Backward compatibility alias
Config = TrainConfig


def get_config(**kwargs) -> TrainConfig:
    """Get configuration object with optional overrides"""
    return TrainConfig(**kwargs)
