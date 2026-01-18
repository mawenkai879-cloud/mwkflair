"""
FLAIR + HarMA + CS Alignment Loss Training Example

This script demonstrates how to train a FLAIR model with:
1. HarMA adapters for parameter-efficient fine-tuning
2. CS Alignment Loss for distributional feature alignment
3. Combined loss: FLAIR + HarMA Triplet + CS Divergence

Usage:
    python examples/train_flair_harma_cs.py --use_cs_alignment --cs_loss_weight 0.3
"""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm

from mwkflair import (
    FLAIR,
    CLIPVisionCfg,
    CLIPTextCfg,
    freeze_model_except_adapters
)
from mwkflair.loss import FlairLoss


class DummyImageTextDataset(Dataset):
    """Dummy dataset for demonstration"""
    def __init__(self, num_samples=1000, num_captions_per_image=4):
        self.num_samples = num_samples
        self.num_captions = num_captions_per_image
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Dummy image (3, 224, 224)
        image = torch.randn(3, 224, 224)
        # Dummy text tokens (num_captions, 77)
        texts = torch.randint(0, 49408, (self.num_captions, 77))
        return image, texts


def create_model_with_harma(use_harma=True, harma_reduction_dim=128):
    """Create FLAIR model with HarMA adapters"""
    
    vision_cfg = CLIPVisionCfg(
        image_size=224,
        patch_size=16,
        width=768,
        layers=12,
        output_tokens=True  # Important for FLAIR
    )
    
    text_cfg = CLIPTextCfg(
        context_length=77,
        vocab_size=49408,
        width=512,
        heads=8,
        layers=12
    )
    
    model = FLAIR(
        embed_dim=512,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        use_harma=use_harma,
        harma_reduction_dim=harma_reduction_dim,
        harma_num_heads=8,
        harma_dropout=0.0
    )
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    flair_loss_sum = 0
    harma_loss_sum = 0
    cs_loss_sum = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, texts) in enumerate(pbar):
        images = images.to(device)
        texts = texts.to(device)
        
        # Flatten texts: (B, K, L) -> (B*K, L)
        batch_size, num_captions, seq_len = texts.shape
        texts_flat = texts.view(-1, seq_len)
        
        # Forward pass
        outputs = model(images, texts_flat)
        
        # Extract features
        image_features = outputs['image_features']  # (B, D)
        text_features = outputs['text_features']    # (B*K, D)
        image_tokens = outputs.get('image_tokens', None)  # (B, L, D)
        
        # HarMA features (if available)
        harma_image_features = outputs.get('harma_image_features', None)
        harma_text_features = outputs.get('harma_text_features', None)
        
        # Compute loss
        loss_dict = criterion(
            image_features=image_features,
            text_features=text_features,
            logit_scale=model.logit_scale.exp(),
            logit_bias=model.logit_bias,
            image_tokens=image_tokens,
            visual_proj=model.visual.post_process if hasattr(model.visual, 'post_process') else None,
            output_dict=True,
            harma_image_features=harma_image_features,
            harma_text_features=harma_text_features
        )
        
        loss = loss_dict['contrastive_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        flair_loss_sum += loss_dict.get('flair_loss', 0).item() if isinstance(loss_dict.get('flair_loss', 0), torch.Tensor) else 0
        
        if 'harma_loss' in loss_dict:
            harma_loss_sum += loss_dict['harma_loss'].item()
        if 'cs_loss' in loss_dict:
            cs_loss_sum += loss_dict['cs_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'flair': f"{loss_dict.get('flair_loss', 0):.4f}" if isinstance(loss_dict.get('flair_loss', 0), torch.Tensor) else "0",
            'harma': f"{loss_dict.get('harma_loss', 0):.4f}" if 'harma_loss' in loss_dict else "N/A",
            'cs': f"{loss_dict.get('cs_loss', 0):.4f}" if 'cs_loss' in loss_dict else "N/A"
        })
        
        if batch_idx >= 10:  # Quick demo, only run 10 batches
            break
    
    num_batches = min(batch_idx + 1, len(dataloader))
    
    return {
        'total_loss': total_loss / num_batches,
        'flair_loss': flair_loss_sum / num_batches,
        'harma_loss': harma_loss_sum / num_batches if harma_loss_sum > 0 else 0,
        'cs_loss': cs_loss_sum / num_batches if cs_loss_sum > 0 else 0
    }


def main(args):
    """Main training function"""
    
    print("="*60)
    print("FLAIR + HarMA + CS Alignment Training")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create model
    print("\n[1] Creating model...")
    model = create_model_with_harma(
        use_harma=args.use_harma,
        harma_reduction_dim=args.harma_reduction_dim
    )
    model = model.to(device)
    
    # Freeze backbone if needed
    if args.freeze_backbone:
        print("\n[2] Freezing backbone, training only adapters...")
        freeze_model_except_adapters(model)
    
    # Print parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Create loss function
    print("\n[3] Creating loss function...")
    criterion = FlairLoss(
        num_cap_per_img=args.num_captions,
        added_mps_loss=args.use_mps_loss,
        use_harma=args.use_harma,
        harma_loss_weight=args.harma_loss_weight,
        harma_margin=args.harma_margin,
        use_cs_alignment=args.use_cs_alignment,
        cs_loss_weight=args.cs_loss_weight,
        cs_sigma=args.cs_sigma,
        cs_use_infonce=args.cs_use_infonce,
        cs_temperature=args.cs_temperature
    )
    
    print(f"\nLoss Configuration:")
    print(f"  FLAIR base loss: ✓")
    print(f"  HarMA Triplet: {'✓' if args.use_harma else '✗'} (weight={args.harma_loss_weight})")
    print(f"  CS Alignment: {'✓' if args.use_cs_alignment else '✗'} (weight={args.cs_loss_weight})")
    if args.use_cs_alignment:
        print(f"    - Sigma: {args.cs_sigma}")
        print(f"    - Use InfoNCE: {args.cs_use_infonce}")
    
    # Create dataset and dataloader
    print("\n[4] Creating dataset...")
    dataset = DummyImageTextDataset(
        num_samples=args.num_samples,
        num_captions_per_image=args.num_captions
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create optimizer
    print("\n[5] Creating optimizer...")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("\n[6] Starting training...")
    print("="*60)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, device, args
        )
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  FLAIR Loss: {metrics['flair_loss']:.4f}")
        if metrics['harma_loss'] > 0:
            print(f"  HarMA Loss: {metrics['harma_loss']:.4f}")
        if metrics['cs_loss'] > 0:
            print(f"  CS Loss: {metrics['cs_loss']:.4f}")
    
    print("\n" + "="*60)
    print("✅ Training completed!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FLAIR with HarMA and CS Alignment')
    
    # Model args
    parser.add_argument('--use_harma', action='store_true', default=True,
                        help='Use HarMA adapters')
    parser.add_argument('--harma_reduction_dim', type=int, default=128,
                        help='HarMA adapter reduction dimension')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze backbone and train only adapters')
    
    # Loss args
    parser.add_argument('--use_mps_loss', action='store_true', default=False,
                        help='Use multi-positive sigmoid loss')
    parser.add_argument('--harma_loss_weight', type=float, default=0.5,
                        help='Weight for HarMA triplet loss')
    parser.add_argument('--harma_margin', type=float, default=0.2,
                        help='Margin for HarMA triplet loss')
    
    # CS Alignment args
    parser.add_argument('--use_cs_alignment', action='store_true', default=False,
                        help='Use CS Alignment Loss')
    parser.add_argument('--cs_loss_weight', type=float, default=0.3,
                        help='Weight for CS alignment loss')
    parser.add_argument('--cs_sigma', type=float, default=1.0,
                        help='Kernel bandwidth for CS divergence')
    parser.add_argument('--cs_use_infonce', action='store_true', default=False,
                        help='Use combined CS+InfoNCE loss')
    parser.add_argument('--cs_temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE (if cs_use_infonce=True)')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_captions', type=int, default=4,
                        help='Number of captions per image')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of training samples')
    
    args = parser.parse_args()
    
    main(args)
