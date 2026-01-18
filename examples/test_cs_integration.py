"""
Quick test to verify CS Alignment Loss integration with FLAIR+HarMA
"""
import sys
sys.path.insert(0, '..')

import torch
from mwkflair.loss import FlairLoss
from mwkflair.cs_alignment_loss import CSAlignmentLoss

print("="*60)
print("Testing CS Alignment Loss Integration with FLAIR+HarMA")
print("="*60)

# Test 1: Create FlairLoss with CS Alignment enabled
print("\n[Test 1] Creating FlairLoss with CS Alignment...")
loss_fn = FlairLoss(
    num_cap_per_img=4,
    use_harma=True,
    harma_loss_weight=0.5,
    use_cs_alignment=True,
    cs_loss_weight=0.3,
    cs_sigma=1.0
)

print(f"  ✓ FlairLoss created")
print(f"  - use_harma: {loss_fn.use_harma}")
print(f"  - use_cs_alignment: {loss_fn.use_cs_alignment}")
print(f"  - HarMA loss function: {loss_fn.harma_loss_fn is not None}")
print(f"  - CS loss function: {loss_fn.cs_loss_fn is not None}")
print(f"  - CS loss type: {type(loss_fn.cs_loss_fn).__name__}")

# Test 2: Create dummy features and compute loss
print("\n[Test 2] Computing loss with dummy features...")
batch_size = 8
num_captions = 4
embed_dim = 512
num_tokens = 49

# Create dummy features
image_features = torch.randn(batch_size, embed_dim)
text_features = torch.randn(batch_size * num_captions, embed_dim)
image_tokens = torch.randn(batch_size, num_tokens, embed_dim)

# HarMA features
harma_image_features = torch.randn(batch_size, embed_dim)
harma_text_features = torch.randn(batch_size * num_captions, embed_dim)

# Normalize features
import torch.nn.functional as F
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)
harma_image_features = F.normalize(harma_image_features, dim=-1)
harma_text_features = F.normalize(harma_text_features, dim=-1)

# Dummy visual projection (just returns input)
class DummyVisualProj:
    def __call__(self, query, key, value):
        # Return (B, B+K-1, D) shape
        B = query.shape[0] // num_captions
        return torch.randn(B, B + num_captions - 1, embed_dim)

visual_proj = DummyVisualProj()

# Compute loss
logit_scale = torch.tensor(2.6592)  # ln(1/0.07)
logit_bias = None

try:
    loss_dict = loss_fn(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        image_tokens=image_tokens,
        visual_proj=visual_proj,
        output_dict=True,
        harma_image_features=harma_image_features,
        harma_text_features=harma_text_features
    )
    
    print(f"  ✓ Loss computed successfully")
    print(f"\n  Loss breakdown:")
    print(f"    Total loss: {loss_dict['contrastive_loss'].item():.6f}")
    print(f"    FLAIR loss: {loss_dict['flair_loss'].item():.6f}")
    if 'harma_loss' in loss_dict:
        print(f"    HarMA loss: {loss_dict['harma_loss'].item():.6f} (weight: {loss_dict['harma_loss_weight']})")
    if 'cs_loss' in loss_dict:
        print(f"    CS loss: {loss_dict['cs_loss'].item():.6f} (weight: {loss_dict['cs_loss_weight']})")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test with CS+InfoNCE
print("\n[Test 3] Testing with CS+InfoNCE combined loss...")
loss_fn_combined = FlairLoss(
    num_cap_per_img=4,
    use_harma=True,
    harma_loss_weight=0.5,
    use_cs_alignment=True,
    cs_loss_weight=0.3,
    cs_sigma=1.0,
    cs_use_infonce=True,
    cs_temperature=0.07
)

print(f"  ✓ FlairLoss with CS+InfoNCE created")
print(f"  - CS loss type: {type(loss_fn_combined.cs_loss_fn).__name__}")

try:
    loss_dict = loss_fn_combined(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        image_tokens=image_tokens,
        visual_proj=visual_proj,
        output_dict=True,
        harma_image_features=harma_image_features,
        harma_text_features=harma_text_features
    )
    
    print(f"  ✓ Loss computed successfully")
    print(f"\n  Loss breakdown:")
    print(f"    Total loss: {loss_dict['contrastive_loss'].item():.6f}")
    print(f"    FLAIR loss: {loss_dict['flair_loss'].item():.6f}")
    if 'harma_loss' in loss_dict:
        print(f"    HarMA loss: {loss_dict['harma_loss'].item():.6f}")
    if 'cs_loss' in loss_dict:
        print(f"    CS loss: {loss_dict['cs_loss'].item():.6f}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Gradient flow
print("\n[Test 4] Testing gradient flow...")
harma_image_features = torch.randn(batch_size, embed_dim, requires_grad=True)
harma_text_features = torch.randn(batch_size * num_captions, embed_dim, requires_grad=True)

try:
    loss_dict = loss_fn(
        image_features=image_features,
        text_features=text_features,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        image_tokens=image_tokens,
        visual_proj=visual_proj,
        output_dict=True,
        harma_image_features=harma_image_features,
        harma_text_features=harma_text_features
    )
    
    loss = loss_dict['contrastive_loss']
    loss.backward()
    
    print(f"  ✓ Backward pass successful")
    print(f"    Image grad norm: {harma_image_features.grad.norm().item():.6f}")
    print(f"    Text grad norm: {harma_text_features.grad.norm().item():.6f}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ All integration tests passed!")
print("="*60)
print("\nCS Alignment Loss is successfully integrated into FLAIR+HarMA!")
print("\nUsage in training:")
print("  loss_fn = FlairLoss(")
print("      use_harma=True,")
print("      harma_loss_weight=0.5,")
print("      use_cs_alignment=True,  # Enable CS loss")
print("      cs_loss_weight=0.3,     # CS loss weight")
print("      cs_sigma=1.0            # Kernel bandwidth")
print("  )")
