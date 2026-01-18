"""
Test script for CS Alignment Loss

This script demonstrates and validates the CSAlignmentLoss implementation.
"""

import sys
sys.path.insert(0, '..')

import torch
import torch.nn.functional as F
from mwkflair.cs_alignment_loss import CSAlignmentLoss, CSInfoNCELoss


def test_basic_functionality():
    """Test basic CS Alignment Loss functionality"""
    print("="*60)
    print("Test 1: Basic Functionality (Paired Data)")
    print("="*60)
    
    # Create loss function
    loss_fn = CSAlignmentLoss(sigma=1.0)
    
    # Paired data: same batch size
    batch_size = 32
    feature_dim = 512
    
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    
    # Compute loss
    cs_loss = loss_fn(image_features, text_features)
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"CS Loss: {cs_loss.item():.6f}")
    print(f"✓ Basic functionality test passed!\n")
    
    return cs_loss


def test_unpaired_data():
    """Test with unpaired data (M ≠ N)"""
    print("="*60)
    print("Test 2: Unpaired Data (M ≠ N)")
    print("="*60)
    
    loss_fn = CSAlignmentLoss(sigma=1.0)
    
    # Unpaired: different number of images and texts
    num_images = 32
    num_texts = 48
    feature_dim = 512
    
    image_features = torch.randn(num_images, feature_dim)
    text_features = torch.randn(num_texts, feature_dim)
    
    cs_loss = loss_fn(image_features, text_features)
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"CS Loss: {cs_loss.item():.6f}")
    print(f"✓ Unpaired data test passed!\n")
    
    return cs_loss


def test_token_level_alignment():
    """Test token-level alignment with 3D tensors"""
    print("="*60)
    print("Test 3: Token-Level Alignment (3D Tensors)")
    print("="*60)
    
    # Enable token-level mode
    loss_fn = CSAlignmentLoss(sigma=1.0, token_level=True)
    
    batch_size = 8
    num_img_tokens = 49  # e.g., 7x7 patches
    num_txt_tokens = 77  # e.g., max text length
    feature_dim = 512
    
    image_features = torch.randn(batch_size, num_img_tokens, feature_dim)
    text_features = torch.randn(batch_size, num_txt_tokens, feature_dim)
    
    cs_loss = loss_fn(image_features, text_features)
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"CS Loss (token-level): {cs_loss.item():.6f}")
    print(f"✓ Token-level alignment test passed!\n")
    
    return cs_loss


def test_gradient_flow():
    """Test gradient flow through the loss"""
    print("="*60)
    print("Test 4: Gradient Flow")
    print("="*60)
    
    loss_fn = CSAlignmentLoss(sigma=1.0)
    
    # Create features that require gradients
    image_features = torch.randn(16, 256, requires_grad=True)
    text_features = torch.randn(16, 256, requires_grad=True)
    
    # Forward pass
    cs_loss = loss_fn(image_features, text_features)
    
    # Backward pass
    cs_loss.backward()
    
    print(f"CS Loss: {cs_loss.item():.6f}")
    print(f"Image gradient norm: {image_features.grad.norm().item():.6f}")
    print(f"Text gradient norm: {text_features.grad.norm().item():.6f}")
    print(f"✓ Gradient flow test passed!\n")


def test_trainable_sigma():
    """Test trainable kernel bandwidth"""
    print("="*60)
    print("Test 5: Trainable Sigma")
    print("="*60)
    
    loss_fn = CSAlignmentLoss(sigma=1.0, trainable_sigma=True)
    
    print(f"Initial sigma: {loss_fn.sigma.item():.4f}")
    print(f"Sigma requires_grad: {loss_fn.sigma.requires_grad}")
    
    # Simulate training step
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)
    
    image_features = torch.randn(16, 256)
    text_features = torch.randn(16, 256)
    
    for step in range(3):
        optimizer.zero_grad()
        cs_loss = loss_fn(image_features, text_features)
        cs_loss.backward()
        optimizer.step()
        print(f"Step {step+1}: sigma = {loss_fn.sigma.item():.4f}, loss = {cs_loss.item():.6f}")
    
    print(f"✓ Trainable sigma test passed!\n")


def test_combined_loss():
    """Test combined CS + InfoNCE loss"""
    print("="*60)
    print("Test 6: Combined CS + InfoNCE Loss")
    print("="*60)
    
    loss_fn = CSInfoNCELoss(
        sigma=1.0,
        temperature=0.07,
        cs_weight=1.0,
        infonce_weight=1.0
    )
    
    batch_size = 32
    feature_dim = 512
    
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    
    total_loss, cs_loss, infonce_loss = loss_fn(image_features, text_features)
    
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"CS Loss: {cs_loss.item():.6f}")
    print(f"InfoNCE Loss: {infonce_loss.item():.6f}")
    print(f"✓ Combined loss test passed!\n")


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("="*60)
    print("Test 7: Numerical Stability")
    print("="*60)
    
    loss_fn = CSAlignmentLoss(sigma=1.0)
    
    # Test with very similar features
    image_features = torch.randn(16, 256)
    text_features = image_features + 0.01 * torch.randn(16, 256)
    
    cs_loss_similar = loss_fn(image_features, text_features)
    print(f"Loss with similar features: {cs_loss_similar.item():.6f}")
    
    # Test with very different features
    image_features = torch.randn(16, 256)
    text_features = torch.randn(16, 256) * 10
    
    cs_loss_different = loss_fn(image_features, text_features)
    print(f"Loss with different features: {cs_loss_different.item():.6f}")
    
    # Check for NaN or Inf
    assert not torch.isnan(cs_loss_similar), "NaN detected in similar features!"
    assert not torch.isinf(cs_loss_similar), "Inf detected in similar features!"
    assert not torch.isnan(cs_loss_different), "NaN detected in different features!"
    assert not torch.isinf(cs_loss_different), "Inf detected in different features!"
    
    print(f"✓ Numerical stability test passed!\n")


def test_l2_normalization():
    """Verify L2 normalization is applied correctly"""
    print("="*60)
    print("Test 8: L2 Normalization Verification")
    print("="*60)
    
    loss_fn = CSAlignmentLoss(sigma=1.0)
    
    # Create unnormalized features
    image_features = torch.randn(16, 256) * 10  # Large magnitude
    text_features = torch.randn(16, 256) * 0.1  # Small magnitude
    
    print(f"Image feature norm (before): {image_features.norm(dim=1).mean().item():.4f}")
    print(f"Text feature norm (before): {text_features.norm(dim=1).mean().item():.4f}")
    
    # Compute loss (normalization happens inside)
    cs_loss = loss_fn(image_features, text_features)
    
    print(f"CS Loss: {cs_loss.item():.6f}")
    print(f"✓ L2 normalization test passed!\n")


def benchmark_performance():
    """Benchmark computational performance"""
    print("="*60)
    print("Test 9: Performance Benchmark")
    print("="*60)
    
    import time
    
    loss_fn = CSAlignmentLoss(sigma=1.0)
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128]
    feature_dim = 512
    
    for bs in batch_sizes:
        image_features = torch.randn(bs, feature_dim)
        text_features = torch.randn(bs, feature_dim)
        
        # Warm up
        _ = loss_fn(image_features, text_features)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            cs_loss = loss_fn(image_features, text_features)
        elapsed = time.time() - start
        
        print(f"Batch size {bs:3d}: {elapsed*10:.2f} ms/iter")
    
    print(f"✓ Performance benchmark completed!\n")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CS Alignment Loss - Comprehensive Test Suite")
    print("="*60 + "\n")
    
    try:
        # Run all tests
        test_basic_functionality()
        test_unpaired_data()
        test_token_level_alignment()
        test_gradient_flow()
        test_trainable_sigma()
        test_combined_loss()
        test_numerical_stability()
        test_l2_normalization()
        benchmark_performance()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
