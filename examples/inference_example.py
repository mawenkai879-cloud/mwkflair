"""
MWK-FLAIR Inference Example
Demonstrates how to use the model for image-text similarity computation.
"""
import torch
import torch.nn.functional as F
from PIL import Image

# Import from mwkflair package
import sys
sys.path.insert(0, '..')

from mwkflair import (
    create_model_and_transforms,
    get_tokenizer,
    HarmaAdapter,
    freeze_model_except_adapters
)


def main():
    """Example inference with MWK-FLAIR"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with HarMA adapters
    print("\nCreating MWK-FLAIR model...")
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'flair-ViT-B-16',
        pretrained=None,  # Set to checkpoint path for pretrained weights
        use_harma=True,
        harma_reduction_dim=128,
        harma_num_heads=8
    )
    model = model.to(device)
    model.eval()
    
    # Get tokenizer
    tokenizer = get_tokenizer('flair-ViT-B-16')
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Example: Create dummy inputs
    print("\nRunning inference...")
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    text_prompts = [
        "a photo of a cat",
        "a photo of a dog", 
        "a beautiful landscape",
        "a person walking"
    ]
    text_tokens = tokenizer(text_prompts).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images, text_tokens)
    
    # Extract features
    image_features = outputs['image_features']
    text_features = outputs['text_features']
    
    print(f"\nImage features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    # Check HarMA features if available
    if outputs.get('harma_image_features') is not None:
        harma_img = outputs['harma_image_features']
        harma_txt = outputs['harma_text_features']
        print(f"HarMA image features shape: {harma_img.shape}")
        print(f"HarMA text features shape: {harma_txt.shape}")
        
        # Compute similarity using HarMA features
        harma_similarity = harma_img @ harma_txt.T
        print(f"\nHarMA Similarity matrix:\n{harma_similarity}")
    
    print("\nInference completed successfully!")


def demo_harma_adapter():
    """Demonstrate standalone HarMA adapter usage"""
    print("\n" + "="*50)
    print("HarMA Adapter Demo")
    print("="*50)
    
    # Create a simple HarMA adapter
    adapter = HarmaAdapter(
        in_features=512,
        adapter_dim=64,
        dropout=0.1
    )
    
    # Count parameters
    params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter parameters: {params:,}")
    
    # Test forward pass
    x = torch.randn(4, 10, 512)  # (batch, seq_len, features)
    output = adapter(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Adapter test passed!")


if __name__ == '__main__':
    # Run HarMA adapter demo first (doesn't require full model)
    demo_harma_adapter()
    
    # Uncomment below to run full model inference
    # (requires open_clip and model weights)
    # main()
