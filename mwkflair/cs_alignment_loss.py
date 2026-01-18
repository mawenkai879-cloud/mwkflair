"""
CS Alignment Loss for Image-Text Alignment
Based on Cauchy-Schwarz Divergence with Kernel Density Estimation

This module implements a distributional distance metric between image and text features
using the Cauchy-Schwarz (CS) divergence. The implementation follows the coding style
from the CSD repository (https://github.com/JiahaoZhang666/CSD).

Mathematical Formula:
    L_CS = log(1/M² ∑_{i,j} K(x_i, x_j)) + log(1/N² ∑_{i,j} K(y_i, y_j)) 
           - 2*log(1/(M*N) ∑_{i,j} K(x_i, y_j))

where:
    - K(u,v) = exp(-||u-v||²/(2σ²)) is the Gaussian kernel
    - x are image features (M samples)
    - y are text features (N samples)
    - σ is the kernel bandwidth

Author: MWK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CSAlignmentLoss(nn.Module):
    """
    Cauchy-Schwarz Divergence Loss for Image-Text Alignment.
    
    This loss computes the distributional distance between image and text feature
    distributions using Kernel Density Estimation (KDE) with Gaussian kernels.
    
    Args:
        sigma (float): Kernel bandwidth for Gaussian kernel. Default: 1.0
        trainable_sigma (bool): Whether sigma is a trainable parameter. Default: False
        epsilon (float): Small value for numerical stability. Default: 1e-8
        token_level (bool): If True, supports token-level alignment for 3D inputs. Default: False
    
    Input:
        image_features: Tensor of shape (M, D) or (B, T, D) for token-level
        text_features: Tensor of shape (N, D) or (B, T, D) for token-level
        
    Output:
        cs_loss: Scalar tensor representing the CS divergence
    
    Example:
        >>> loss_fn = CSAlignmentLoss(sigma=1.0)
        >>> img_feat = torch.randn(32, 512)  # 32 images, 512-dim features
        >>> txt_feat = torch.randn(48, 512)  # 48 texts, 512-dim features
        >>> loss = loss_fn(img_feat, txt_feat)
    """
    
    def __init__(self, sigma=1.0, trainable_sigma=False, epsilon=1e-8, token_level=False):
        super(CSAlignmentLoss, self).__init__()
        
        self.epsilon = epsilon
        self.token_level = token_level
        
        # Kernel bandwidth parameter
        if trainable_sigma:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32))
    
    def _compute_kernel(self, x, y):
        """
        Compute Gaussian kernel matrix K(x, y) = exp(-||x-y||²/(2σ²))
        
        This follows the efficient pairwise distance computation style from CSD repo.
        
        Args:
            x: Tensor of shape (M, D)
            y: Tensor of shape (N, D)
            
        Returns:
            kernel_matrix: Tensor of shape (M, N)
        """
        # Compute pairwise squared Euclidean distance
        # ||x - y||² = ||x||² + ||y||² - 2<x, y>
        x_square = torch.sum(x * x, dim=1, keepdim=True)  # (M, 1)
        y_square = torch.sum(y * y, dim=1, keepdim=True)  # (N, 1)
        
        # Pairwise distance matrix: (M, N)
        distance = x_square + y_square.t() - 2 * torch.matmul(x, y.t())
        
        # Clamp to avoid negative values due to numerical errors
        distance = torch.clamp(distance, min=0.0)
        
        # Gaussian kernel: K(x,y) = exp(-dist/(2σ²))
        kernel_matrix = torch.exp(-distance / (2 * self.sigma ** 2))
        
        return kernel_matrix
    
    def _compute_cs_divergence_2d(self, image_features, text_features):
        """
        Compute CS divergence for 2D feature tensors (batch-level).
        
        Args:
            image_features: (M, D) - M image samples
            text_features: (N, D) - N text samples
            
        Returns:
            cs_loss: Scalar tensor
        """
        M = image_features.size(0)
        N = text_features.size(0)
        
        # L2 normalize features (crucial for stability)
        image_norm = F.normalize(image_features, p=2, dim=1)
        text_norm = F.normalize(text_features, p=2, dim=1)
        
        # Compute kernel matrices
        K_xx = self._compute_kernel(image_norm, image_norm)  # (M, M)
        K_yy = self._compute_kernel(text_norm, text_norm)    # (N, N)
        K_xy = self._compute_kernel(image_norm, text_norm)   # (M, N)
        
        # Term 1: log(1/M² * ∑K(x_i, x_j))
        # Use logsumexp for numerical stability
        log_K_xx = torch.logsumexp(K_xx.flatten(), dim=0) - math.log(M * M)
        
        # Term 2: log(1/N² * ∑K(y_i, y_j))
        log_K_yy = torch.logsumexp(K_yy.flatten(), dim=0) - math.log(N * N)
        
        # Term 3: log(1/(M*N) * ∑K(x_i, y_j))
        log_K_xy = torch.logsumexp(K_xy.flatten(), dim=0) - math.log(M * N)
        
        # CS Divergence: L_CS = log(K_xx) + log(K_yy) - 2*log(K_xy)
        cs_loss = log_K_xx + log_K_yy - 2 * log_K_xy
        
        return cs_loss
    
    def _compute_cs_divergence_3d(self, image_features, text_features):
        """
        Compute CS divergence for 3D feature tensors (token-level alignment).
        
        For each sample in the batch, treat tokens as a distribution and compute
        the CS divergence between image tokens and text tokens.
        
        Args:
            image_features: (B, T_img, D) - B samples, T_img image tokens each
            text_features: (B, T_txt, D) - B samples, T_txt text tokens each
            
        Returns:
            cs_loss: Scalar tensor (averaged over batch)
        """
        B = image_features.size(0)
        
        if image_features.size(0) != text_features.size(0):
            raise ValueError(
                f"Batch size mismatch for token-level alignment: "
                f"image {image_features.size(0)} vs text {text_features.size(0)}"
            )
        
        cs_losses = []
        
        for b in range(B):
            # Extract tokens for this sample
            img_tokens = image_features[b]  # (T_img, D)
            txt_tokens = text_features[b]   # (T_txt, D)
            
            # Compute CS divergence for this sample's token distributions
            sample_loss = self._compute_cs_divergence_2d(img_tokens, txt_tokens)
            cs_losses.append(sample_loss)
        
        # Average across batch
        cs_loss = torch.stack(cs_losses).mean()
        
        return cs_loss
    
    def forward(self, image_features, text_features):
        """
        Forward pass to compute CS Alignment Loss.
        
        Args:
            image_features: Tensor of shape (M, D) or (B, T, D)
                - (M, D): M image samples with D-dimensional features
                - (B, T, D): B samples, each with T tokens of D dimensions
            text_features: Tensor of shape (N, D) or (B, T, D)
                - (N, D): N text samples with D-dimensional features
                - (B, T, D): B samples, each with T tokens of D dimensions
                
        Returns:
            cs_loss: Scalar tensor representing the CS divergence
        """
        # Check input dimensions
        if image_features.dim() not in [2, 3]:
            raise ValueError(
                f"image_features must be 2D (M, D) or 3D (B, T, D), "
                f"got shape {image_features.shape}"
            )
        
        if text_features.dim() not in [2, 3]:
            raise ValueError(
                f"text_features must be 2D (N, D) or 3D (B, T, D), "
                f"got shape {text_features.shape}"
            )
        
        # Check feature dimension consistency
        if image_features.size(-1) != text_features.size(-1):
            raise ValueError(
                f"Feature dimension mismatch: "
                f"image {image_features.size(-1)} vs text {text_features.size(-1)}"
            )
        
        # Determine if token-level or batch-level
        is_3d = (image_features.dim() == 3 or text_features.dim() == 3)
        
        if is_3d:
            if not self.token_level:
                raise ValueError(
                    "Received 3D input but token_level=False. "
                    "Set token_level=True to enable token-level alignment."
                )
            
            # Ensure both are 3D
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(1)
            if text_features.dim() == 2:
                text_features = text_features.unsqueeze(1)
            
            cs_loss = self._compute_cs_divergence_3d(image_features, text_features)
        else:
            # 2D batch-level alignment (supports unpaired data M ≠ N)
            cs_loss = self._compute_cs_divergence_2d(image_features, text_features)
        
        return cs_loss


class CSInfoNCELoss(nn.Module):
    """
    Combined CS Divergence + InfoNCE Loss for Image-Text Alignment.
    
    This combines the distributional CS divergence with the contrastive InfoNCE loss
    for a more robust alignment objective.
    
    Args:
        sigma (float): Kernel bandwidth for CS divergence. Default: 1.0
        temperature (float): Temperature for InfoNCE. Default: 0.07
        cs_weight (float): Weight for CS divergence term. Default: 1.0
        infonce_weight (float): Weight for InfoNCE term. Default: 1.0
        epsilon (float): Small value for numerical stability. Default: 1e-8
    """
    
    def __init__(self, sigma=1.0, temperature=0.07, cs_weight=1.0, 
                 infonce_weight=1.0, epsilon=1e-8):
        super(CSInfoNCELoss, self).__init__()
        
        self.cs_loss_fn = CSAlignmentLoss(sigma=sigma, epsilon=epsilon)
        self.temperature = temperature
        self.cs_weight = cs_weight
        self.infonce_weight = infonce_weight
        self.epsilon = epsilon
    
    def _compute_infonce(self, image_features, text_features):
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            image_features: (B, D)
            text_features: (B, D)
            
        Returns:
            infonce_loss: Scalar tensor
        """
        B = image_features.size(0)
        
        # L2 normalize
        image_norm = F.normalize(image_features, p=2, dim=1)
        text_norm = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_norm, text_norm.t()) / self.temperature  # (B, B)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=image_features.device)
        
        # Cross-entropy loss (image-to-text and text-to-image)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        infonce_loss = (loss_i2t + loss_t2i) / 2
        
        return infonce_loss
    
    def forward(self, image_features, text_features):
        """
        Forward pass combining CS divergence and InfoNCE.
        
        Args:
            image_features: (B, D) - Batch of image features
            text_features: (B, D) - Batch of text features
            
        Returns:
            total_loss: Combined loss
            cs_loss: CS divergence component
            infonce_loss: InfoNCE component
        """
        # CS Divergence
        cs_loss = self.cs_loss_fn(image_features, text_features)
        
        # InfoNCE (requires paired data)
        if image_features.size(0) == text_features.size(0):
            infonce_loss = self._compute_infonce(image_features, text_features)
        else:
            infonce_loss = torch.tensor(0.0, device=image_features.device)
        
        # Combined loss
        total_loss = self.cs_weight * cs_loss + self.infonce_weight * infonce_loss
        
        return total_loss, cs_loss, infonce_loss
