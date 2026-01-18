"""
HarMA Modules for FLAIR Integration - 核心适配器和损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class BiShareAdapter(nn.Module):
    """双向共享适配器"""
    def __init__(self, hidden_dim, num_heads=8, reduction_factor=2, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = hidden_dim // reduction_factor
        
        self.down_proj = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.up_proj = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.multihead_attention = nn.MultiheadAttention(
            self.bottleneck_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Parameter(torch.tensor(0.6))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
        
    def _init_weights(self):
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        
        x = self.down_proj(x)
        x_bottleneck = x
        attn_output, _ = self.multihead_attention(x, x, x)
        x = F.gelu(x)
        alpha = torch.sigmoid(self.gate)
        x = alpha * attn_output + (1 - alpha) * x_bottleneck
        x = self.dropout(x)
        x = self.up_proj(x)
        
        if squeeze_output:
            x = x.squeeze(1)
        return x + residual


class MultimodalGatedAdapter(nn.Module):
    """多模态门控适配器 (MGA)"""
    def __init__(self, hidden_dim, reduction_dim=128, num_heads=8, 
                 shared_adapter=None, dropout=0.0, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reduction_dim = reduction_dim
        
        self.down_proj = nn.Linear(hidden_dim, reduction_dim)
        self.up_proj = nn.Linear(reduction_dim, hidden_dim)
        self.multihead_attention = nn.MultiheadAttention(
            reduction_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.shared_adapter = shared_adapter
        self.gate = nn.Parameter(torch.tensor(0.6))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x, context=None):
        residual = x
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        
        x = self.down_proj(x)
        x = F.gelu(x)
        x_mid = x
        
        if context is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1)
            if context.size(-1) != self.reduction_dim:
                context = self.down_proj(context)
            attn_output, _ = self.multihead_attention(x, context, context)
        else:
            attn_output, _ = self.multihead_attention(x, x, x)
        
        x = attn_output
        if self.shared_adapter is not None:
            x = self.shared_adapter(x)
        
        attn_output, _ = self.multihead_attention(x, x, x)
        alpha = torch.sigmoid(self.gate)
        x = alpha * x_mid + (1 - alpha) * attn_output
        x = self.dropout(x)
        x = self.up_proj(x)
        
        if squeeze_output:
            x = x.squeeze(1)
        return residual + x


class AdaptiveTripletLoss(nn.Module):
    """自适应三元组损失"""
    def __init__(self, margin=0.2, gamma=2.0, max_violation=False, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.max_violation = max_violation
        self.reduction = reduction
        
    def forward(self, image_features, text_features):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            image_list = [torch.zeros_like(image_features) for _ in range(world_size)]
            text_list = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(image_list, image_features)
            dist.all_gather(text_list, text_features)
            image_features_all = torch.cat(image_list, dim=0)
            text_features_all = torch.cat(text_list, dim=0)
        else:
            image_features_all = image_features
            text_features_all = text_features
        
        scores = image_features_all @ text_features_all.t()
        batch_size = image_features_all.shape[0]
        diagonal = scores.diag().view(batch_size, 1)
        d_image = diagonal.expand_as(scores)
        d_text = diagonal.t().expand_as(scores)
        
        cost_text = (self.margin + scores - d_image).clamp(min=0)
        cost_image = (self.margin + scores - d_text).clamp(min=0)
        
        mask = torch.eye(batch_size, device=scores.device, dtype=torch.bool)
        cost_text = cost_text.masked_fill(mask, 0)
        cost_image = cost_image.masked_fill(mask, 0)
        
        p_text = torch.exp(-cost_text)
        weights_text = (1 - p_text) ** self.gamma
        p_image = torch.exp(-cost_image)
        weights_image = (1 - p_image) ** self.gamma
        
        cost_text = weights_text * cost_text
        cost_image = weights_image * cost_image
        
        if self.max_violation:
            cost_text = cost_text.max(1)[0]
            cost_image = cost_image.max(0)[0]
        
        if self.reduction == 'mean':
            loss = (cost_text.sum() + cost_image.sum()) / (2.0 * batch_size)
        else:
            loss = (cost_text.sum() + cost_image.sum()) / 2.0
        return loss


def build_harma_adapters(num_layers, hidden_dim, reduction_dim=128, num_heads=8, 
                         use_shared_adapter=True, dropout=0.0):
    """构建 HarMA 适配器列表"""
    shared_adapter = None
    if use_shared_adapter:
        shared_adapter = BiShareAdapter(
            hidden_dim=reduction_dim, num_heads=num_heads,
            reduction_factor=2, dropout=dropout
        )
    
    adapters = nn.ModuleList([
        MultimodalGatedAdapter(
            hidden_dim=hidden_dim, reduction_dim=reduction_dim,
            num_heads=num_heads, shared_adapter=shared_adapter,
            dropout=dropout, layer_id=i
        )
        for i in range(num_layers)
    ])
    return adapters


class StandardTripletLoss(nn.Module):
    """标准三元组损失（无自适应权重）"""
    def __init__(self, margin=0.2, max_violation=False):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation
    
    def forward(self, image_features, text_features):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            image_list = [torch.zeros_like(image_features) for _ in range(world_size)]
            text_list = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(image_list, image_features)
            dist.all_gather(text_list, text_features)
            image_features_all = torch.cat(image_list, dim=0)
            text_features_all = torch.cat(text_list, dim=0)
        else:
            image_features_all = image_features
            text_features_all = text_features
        
        scores = image_features_all @ text_features_all.t()
        batch_size = scores.size(0)
        diagonal = scores.diag().view(batch_size, 1)
        d_image = diagonal.expand_as(scores)
        d_text = diagonal.t().expand_as(scores)
        
        cost_text = (self.margin + scores - d_image).clamp(min=0)
        cost_image = (self.margin + scores - d_text).clamp(min=0)
        
        mask = torch.eye(batch_size, device=scores.device, dtype=torch.bool)
        cost_text = cost_text.masked_fill(mask, 0)
        cost_image = cost_image.masked_fill(mask, 0)
        
        if self.max_violation:
            cost_text = cost_text.max(1)[0]
            cost_image = cost_image.max(0)[0]
        
        loss = (cost_text.sum() + cost_image.sum()) / batch_size
        return loss
