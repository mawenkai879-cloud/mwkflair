"""
使用Flair预训练模型 + Harma适配器进行图像-文本检索训练
参数高效：只训练适配器，冻结预训练模型
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
import open_clip

from corn_dataset import get_corn_dataloaders
from train_config import get_config
from harma_adapter import HarmaAdapter, freeze_model_except_adapters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlairWithHarmaAdapter(nn.Module):
    """
    Flair预训练模型 + Harma适配器
    冻结预训练权重，只训练适配器
    """
    def __init__(self, pretrained_path, adapter_dim=64):
        super(FlairWithHarmaAdapter, self).__init__()
        
        print(f"加载Flair预训练模型: {pretrained_path}")
        
        # 创建OpenCLIP模型
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained=None
        )
        
        # 加载预训练权重
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state_dict[new_k] = v
        
        missing, unexpected = self.clip_model.load_state_dict(new_state_dict, strict=False)
        print(f"预训练权重加载: {len(new_state_dict) - len(missing)}/{len(new_state_dict)} 个参数")
        
        self.embed_dim = self.clip_model.visual.output_dim
        
        # 添加Harma适配器到图像和文本编码器
        self.vision_adapter = HarmaAdapter(
            in_features=self.embed_dim,
            adapter_dim=adapter_dim,
            dropout=0.1
        )
        
        self.text_adapter = HarmaAdapter(
            in_features=self.embed_dim,
            adapter_dim=adapter_dim,
            dropout=0.1
        )
        
        # 冻结预训练模型，只训练适配器
        self._freeze_pretrained_model()
        
    def _freeze_pretrained_model(self):
        """冻结CLIP预训练模型的所有参数"""
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 适配器参数保持可训练
        for param in self.vision_adapter.parameters():
            param.requires_grad = True
        for param in self.text_adapter.parameters():
            param.requires_grad = True
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {100 * trainable_params / total_params:.2f}%")
        
    def encode_image(self, images):
        """编码图像并通过Harma适配器"""
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        
        # 通过Harma适配器 (需要添加seq维度)
        image_features = image_features.unsqueeze(1)  # (batch, 1, dim)
        image_features = self.vision_adapter(image_features)
        image_features = image_features.squeeze(1)  # (batch, dim)
        
        return image_features
    
    def encode_text(self, text_tokens):
        """编码文本并通过Harma适配器"""
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
        
        # 通过Harma适配器
        text_features = text_features.unsqueeze(1)  # (batch, 1, dim)
        text_features = self.text_adapter(text_features)
        text_features = text_features.squeeze(1)  # (batch, dim)
        
        return text_features
    
    def forward(self, images, text_tokens):
        """前向传播，返回归一化的特征"""
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        
        # L2归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features


def triplet_ranking_loss(image_features, text_features, margin=0.2):
    """Triplet Ranking Loss"""
    batch_size = image_features.size(0)
    scores = torch.matmul(image_features, text_features.t())
    
    diagonal = scores.diag().view(batch_size, 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_i2t = (margin + scores - d1).clamp(min=0)
    cost_t2i = (margin + scores - d2).clamp(min=0)
    
    mask = torch.eye(batch_size, device=scores.device) > 0.5
    cost_i2t = cost_i2t.masked_fill_(mask, 0)
    cost_t2i = cost_t2i.masked_fill_(mask, 0)
    
    loss_i2t = cost_i2t.sum(1).mean()
    loss_t2i = cost_t2i.sum(0).mean()
    
    return (loss_i2t + loss_t2i) / 2


def compute_retrieval_metrics(image_features, text_features):
    """计算检索指标"""
    batch_size = image_features.size(0)
    scores = torch.matmul(image_features, text_features.t()).cpu().numpy()
    
    i2t_ranks = []
    for i in range(batch_size):
        inds = np.argsort(scores[i])[::-1]
        rank = np.where(inds == i)[0][0] + 1
        i2t_ranks.append(rank)
    
    t2i_ranks = []
    for i in range(batch_size):
        inds = np.argsort(scores[:, i])[::-1]
        rank = np.where(inds == i)[0][0] + 1
        t2i_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    return {
        'i2t_r1': 100.0 * np.mean(i2t_ranks <= 1),
        'i2t_r5': 100.0 * np.mean(i2t_ranks <= 5),
        'i2t_r10': 100.0 * np.mean(i2t_ranks <= 10),
        't2i_r1': 100.0 * np.mean(t2i_ranks <= 1),
        't2i_r5': 100.0 * np.mean(t2i_ranks <= 5),
        't2i_r10': 100.0 * np.mean(t2i_ranks <= 10),
        'rsum': 100.0 * (np.mean(i2t_ranks <= 1) + np.mean(i2t_ranks <= 5) + np.mean(i2t_ranks <= 10) +
                         np.mean(t2i_ranks <= 1) + np.mean(t2i_ranks <= 5) + np.mean(t2i_ranks <= 10))
    }


def train_epoch(model, train_loader, optimizer, device, config, epoch, tokenizer):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
    
    for batch_idx, (images, captions, labels, _) in enumerate(pbar):
        images = images.to(device)
        text_tokens = tokenizer(captions).to(device)
        
        optimizer.zero_grad()
        image_features, text_features = model(images, text_tokens)
        loss = triplet_ranking_loss(image_features, text_features, margin=config.margin)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device, tokenizer):
    model.eval()
    
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for images, captions, labels, _ in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            text_tokens = tokenizer(captions).to(device)
            
            image_features, text_features = model(images, text_tokens)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    all_image_features = torch.cat(all_image_features, dim=0).to(device)
    all_text_features = torch.cat(all_text_features, dim=0).to(device)
    
    metrics = compute_retrieval_metrics(all_image_features, all_text_features)
    
    return metrics


def main():
    config = get_config()
    set_seed(config.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print('\n加载数据集...')
    train_loader, val_loader, test_loader, num_classes = get_corn_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    print(f'训练集: {len(train_loader.dataset)} 样本')
    print(f'验证集: {len(val_loader.dataset)} 样本')
    print(f'测试集: {len(test_loader.dataset)} 样本')
    
    # 创建模型（Flair + Harma适配器）
    print('\n创建模型（Flair预训练 + Harma适配器）...')
    pretrained_path = '/home/vision/mwk/Flair/flair-merged30m.pt'
    model = FlairWithHarmaAdapter(pretrained_path, adapter_dim=config.adapter_dim)
    model = model.to(device)
    
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    
    # 只优化适配器参数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 添加warmup的学习率调度
    def lr_lambda(epoch):
        if epoch < 5:  # warmup 5 epochs
            return (epoch + 1) / 5
        else:
            # Cosine annealing
            progress = (epoch - 5) / (config.num_epochs - 5)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir + '_harma')
    
    print('\n开始训练（Flair + Harma适配器）...')
    print('=' * 80)
    
    best_rsum = 0
    patience = 0
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, tokenizer)
        val_metrics = validate(model, val_loader, device, tokenizer)
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Retrieval/i2t_r1', val_metrics['i2t_r1'], epoch)
        writer.add_scalar('Retrieval/t2i_r1', val_metrics['t2i_r1'], epoch)
        writer.add_scalar('Retrieval/rsum', val_metrics['rsum'], epoch)
        
        print(f'\nEpoch {epoch+1}/{config.num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}')
        print(f'  图像→文本: R@1={val_metrics["i2t_r1"]:.2f}%, R@5={val_metrics["i2t_r5"]:.2f}%, R@10={val_metrics["i2t_r10"]:.2f}%')
        print(f'  文本→图像: R@1={val_metrics["t2i_r1"]:.2f}%, R@5={val_metrics["t2i_r5"]:.2f}%, R@10={val_metrics["t2i_r10"]:.2f}%')
        print(f'  R-Sum: {val_metrics["rsum"]:.2f}')
        
        if val_metrics['rsum'] > best_rsum:
            best_rsum = val_metrics['rsum']
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rsum': val_metrics['rsum'],
                'metrics': val_metrics,
                'config': config
            }, os.path.join(config.save_dir, 'best_flair_harma_model.pth'))
            print(f'  ✓ 保存最佳模型 (R-Sum: {val_metrics["rsum"]:.2f})')
        else:
            patience += 1
        
        if patience >= config.early_stopping_patience:
            print(f'\n早停触发！')
            break
        
        print('=' * 80)
    
    # 测试
    print('\n在测试集上评估...')
    checkpoint = torch.load(os.path.join(config.save_dir, 'best_flair_harma_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = validate(model, test_loader, device, tokenizer)
    
    print(f'\n最终结果:')
    print(f'  最佳验证 R-Sum: {best_rsum:.2f}')
    print(f'  测试集:')
    print(f'    图像→文本 R@1: {test_metrics["i2t_r1"]:.2f}%')
    print(f'    文本→图像 R@1: {test_metrics["t2i_r1"]:.2f}%')
    print(f'    R-Sum: {test_metrics["rsum"]:.2f}')
    
    writer.close()
    print('\n训练完成！')


if __name__ == '__main__':
    main()
