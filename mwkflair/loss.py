import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math
import logging

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

# Import HarMA modules for adaptive triplet loss
from .harma_modules import AdaptiveTripletLoss, StandardTripletLoss
# Import CS Alignment Loss for distributional alignment
from .cs_alignment_loss import CSAlignmentLoss, CSInfoNCELoss


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features




def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)



def get_multi_positive_mps(target, k):
    """
    :param target: tensor of shape (b, b*k), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b*k), for each row i, the col [i*k, (i+1)*k] should be ones
    """
    for i in range(target.shape[0]):
        target[i, i * k:(i + 1) * k] = 1
    return target



def get_multi_positive_tcs(target, k):
    """
    :param target: tensor of shape (b, b+k-1), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b+k-1), for each row i, the col [i, i+k) should be ones
    """
    for i in range(target.shape[0]):
        target[i, i: i + k] = 1
    return target




def get_mps_logits(image_features, text_features, logit_scale, logit_bias=None):
    logits = logit_scale * image_features @ text_features.T  # if multi-cap: (B, B*K)
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_mps_ground_truth(device, dtype, target_shape, negative_only=False,
                                        num_captions=4):
    dim0, dim1 = target_shape  # (B, B*K)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B*K)
    if not negative_only:
        labels = get_multi_positive_mps(target=labels, k=num_captions)
    return labels

def get_intra_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, K, D),
    text_features: (B, K, D).
    Target: (B, K, K)
    """
    logits = logit_scale * torch.einsum('bkd,bjd->bkj', image_features, text_features)
    # logits = logit_scale * image_features @ text_features.T  
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_tcs_ground_truth(device, dtype, target_shape, negative_only=False, num_captions=4):
    dim0, dim1 = target_shape  # (B, B+K-1)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B+K-1)
    if not negative_only:
        labels = get_multi_positive_tcs(target=labels, k=num_captions)
    return labels

def get_tcs_logits(features_0, features_1, logit_scale, logit_bias=None):
    logits = logit_scale * torch.einsum('bij,bij->bi', features_0, features_1)
    if logit_bias is not None:
        logits += logit_bias
    return logits


class FlairLoss(nn.Module):
    """
    Implementation of FLAIR loss in: https://arxiv.org/pdf/2412.03561
    When setting added_mps_loss=False, this class is simply text-conditioned sigmoid loss;
    When added_mps_loss=True, this class is 'text-conditioned sigmod loss + multi-positive sigmoid loss'
    
    HarMA Integration:
    When use_harma=True, adds AdaptiveTripletLoss on top of FLAIR loss for better feature alignment.
    The HarMA loss acts as a regularization term to prevent feature collapse and improve retrieval.
    """

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            num_cap_per_img=8,
            added_mps_loss=False,
            use_harma=False,
            harma_loss_weight=0.5,
            harma_margin=0.2,
            harma_gamma=2.0,
            harma_max_violation=False,
            use_cs_alignment=False,
            cs_loss_weight=0.3,
            cs_sigma=1.0,
            cs_use_infonce=False,
            cs_temperature=0.07,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}
        self.num_cap_per_img = num_cap_per_img
        self.added_mps_loss = added_mps_loss
        
        # === HarMA Integration ===
        self.use_harma = use_harma
        self.harma_loss_weight = harma_loss_weight
        
        if self.use_harma:
            # Initialize AdaptiveTripletLoss for HarMA features
            self.harma_loss_fn = AdaptiveTripletLoss(
                margin=harma_margin,
                gamma=harma_gamma,
                max_violation=harma_max_violation,
                reduction='mean'
            )
            logging.info(f"HarMA loss initialized: weight={harma_loss_weight}, "
                        f"margin={harma_margin}, gamma={harma_gamma}")
        else:
            self.harma_loss_fn = None
        
        # === CS Alignment Integration ===
        self.use_cs_alignment = use_cs_alignment
        self.cs_loss_weight = cs_loss_weight
        
        if self.use_cs_alignment:
            if cs_use_infonce:
                # Use combined CS + InfoNCE loss
                self.cs_loss_fn = CSInfoNCELoss(
                    sigma=cs_sigma,
                    temperature=cs_temperature,
                    cs_weight=1.0,
                    infonce_weight=1.0
                )
                logging.info(f"CS+InfoNCE loss initialized: weight={cs_loss_weight}, "
                            f"sigma={cs_sigma}, temperature={cs_temperature}")
            else:
                # Use pure CS divergence loss
                self.cs_loss_fn = CSAlignmentLoss(
                    sigma=cs_sigma,
                    trainable_sigma=False
                )
                logging.info(f"CS Alignment loss initialized: weight={cs_loss_weight}, "
                            f"sigma={cs_sigma}")
        else:
            self.cs_loss_fn = None


    def _loss_with_attn_pool(self, image_features, image_tokens, text_features, logit_scale,
                             logit_bias=None, negative_only=False, visual_proj=None, g_text_features=None):

        local_image_features = visual_proj(text_features, image_tokens, image_tokens)  # (B, B+K-1, D)

        local_image_features = F.normalize(local_image_features, dim=-1)
        global_text_features = F.normalize(text_features, dim=-1)

        i2t_logits = get_tcs_logits(local_image_features, global_text_features, logit_scale, logit_bias)

        i2t_labels = get_tcs_ground_truth(device=text_features.device,
                                        dtype=text_features.dtype,
                                        target_shape=i2t_logits.size(),
                                        negative_only=negative_only,
                                        num_captions=self.num_cap_per_img)

        tcs_loss = -F.logsigmoid(i2t_labels * i2t_logits).sum() / text_features.shape[1] # text-conditioned sigmoid loss


        if self.added_mps_loss:
            g_image_features = F.normalize(image_features, dim=-1)  #(B, D)
            g_text_features = F.normalize(g_text_features, dim=-1)  #(B*K, D)
            mps_logits = get_mps_logits(image_features=g_image_features, text_features=g_text_features,
                                                  logit_scale=logit_scale, logit_bias=logit_bias)
            g2g_labels = get_mps_ground_truth(device=g_text_features.device,
                                              dtype=g_text_features.dtype,
                                              target_shape=mps_logits.size(),
                                              negative_only=negative_only,
                                              num_captions=self.num_cap_per_img)
            mps_loss = -F.logsigmoid(g2g_labels * mps_logits).sum() / g_text_features.shape[0] # multi-positive sigmoid loss

            loss = (tcs_loss + mps_loss) / 2
        else:
            loss = tcs_loss


        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, image_tokens=None,
                visual_proj=None, output_dict=False, harma_image_features=None, harma_text_features=None):
        '''
        expected shape: 
            text_features: (B*K, D) - FLAIR text features
            image_features: (B, D) - FLAIR global image features
            image_tokens: (B, L, D) - FLAIR local image tokens
            harma_image_features: (B, D) - HarMA-enhanced image features (optional)
            harma_text_features: (B*K, D) - HarMA-enhanced text features (optional)
        '''
        if self.added_mps_loss:
            g_text_features = text_features  # (B*K, D)
        else:
            g_text_features = None
        

        # We don't change the shape of image tokens anywhere before the loss function.
        batch_size = image_tokens.shape[0]
        num_captions = self.num_cap_per_img
        caption_indices = torch.arange(batch_size * num_captions).view(batch_size, num_captions).to(
            text_features.device)

        text_features_downsampled = downsample_text_features(text_features=text_features, batch_size=batch_size,
                                                 caption_indices=caption_indices,
                                                 num_captions=num_captions)

        # Compute FLAIR loss (text-conditioned sigmoid loss)
        loss_flair = self._loss_with_attn_pool(image_features=image_features,
                                         image_tokens=image_tokens,
                                         text_features=text_features_downsampled,
                                         visual_proj=visual_proj,
                                         logit_scale=logit_scale,
                                         logit_bias=logit_bias,
                                         g_text_features=g_text_features)
        
        loss = loss_flair

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features_to_left = g_text_features

                num_bidir, remainder = divmod(self.world_size - 1, 2)

                g_text_features_recv = None  # predefine it to be None

                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            g_text_features_to_left,
                            g_text_features_to_right,
                        )
                        for j in range(len(text_features_recv)):
                            loss += self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=text_features_recv[j],
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=g_text_features_recv[j]
                            )
                    else:
                        for f in text_features_recv:
                            loss += self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=f,
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=None)
                    text_features_to_left, text_features_to_right = text_features_recv
                    if self.added_mps_loss:
                        g_text_features_to_left, g_text_features_to_right = g_text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=g_text_features_recv
                        )
                    else:
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=None)
            else:
                text_features_to_right = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features

                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    if self.added_mps_loss:
                        g_text_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                    else:
                        g_text_features_from_left = None

                    loss += self._loss_with_attn_pool(
                        image_features=image_features,
                        image_tokens=image_tokens,
                        text_features=text_features_from_left,
                        visual_proj=visual_proj,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True,
                        g_text_features=g_text_features_from_left)

                    text_features_to_right = text_features_from_left

        # === HarMA Loss Computation ===
        # Add adaptive triplet loss on HarMA-enhanced features as a regularization term
        loss_harma = None
        loss_cs = None
        
        if self.use_harma and harma_image_features is not None and harma_text_features is not None:
            # Ensure features are normalized (should already be normalized in model.forward)
            if not torch.allclose(harma_image_features.norm(dim=-1), torch.ones(1, device=harma_image_features.device), atol=1e-3):
                harma_image_features = F.normalize(harma_image_features, dim=-1)
            if not torch.allclose(harma_text_features.norm(dim=-1), torch.ones(1, device=harma_text_features.device), atol=1e-3):
                harma_text_features = F.normalize(harma_text_features, dim=-1)
            
            # For HarMA loss, we need to match image (B, D) with averaged text (B, D)
            # Average multiple captions per image: (B*K, D) -> (B, D)
            harma_text_features_avg = harma_text_features.view(batch_size, num_captions, -1).mean(dim=1)  # (B, D)
            
            # Compute adaptive triplet loss
            loss_harma = self.harma_loss_fn(harma_image_features, harma_text_features_avg)
            
            # Add weighted HarMA loss
            loss = loss + self.harma_loss_weight * loss_harma
        
        # === CS Alignment Loss Computation ===
        # Add CS divergence for distributional alignment between image and text features
        if self.use_cs_alignment and harma_image_features is not None and harma_text_features is not None:
            # Ensure features are normalized
            if not torch.allclose(harma_image_features.norm(dim=-1), torch.ones(1, device=harma_image_features.device), atol=1e-3):
                harma_image_features = F.normalize(harma_image_features, dim=-1)
            if not torch.allclose(harma_text_features.norm(dim=-1), torch.ones(1, device=harma_text_features.device), atol=1e-3):
                harma_text_features = F.normalize(harma_text_features, dim=-1)
            
            # Average text features: (B*K, D) -> (B, D)
            harma_text_features_avg = harma_text_features.view(batch_size, num_captions, -1).mean(dim=1)
            
            # Compute CS divergence loss
            if isinstance(self.cs_loss_fn, CSInfoNCELoss):
                # Combined CS + InfoNCE
                loss_cs_total, loss_cs_pure, loss_infonce = self.cs_loss_fn(
                    harma_image_features, harma_text_features_avg
                )
                loss_cs = loss_cs_total
            else:
                # Pure CS divergence
                loss_cs = self.cs_loss_fn(harma_image_features, harma_text_features_avg)
            
            # Add weighted CS loss
            loss = loss + self.cs_loss_weight * loss_cs
        
        # Return loss with optional detailed breakdown
        if output_dict:
            result = {
                "contrastive_loss": loss,
                "flair_loss": loss_flair,
            }
            if loss_harma is not None:
                result["harma_loss"] = loss_harma
                result["harma_loss_weight"] = self.harma_loss_weight
            if loss_cs is not None:
                result["cs_loss"] = loss_cs
                result["cs_loss_weight"] = self.cs_loss_weight
            return result
        else:
            return loss




def downsample_text_features(text_features, batch_size, caption_indices, num_captions):
    device = text_features.device
    own_caption_indices = caption_indices  # Shape: (B, K)

    mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)

    other_image_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(batch_size, batch_size)
    other_image_indices = other_image_indices[mask].view(batch_size, batch_size - 1)
    random_offsets = torch.randint(0, num_captions, (batch_size, batch_size - 1), device=device)  # (B, B-1)
    other_caption_indices = caption_indices[other_image_indices, random_offsets]  # sampled indices (B, B-1)

    combined_indices = torch.cat([own_caption_indices, other_caption_indices], dim=1)
    combined_indices, _ = combined_indices.sort(dim=1)
    flat_combined_indices = combined_indices.view(-1)  # flatten to take the text_features out

    downsampled_text_features = text_features[flat_combined_indices]

    embed_dim = text_features.shape[-1]  # Reshape to (B, K + B - 1, D)
    downsampled_text_features = downsampled_text_features.view(batch_size, num_captions + batch_size - 1, embed_dim)
    return downsampled_text_features


# ================================================================================
# Parameter-Efficient Fine-Tuning (PEFT) Utilities for HarMA + FLAIR
# ================================================================================

def setup_peft_for_harma_flair(model, freeze_backbone=True, freeze_text_encoder=True, 
                                 freeze_visual_encoder=True, verbose=True):
    """
    Setup Parameter-Efficient Fine-Tuning (PEFT) for HarMA + FLAIR model.
    
    This function implements a freezing strategy suitable for small datasets:
    - Freezes the CLIP backbone (Visual Transformer + Text Transformer)
    - Keeps trainable: HarMA adapters, FLAIR attention pooling, projection layers
    
    Args:
        model: The FLAIR model instance
        freeze_backbone: Whether to freeze the backbone encoders. Default: True
        freeze_text_encoder: Whether to freeze text encoder. Default: True
        freeze_visual_encoder: Whether to freeze visual encoder. Default: True
        verbose: Whether to print freezing information. Default: True
    
    Returns:
        tuple: (trainable_params, total_params, trainable_percentage)
    """
    
    trainable_param_names = []
    frozen_param_names = []
    
    for name, param in model.named_parameters():
        # Default: freeze everything
        param.requires_grad = False
        
        # === MUST TRAIN: HarMA Adapters ===
        if 'harma' in name.lower():
            param.requires_grad = True
            trainable_param_names.append(name)
            continue
        
        # === MUST TRAIN: FLAIR Attention Pooling (visual_proj) ===
        if 'visual_proj' in name:
            param.requires_grad = True
            trainable_param_names.append(name)
            continue
        
        # === MUST TRAIN: Post-processing layers (image_post, text_post) ===
        if 'image_post' in name or 'text_post' in name:
            param.requires_grad = True
            trainable_param_names.append(name)
            continue
        
        # === MUST TRAIN: Learnable parameters in loss (logit_scale, logit_bias) ===
        if 'logit_scale' in name or 'logit_bias' in name:
            param.requires_grad = True
            trainable_param_names.append(name)
            continue
        
        # === OPTIONAL: Visual Encoder ===
        if not freeze_visual_encoder and 'visual' in name:
            # Only unfreeze if explicitly requested
            param.requires_grad = True
            trainable_param_names.append(name)
            continue
        
        # === OPTIONAL: Text Encoder ===
        if not freeze_text_encoder and ('transformer' in name or 'token_embedding' in name 
                                         or 'positional_embedding' in name or 'ln_final' in name):
            # Only unfreeze if explicitly requested
            param.requires_grad = True
            trainable_param_names.append(name)
            continue
        
        # Everything else is frozen
        frozen_param_names.append(name)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    
    if verbose:
        logging.info("="*80)
        logging.info("Parameter-Efficient Fine-Tuning (PEFT) Configuration")
        logging.info("="*80)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}")
        logging.info(f"Trainable percentage: {trainable_percentage:.2f}%")
        logging.info(f"Memory efficiency: {100.0 - trainable_percentage:.2f}% reduction")
        logging.info("-"*80)
        logging.info(f"Number of trainable modules: {len(trainable_param_names)}")
        logging.info(f"Number of frozen modules: {len(frozen_param_names)}")
        logging.info("-"*80)
        
        if len(trainable_param_names) <= 50:  # Only print if not too many
            logging.info("Trainable parameters:")
            for name in trainable_param_names:
                param_count = sum(p.numel() for n, p in model.named_parameters() 
                                 if n == name and p.requires_grad)
                logging.info(f"  ✓ {name}: {param_count:,} params")
        else:
            logging.info(f"Trainable parameters (showing first 20 of {len(trainable_param_names)}):")
            for name in trainable_param_names[:20]:
                param_count = sum(p.numel() for n, p in model.named_parameters() 
                                 if n == name and p.requires_grad)
                logging.info(f"  ✓ {name}: {param_count:,} params")
            logging.info(f"  ... and {len(trainable_param_names) - 20} more")
        
        logging.info("="*80)
    
    return trainable_params, total_params, trainable_percentage


def get_optimizer_param_groups(model, learning_rate=1e-4, adapter_lr_mult=1.0, 
                                 weight_decay=0.01, adapter_weight_decay=0.01):
    """
    Create parameter groups for optimizer with different learning rates.
    
    This allows fine-grained control over learning rates for different components:
    - HarMA adapters: can have a different learning rate multiplier
    - Other trainable parameters: base learning rate
    
    Args:
        model: The FLAIR model instance
        learning_rate: Base learning rate for trainable parameters
        adapter_lr_mult: Learning rate multiplier for HarMA adapters. Default: 1.0
        weight_decay: Weight decay for regular parameters. Default: 0.01
        adapter_weight_decay: Weight decay for adapter parameters. Default: 0.01
    
    Returns:
        list: List of parameter groups for optimizer
    """
    
    # Separate parameters into different groups
    adapter_params = []
    other_trainable_params = []
    no_decay_params = []  # For biases and layer norms
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and layer norms
        if 'bias' in name or 'ln' in name or 'norm' in name:
            no_decay_params.append(param)
        # HarMA adapters get special treatment
        elif 'harma' in name.lower():
            adapter_params.append(param)
        # Everything else
        else:
            other_trainable_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    if adapter_params:
        param_groups.append({
            'params': adapter_params,
            'lr': learning_rate * adapter_lr_mult,
            'weight_decay': adapter_weight_decay,
            'name': 'harma_adapters'
        })
        logging.info(f"Adapter parameter group: {sum(p.numel() for p in adapter_params):,} params, "
                    f"lr={learning_rate * adapter_lr_mult:.2e}")
    
    if other_trainable_params:
        param_groups.append({
            'params': other_trainable_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'other_trainable'
        })
        logging.info(f"Other trainable group: {sum(p.numel() for p in other_trainable_params):,} params, "
                    f"lr={learning_rate:.2e}")
    
    if no_decay_params:
        param_groups.append({
            'params': no_decay_params,
            'lr': learning_rate,
            'weight_decay': 0.0,
            'name': 'no_decay'
        })
        logging.info(f"No decay group: {sum(p.numel() for p in no_decay_params):,} params, "
                    f"lr={learning_rate:.2e}, weight_decay=0.0")
    
    return param_groups


def verify_trainable_parameters(model, expected_frozen_modules=None):
    """
    Verify that the parameter freezing is correctly applied.
    
    This is a sanity check to ensure:
    - HarMA adapters are trainable
    - Attention pooling is trainable
    - Backbone is frozen (if expected)
    
    Args:
        model: The FLAIR model instance
        expected_frozen_modules: List of module name patterns that should be frozen
    
    Returns:
        bool: True if verification passes, False otherwise
    """
    if expected_frozen_modules is None:
        expected_frozen_modules = ['visual.transformer', 'transformer.resblocks']
    
    issues = []
    
    # Check that adapters are trainable
    adapter_params = [n for n, p in model.named_parameters() 
                     if 'harma' in n.lower() and p.requires_grad]
    if hasattr(model, 'use_harma') and model.use_harma and len(adapter_params) == 0:
        issues.append("❌ HarMA adapters are not trainable!")
    
    # Check that attention pooling is trainable
    pooling_params = [n for n, p in model.named_parameters() 
                      if 'visual_proj' in n and p.requires_grad]
    if len(pooling_params) == 0:
        issues.append("❌ Attention pooling (visual_proj) is not trainable!")
    
    # Check that expected modules are frozen
    for module_pattern in expected_frozen_modules:
        unfrozen = [n for n, p in model.named_parameters() 
                   if module_pattern in n and p.requires_grad]
        if len(unfrozen) > 0:
            issues.append(f"⚠️  Some parameters in '{module_pattern}' are not frozen: {len(unfrozen)} params")
    
    # Print results
    if issues:
        logging.warning("Parameter freezing verification found issues:")
        for issue in issues:
            logging.warning(f"  {issue}")
        return False
    else:
        logging.info("✓ Parameter freezing verification passed!")
        return True
