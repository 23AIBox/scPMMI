import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 


class CLIPLoss(nn.Module):
    def __init__(self,  temperature: float = 0.07, alpha: float = 1.0, requires_grad=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature), requires_grad)
        self.alpha = alpha
    
    def forward(self, mod1_embs, mod2_embs):
 
        mod1_embs = F.normalize(mod1_embs, dim=-1)
        mod2_embs = F.normalize(mod2_embs, dim=-1)

        # Cosine similarity scaled by temperature
        logits_per_mod1 = self.logit_scale.exp() * (mod1_embs @ mod2_embs.T)
        logits_per_mod2 = logits_per_mod1.T

        batch_size = mod1_embs.size(0)
        labels = torch.arange(batch_size, device=mod1_embs.device)

        # Cross-entropy loss for both directions
        loss_i2t = F.cross_entropy(logits_per_mod1, labels)
        loss_t2i = F.cross_entropy(logits_per_mod2, labels)

        cos_similarity = F.cosine_similarity(mod1_embs, mod2_embs)
        cos_similarity_loss = (1 - torch.mean(cos_similarity)) * self.alpha

        return (loss_i2t + loss_t2i) / 2 + cos_similarity_loss
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, alpha=1.0, requires_grad=True):
        """
        Implements the contrastive loss function with learnable temperature parameter.

        Args:
            initial_temperature: Scalar, the initial value for the temperature \u03c3.
        """
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature), requires_grad)
        # self.temperature = temperature
        self.alpha = alpha
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, y):
        """
        Computes the contrastive loss.

        Args:
            x: Tensor of shape (N, D), where N is the batch size and D is the feature dimension (image embeddings).
            y: Tensor of shape (N, D), where N is the batch size and D is the feature dimension (text embeddings).

        Returns:
            loss: Scalar, the computed contrastive loss.
        """
        # x = F.normalize(x, dim=-1)
        # y = F.normalize(y, dim=-1)
        # Compute similarity matrices
        sim_matrix = self.logit_scale * torch.matmul(x, y.T)  # Shape: (N, N)

        # Compute image-to-text loss
        image_to_text_loss = -torch.mean(
            torch.log_softmax(sim_matrix, dim=1).diag()
        )

        # Compute text-to-image loss
        text_to_image_loss = -torch.mean(
            torch.log_softmax(sim_matrix.T, dim=1).diag()
        )

        # Total contrastive loss
        loss = (image_to_text_loss + text_to_image_loss) / 2
        cosine_similarity_loss = 1 - torch.mean(self.cos(x, y))
        loss += cosine_similarity_loss * self.alpha
        return loss

def contrastive_loss(image_embeddings, text_embeddings, logit_scale=None, alpha=1.0, requires_grad=True):
    """
    Computes the contrastive loss between image and text embeddings with an optional learnable temperature.

    Args:
        image_embeddings: Tensor of shape (batch_size, feature_dim)
        text_embeddings: Tensor of shape (batch_size, feature_dim)
        logit_scale: Optional nn.Parameter for temperature scaling. If None, a default learnable parameter is created.
        alpha: Weight for the cosine similarity term
        requires_grad: Whether logit_scale should require gradient

    Returns:
        loss: Scalar tensor representing the contrastive loss
        logit_scale: The temperature scaling parameter (nn.Parameter)
    """
    batch_size = image_embeddings.size(0)

    # Initialize learnable logit scale if not provided
    if logit_scale is None:
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07), requires_grad=requires_grad)

    # Compute similarity matrix (image-to-text)
    similarity_matrix = logit_scale * torch.matmul(image_embeddings, text_embeddings.T)  # Shape: (batch_size, batch_size)

    # Image-to-text contrastive loss
    image_to_text_loss = -torch.mean(torch.log_softmax(similarity_matrix, dim=1).diag())

    # Text-to-image contrastive loss
    text_to_image_loss = -torch.mean(torch.log_softmax(similarity_matrix.T, dim=1).diag())

    # Average contrastive loss
    contrastive_loss = (image_to_text_loss + text_to_image_loss) / 2

    # Cosine similarity regularization
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_loss = 1 - torch.mean(cos_sim(image_embeddings, text_embeddings))

    # Total loss
    total_loss = contrastive_loss + alpha * cosine_loss

    return total_loss, logit_scale


class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, 
        log_sigma1: float =1.0,
        log_sigma2: float = 1.0,
        log_sigma3: float = 1.0,
        temperature: float = 0.07, 
        alpha: float = 1.0, 
        requires_grad=True
    ) -> None:
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.tensor(log_sigma1), requires_grad)
        self.log_sigma2 = nn.Parameter(torch.tensor(log_sigma2), requires_grad)
        self.log_sigma3 = nn.Parameter(torch.tensor(log_sigma3), requires_grad)        

        self.clip_loss = ContrastiveLoss(temperature, alpha, requires_grad=False)
        self.mse_mod1 = nn.MSELoss()
        self.mse_mod2 = nn.BCEWithLogitsLoss()
    
    def forward(self, emb_mod1, emb_mod2, pred_mod1, pred_mod2, target_mod1, target_mod2):
        target_mod1 = target_mod1.to(pred_mod1)
        target_mod2 = target_mod2.to(pred_mod2)
        self.log_sigma1 = self.log_sigma1.to(emb_mod1)
        self.log_sigma2 = self.log_sigma2.to(emb_mod1)
        self.log_sigma3 = self.log_sigma3.to(emb_mod1)

        loss_clip = self.clip_loss(emb_mod1, emb_mod2)
        loss_mse_mod1 = self.mse_mod1(pred_mod1, target_mod1)
        loss_mse_mod2 = self.mse_mod2(pred_mod2, target_mod2)

        loss = (
            torch.exp(-self.log_sigma1) * loss_clip + self.log_sigma1 +
            torch.exp(-self.log_sigma2) * loss_mse_mod1 + self.log_sigma2 +
            torch.exp(-self.log_sigma3) * loss_mse_mod2 + self.log_sigma3
        )
        return loss, loss_clip, loss_mse_mod1, loss_mse_mod2
    
class WeightedMultiTaskRNAADTLoss(nn.Module):
    def __init__(self, 
        log_sigma1: float =1.0,
        log_sigma2: float = 1.0,
        log_sigma3: float = 1.0,
        temperature: float = 0.07, 
        alpha: float = 1.0, 
        requires_grad=True
    ) -> None:
        super().__init__()
        self.log_sigma1 = nn.Parameter(torch.tensor(log_sigma1), requires_grad)
        self.log_sigma2 = nn.Parameter(torch.tensor(log_sigma2), requires_grad)
        self.log_sigma3 = nn.Parameter(torch.tensor(log_sigma3), requires_grad)        

        self.clip_loss = ContrastiveLoss(temperature, alpha, requires_grad=False)
        self.mse_mod1 = nn.MSELoss()
        self.mse_mod2 = nn.MSELoss()
    
    def forward(self, emb_mod1, emb_mod2, pred_mod1, pred_mod2, target_mod1, target_mod2):
        target_mod1 = target_mod1.to(pred_mod1)
        target_mod2 = target_mod2.to(pred_mod2)
        self.log_sigma1 = self.log_sigma1.to(emb_mod1)
        self.log_sigma2 = self.log_sigma2.to(emb_mod1)
        self.log_sigma3 = self.log_sigma3.to(emb_mod1)

        loss_clip = self.clip_loss(emb_mod1, emb_mod2)
        loss_mse_mod1 = self.mse_mod1(pred_mod1, target_mod1)
        loss_mse_mod2 = self.mse_mod2(pred_mod2, target_mod2)

        loss = (
            torch.exp(-self.log_sigma1) * loss_clip + self.log_sigma1 +
            torch.exp(-self.log_sigma2) * loss_mse_mod1 + self.log_sigma2 +
            torch.exp(-self.log_sigma3) * loss_mse_mod2 + self.log_sigma3
        )
        return loss, loss_clip, loss_mse_mod1, loss_mse_mod2
    


def ecs_loss(cell_emb, ecs_threshold=0.3):
    """
    Compute Elastic Cell Similarity (ECS) loss.
    
    Args:
        cell_emb: Tensor of shape (batch_size, dim), cell embeddings.
        beta: float, threshold for similarity.
    
    Returns:
        loss: scalar ECS loss.
    """
    # Normalize embeddings for cosine similarity
    normed = F.normalize(cell_emb, p=2, dim=1)
    
    # Compute cosine similarity matrix
    cos_sim = torch.matmul(normed, normed.T)  # shape (B, B)
    # mask out diagnal elements
    mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
    cos_sim = cos_sim.masked_fill(mask, 0.0)
    # only optimize positive similarities
    cos_sim = F.relu(cos_sim)        
    # Compute ECS loss
    loss = torch.mean(1 - (cos_sim - ecs_threshold) ** 2)
    return loss


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input, target, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    # loss = ((input - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    return loss


def masked_bce_with_logits(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor=torch.tensor(5)
) -> torch.Tensor:
    """
    Compute masked BCE-with-logits loss.
    """
    mask = mask.float()
    loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss
