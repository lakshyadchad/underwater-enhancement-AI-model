import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG19_Perceptual(nn.Module):
    def __init__(self):
        super(VGG19_Perceptual, self).__init__()
        # Load VGG19 pre-trained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Slice the network: We only want the first 35 layers (up to Conv4_4)
        # This captures high-level textures but ignores high-level logic (like "is this a dog?")
        self.features = nn.Sequential(*list(vgg.children())[:35]).eval()
        
        # FREEZE THE WEIGHTS: We are not training VGG. We are just using it as a judge.
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

class UnderwaterLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(UnderwaterLoss, self).__init__()
        
        # 1. The Basic Pixel Judge
        self.l1_loss = nn.L1Loss()
        
        # 2. The Perceptual Judge (VGG)
        self.vgg = VGG19_Perceptual().to(device)  # Move to GPU
        
        # Weights: How much do we care about each part?
        # These are "Hyperparameters" you can tune.
        self.lambda_l1 = 1.0      # Base structural truth
        self.lambda_percep = 0.2  # Texture realism (don't make it too high or you get artifacts)
        self.lambda_red = 0.5     # Force Red recovery (Crucial for Indian Ocean)
        self.lambda_cosine = 0.2  # Color tone accuracy

    def forward(self, generated, target):
        # --- 1. PIXEL LOSS (L1) ---
        # "Are the pixels numerically close?"
        loss_pixel = self.l1_loss(generated, target)
        
        # --- 2. PERCEPTUAL LOSS (VGG) ---
        # "Do the images LOOK the same to a human eye?"
        # We must normalize images for VGG (standard practice)
        # VGG expects images roughly in range [0, 1] or normalized. 
        # Assuming our generator outputs [-1, 1] (Tanh), we shift to [0, 1]
        gen_norm = (generated + 1) / 2
        tgt_norm = (target + 1) / 2
        
        gen_features = self.vgg(gen_norm)
        tgt_features = self.vgg(tgt_norm)
        loss_percep = self.l1_loss(gen_features, tgt_features)
        
        # --- 3. RED CHANNEL LOSS (Custom Physics) ---
        # PyTorch images are (Batch, Channel, Height, Width)
        # Index 0 = Red, 1 = Green, 2 = Blue (if RGB)
        # NOTE: Verify if your OpenCV loader uses BGR or RGB. 
        # If using standard PyTorch `ImageFolder`, it loads RGB.
        gen_red = generated[:, 0, :, :]
        tgt_red = target[:, 0, :, :]
        loss_red = self.l1_loss(gen_red, tgt_red)
        
        # --- 4. COSINE SIMILARITY LOSS (Color/Hue) ---
        # "Is the color pointing in the right direction?"
        # Cosine Similarity returns 1.0 for perfect match, -1.0 for opposite.
        # We want to minimize Loss, so we do (1 - Similarity).
        # dim=1 means we compare along the Color Channel axis.
        similarity = F.cosine_similarity(generated, target, dim=1)
        loss_cosine = 1 - similarity.mean()

        # --- FINAL COCKTAIL ---
        total_loss = (self.lambda_l1 * loss_pixel) + \
                     (self.lambda_percep * loss_percep) + \
                     (self.lambda_red * loss_red) + \
                     (self.lambda_cosine * loss_cosine)
                     
        return total_loss
    
