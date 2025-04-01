import argparse
from collections import OrderedDict
from functools import partial
import json
import os
import pickle
import time
import numpy as np
import pandas as pd

import kornia as K
import scipy
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from avion.data.clip_dataset import get_downstream_dataset
from avion.data.tokenizer import tokenize
from avion.data.transforms import Permute

import avion.models.model_clip as model_clip
from avion.models.utils import inflate_positional_embeds
from avion.optim.schedulers import cosine_scheduler
import avion.utils.distributed as dist_utils
from avion.utils.evaluation_ek100cls import get_marginal_indexes, get_mean_accuracy, marginalize
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan, generate_label_map
import ast
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read class mapping CSV files
noun_to_noun_text = pd.read_csv('datasets/EK100/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv')
verb_to_verb_text = pd.read_csv('datasets/EK100/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv')

# Load label mapping
label, mapping_vn2act = generate_label_map('ek100_cls')
mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}

# Load pretraining checkpoint and update state dict keys
ckpt = torch.load("/home/eongan/ethz/3d_vision/AVION/avion_pretrain_lavila_vitb_best.pt", map_location='cpu')
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
    state_dict[k.replace('module.', '')] = v

old_args = ckpt['args']
print("=> creating model: {}".format(old_args.model))

model = getattr(model_clip, "CLIP_VITB16")(
    freeze_temperature=True,
    use_grad_checkpointing=True,
    context_length=old_args.context_length,
    vocab_size=old_args.vocab_size,
    patch_dropout=0,
    num_frames=16,
    drop_path_rate=0.1,
    use_fast_conv1=True,
    use_flash_attn=True,
    use_quick_gelu=True,
    project_embed_dim=old_args.project_embed_dim,
    pretrain_zoo=old_args.pretrain_zoo,
    pretrain_path=old_args.pretrain_path,
)

model.logit_scale.requires_grad = False

state_dict = inflate_positional_embeds(
    model.state_dict(), state_dict,
    num_frames=16,
    load_temporal_fix='bilinear',
)
model.load_state_dict(state_dict, strict=True)

model = model_clip.VideoClassifier(
    model.visual,
    dropout=0.0,
    num_classes=3806
)
model = model.cuda()

# Load finetuning checkpoint correctly
checkpoint = torch.load("/home/eongan/ethz/3d_vision/AVION/avion_finetune_cls_lavila_vitb_best.pt", map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))

# Fix module prefix issue in checkpoint
if 'state_dict' in checkpoint:
    state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        # Remove the 'module.' prefix from keys
        name = k.replace('module.', '')
        state_dict[name] = v
    
    # Now load the fixed state dict
    result = model.load_state_dict(state_dict, strict=False)
    print("Loaded model weights:", result)
else:
    print("Error: Checkpoint doesn't contain 'state_dict' key")

model.eval()
# AFTER loading weights, convert to bfloat16
model = model.to(torch.bfloat16)
print("Model loaded and converted to bfloat16")

# Define normalization parameters and crop size
mean, std = [108.3272985, 116.7460125, 104.09373615000001], [68.5005327, 66.6321579, 70.32316305]
# Important: The model was trained expecting these values in 0-255 range, not 0-1!
crop_size = 336 if old_args.model.endswith("_336PX") else 224
print("Crop size:", crop_size)
print(f"Using mean: {mean}, std: {std}")

# Load image file paths from folder
IMAGE_FOLDER = "/home/eongan/EPIC-KITCHENS/P01/rgb_frames"
image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])[:1000]  # Limit to 1000 frames for testing
print(f"Loaded {len(image_files)} frames")

# Setup matplotlib for display
fig, ax = plt.subplots(figsize=(10, 6))
frames = []
for img_path in image_files:
    # Read with OpenCV to match Decord format
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: Could not read {img_path}")
        continue
    frames.append(frame)

print(f"Successfully loaded {len(frames)} frames")
im = ax.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
ax.axis('off')

# Create text display for predictions
pred_text = ax.text(0.5, 0.95, "", horizontalalignment='center', 
                   verticalalignment='center', transform=ax.transAxes,
                   color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

# Define uniform frame sampling function
def sample_frames(video_frames, num_frames=16):
    """Sample frames uniformly from the video."""
    if len(video_frames) <= num_frames:
        # Duplicate frames if video is too short
        return video_frames * (num_frames // len(video_frames) + 1)
    
    # Calculate sampling indices
    indices = np.linspace(0, len(video_frames)-1, num_frames, dtype=int)
    return [video_frames[i] for i in indices]

# Global variables for tracking
idx = 0
buffer_size = 32  # Larger buffer to ensure we have enough frames
frame_buffer = []

def update(frame):
    global idx, frame_buffer
    
    # Display frame in RGB for visualization
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im.set_data(rgb_frame)
    
    # Add frame to buffer (keep in BGR as per Decord)
    frame_buffer.append(frame)
    idx += 1
    
    # Process when buffer has enough frames
    if len(frame_buffer) >= buffer_size:
        with torch.no_grad():
            # Sample 16 frames uniformly from buffer
            sampled_frames = sample_frames(frame_buffer, num_frames=16)
            
            # Important: Keep frames in BGR format to match training pipeline
            # Stack frames into numpy array (T,H,W,C)
            video_array = np.stack(sampled_frames, axis=0).astype(np.float32)
            
            # Convert to torch tensor (T,H,W,C)
            video_tensor = torch.from_numpy(video_array)
            
            # Print shape for debugging
            print(f"Video tensor shape before processing: {video_tensor.shape}")
            
            # CORRECT BGR->RGB conversion: OpenCV loads as BGR, need to swap to RGB
            # BGR (0,1,2) → RGB (2,1,0)
            video_tensor = video_tensor[:, :, :, [2, 1, 0]]  # Correct BGR → RGB swap
            
            # Now permute to (T,C,H,W) for resizing
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # THWC -> TCHW
            
            # Resize frames to match crop_size (shortest side)
            t, c, h, w = video_tensor.shape
            scale = crop_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            video_tensor = F.interpolate(
                video_tensor, size=(new_h, new_w),
                mode='bilinear', align_corners=False
            )
            
            # Center crop to crop_size x crop_size
            start_h = (new_h - crop_size) // 2
            start_w = (new_w - crop_size) // 2
            video_tensor = video_tensor[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
            
            # CRITICAL FIX: Apply normalization EXACTLY as in training
            # Use original mean/std directly (not divided by 255)
            # The model expects 0-255 input range normalized with these values
            for i in range(3):  # Apply normalization per channel
                video_tensor[:, i] = (video_tensor[:, i] - mean[i]) / std[i]
                
            # Permute to C, T, H, W as expected by model
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # TCHW -> CTHW
            
            # Add batch dimension and move to CUDA
            video_tensor = video_tensor.unsqueeze(0).to('cuda')  # Use float32 initially
            
            # Debug tensor values BEFORE model inference
            print(f"Input tensor max: {video_tensor.max().item()}")
            print(f"Input tensor min: {video_tensor.min().item()}")
            print(f"Input tensor mean: {video_tensor.mean().item()}")
            
            # Forward pass with proper precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(video_tensor)
            
            # Print the shape and value range of logits for debugging
            print(f"Logits shape: {logits.shape}")
            print(f"Logits range: [{logits.min().item()}, {logits.max().item()}]")
            
            # Get top 5 predictions
            probs = torch.softmax(logits, dim=1)
            
            # Get top 5 predictions for better debugging
            print("Maximum logits:", logits.max().item())
            print("MAXIMUM logits:", logits.argmax(dim=1)[0].item())
            print("Maximum probability:", probs.max().item())
            print("MAXIMUM probability:", probs.argmax(dim=1)[0].item())
            top5_values, top5_indices = torch.topk(probs, 5, dim=1)
            
            # Display top prediction
            top_pred = top5_indices[0, 0].item()
            noun_idx = mapping_act2n[top_pred]
            verb_idx = mapping_act2v[top_pred]
            
            # Get class names
            noun_text = noun_to_noun_text['key'][noun_idx]
            verb_text = verb_to_verb_text['key'][verb_idx]
            
            # Display prediction
            pred_text.set_text(f"Action: {verb_text} {noun_text}")
            
            # Print all top 5 predictions for debugging
            print("\nTop 5 predictions:")
            for i in range(5):
                pred_idx = top5_indices[0, i].item()
                prob = top5_values[0, i].item()
                n_idx = mapping_act2n[pred_idx]
                v_idx = mapping_act2v[pred_idx]
                n_text = noun_to_noun_text['key'][n_idx]
                v_text = verb_to_verb_text['key'][v_idx]
                print(f"{i+1}. {v_text} {n_text} ({prob:.4f})")
            
            # Clear buffer
            frame_buffer = frame_buffer[-16:]  # Keep last 16 frames for continuity
            
    return [im, pred_text]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
plt.show()
