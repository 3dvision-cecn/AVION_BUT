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


model = getattr(model_clip, "VideoClassifier")(
    num_classes=400,
    dropout=0.0,
    vision_model="vit_base_patch16_224",
    freeze_temperature=True,
    use_grad_checkpointing=False,
    patch_dropout=False,
    num_frames=1,
    drop_path_rate=False,
    use_fast_conv1=True,
    use_flash_attn=True,
    use_quick_gelu=True,
)

