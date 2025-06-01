from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import avion.models.model_clip as model_clip
from avion.models.utils import inflate_positional_embeds
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv


def generate_label_map(dataset):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            'conf/ek100/EPIC_100_train.csv',
            'conf/ek100/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print('mapping_vn2act', mapping_vn2act)
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        # shape of the labels
        # print(len(labels), len(labels[0]), labels[0])
        # print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open('datasets/EGTEA/action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act


class AVIONForwardModule(nn.Module):
    """
    Wraps preprocessing + AVION classifier
    so it can be TorchScripted and reused.
    Expects a float32 tensor shaped (T, H, W, C) in BGR.
    Returns (argmax, softmax).
    """
    def __init__(self):
        super().__init__()
        self.backbone = self.initialize_backbone()
        self.crop_size = 224

        # register mean / std as buffers so they are part of the .pt file
        mean = torch.tensor([108.3272985, 116.7460125, 104.09373615])
        std  = torch.tensor([68.5005327, 66.6321579, 70.32316305])
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

        # Read class mapping CSV files
        self.noun_to_noun_text = pd.read_csv('conf/ek100/EPIC_100_noun_classes.csv')
        self.verb_to_verb_text = pd.read_csv('conf/ek100/EPIC_100_verb_classes.csv')

        label, mapping_vn2act = generate_label_map('ek100_cls')
        self.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        self.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}


    """
    initializes the backbone model
    """
    def initialize_backbone(self, pretrain_path="conf/checkpoints/avion/avion_pretrain_lavila_vitb_best.pt", fine_tune_path="conf/checkpoints/avion/avion_finetune_cls_lavila_vitb_best.pt"):

        ckpt = torch.load(pretrain_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        print("=> creating model: {}".format(old_args.model))

        model = getattr(model_clip, "CLIP_VITB16")(
            freeze_temperature=True,
            use_grad_checkpointing=False,
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
        checkpoint = torch.load(fine_tune_path, map_location='cpu')
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

        return model


    def forward(self, frames: np.ndarray):                # (T,H,W,C)  BGR
        """
        * frames : (T, H, W, C)  BGR  float32
        * returns: (argmax, softmax)   int64, float32
        """     
        frames = torch.tensor(frames, dtype=torch.bfloat16)
        # --- preprocessing --------------------------------------------------
        # 1. BGR → RGB   and   THWC → TCHW
        frames = frames[:, :, :, [2, 1, 0]].permute(0, 3, 1, 2)

        # 2. resize so the shorter side == crop_size
        t, c, h, w = frames.shape
        scale      = self.crop_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frames = F.interpolate(frames, size=(new_h, new_w),
                               mode="bilinear", align_corners=False)

        # 3. centre-crop
        sh = (new_h - self.crop_size) // 2
        sw = (new_w - self.crop_size) // 2
        frames = frames[:, :, sh : sh + self.crop_size,
                               sw : sw + self.crop_size]

        # 4. normalise exactly as in training (0-255 range!)
        mean = self.mean.view(1, 3, 1, 1)
        std  = self.std.view(1, 3, 1, 1)
        frames = (frames - mean) / std

        # 5. TCHW → CTHW, add batch dim, move to CUDA
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to("cuda").to(torch.bfloat16)

        # 6. imitate the autocast in your update() loop:
        #    keep the graph JIT-safe by *explicitly* casting to BF16
        # --------------------------------------------------------------------
        with torch.no_grad():                     # same as update()
            logits = self.backbone(frames)

        # logits are BF16 – do softmax in FP32 for numerical stability
        probs = torch.softmax(logits.float(), dim=1)
        return logits.argmax(1), probs
    
    def get_text(self, pred):
        n_idx = self.mapping_act2n[pred]
        v_idx = self.mapping_act2v[pred]
        n_text = self.noun_to_noun_text['key'][n_idx]
        v_text = self.verb_to_verb_text['key'][v_idx]

        return v_text, n_text