import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from config_sd import HEIGHT, WIDTH

class RewardModel(torch.nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
    
    def load_from_hf(self, device):
        ckpt_path = hf_hub_download(repo_id="sakshamrig/smb-mlp", filename="smb_mlp.pt", repo_type="model")
        ckpt = torch.load(ckpt_path, map_location=device)
        in_dim = ckpt["in_dim"]
        print("Loading RewardModel with in_dim:", in_dim)

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        ).to(device)
        self.reg_head = nn.Linear(256, 2).to(device)
        self.cls_head = nn.Linear(256, 1).to(device)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.reg_head.load_state_dict(ckpt["reg_out"])
        self.cls_head.load_state_dict(ckpt["cls_out"])
        self.backbone.eval()
        self.reg_head.eval()
        self.cls_head.eval()

    def forward(self, latent_stack: torch.Tensor):
        latent_stack = latent_stack.flatten(start_dim=1)
        logits = self.backbone(latent_stack)
        vel = self.reg_head(logits)
        done = F.sigmoid(self.cls_head(logits))
        return vel, done