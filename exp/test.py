import torch

model = torch.load('../fine_tuned_sam_vit_h.pth')
print(model.keys())