import torch

checkpoint_path = "checkpoints/train/seqtrack/seqtrack_b256/SEQTRACK_ep0001.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"""
Checpoint Keys:{checkpoint.keys()}\n\n
Epoch:{checkpoint['epoch']}\n\n
Optimizer:{checkpoint['optimizer'].keys()}\n\n
Optimzer Params:{checkpoint['optimizer']['param_groups']}\n\n
""")