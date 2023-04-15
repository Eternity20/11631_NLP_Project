import torch
import sys
if __name__=="__main__":
    print(f'Is CUDA available: {torch.cuda.is_available()}')
    print(f'Nr of Devices:{torch.cuda.device_count()}')
    sys.exit(0)