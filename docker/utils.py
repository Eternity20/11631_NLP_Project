import torch

def set_device():
	if torch.cuda.is_available():
		print("cuda")
		device = torch.device('cuda')
	else:
		print("cpu")
		device = torch.device('cpu')
	#logger.info(f'Using Device: {device}')
	return device

