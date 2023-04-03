import torch

def set_device():
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	#logger.info(f'Using Device: {device}')
	return device

