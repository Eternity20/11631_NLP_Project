import torch

def set_device():
	print(f'Is CUDA available: {torch.cuda.is_available()}')
	print(f'Nr of Devices:{torch.cuda.device_count()}')
	if torch.cuda.is_available():
		# print("cuda")
		device = torch.device('cuda')
	else:
		# print("cpu")
		device = torch.device('cpu')
	#logger.info(f'Using Device: {device}')
	return device

