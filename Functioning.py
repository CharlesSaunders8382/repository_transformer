import torch

device = torch.device('cpu')
print(torch.__version__)
print(torch.cuda.is_available())