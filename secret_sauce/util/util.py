import torch

def print_master(arg):
    if torch.distributed.get_rank() == 0:
        print(arg)

def is_master():
    return torch.distributed.get_rank() == 0