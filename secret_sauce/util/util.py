import torch

def print_master(arg):
    if torch.distributed.get_rank() == 0:
        print(arg)
