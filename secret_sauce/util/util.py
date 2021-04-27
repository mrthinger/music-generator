import argparse
import torch
import deepspeed
import torch.distributed as dist

def print_master(arg):
    if is_master():
        print(arg)

def is_master():
    return dist.get_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    args = parser.parse_args()
    return args


def wait_for_debugger_on_master():
    if is_master():
        wait_for_debugger()

def wait_for_debugger():
    import debugpy
    debugpy.listen(5763)
    print('waiting')
    debugpy.wait_for_client()
    print('connected')