import numbers, argparse, colorama
from termcolor import colored
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif type(v) == str:
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
    elif isinstance(v, numbers.Number):
        assert v in [0, 1]
        if v == 1:
            return True
        if v == 0:
            return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid Value: {type(v)}")
    
    
def warning(str):
    print(colored(str, color='green', on_color='on_red'))


old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


torch.Tensor.__repr__ = tensor_info