import math

def get_warmup_steps(warmup_steps: int, warmup_ratio:float, num_training_steps: int):
    return warmup_steps if warmup_steps > 0 else math.ceil(num_training_steps * warmup_ratio)
