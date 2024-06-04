import torch

def check_gpu():
    """this is to check gpu

    Raises:
        Exception: if GPU is not available
    """
    if not torch.cuda.is_available():
      raise Exception("GPU not availalbe. CPU training will be too slow.")

    print("device name", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    check_gpu()