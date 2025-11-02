import subprocess
import torch

# Cached device to ensure the selection logic (and its print) runs only once
_cached_device: torch.device | None = None
_cached_all: list[torch.device] | None = None

def select_device(prefer_memory: bool = True) -> torch.device:
    """
    Selects and returns the most appropriate torch.device.
    
    1) If CUDA is not available -> CPU
    2) If CUDA is available and prefer_memory=True -> use nvidia-smi to pick
       the GPU with the most free memory
    3) If CUDA is available and prefer_memory=False -> use cuda:0
    """
    global _cached_device
    # Return cached device if already chosen
    if _cached_device is not None:
        return _cached_device

    # Case 1: No CUDA -> CPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        _cached_device = torch.device("cpu")
        return _cached_device

    # Case 2 & 3: CUDA is available
    if prefer_memory:
        try:
            # Query free memory on all GPUs (MiB) via nvidia-smi
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            free_memories = [int(line) for line in output.strip().splitlines()]
            # Pick the GPU index with the maximum free memory
            best_index = max(range(len(free_memories)), key=lambda i: free_memories[i])
            print(f"Selecting GPU:{best_index} (free memory: {free_memories[best_index]} MiB)")
            _cached_device = torch.device(f"cuda:{best_index}")
        except Exception as e:
            # Fall back to default GPU 0 on failure
            print(f"nvidia-smi query failed ({e}); falling back to cuda:0")
            _cached_device = torch.device("cuda:0")
    else:
        # Always use GPU 0 when not preferring memory-based selection
        print("Using default GPU:0")
        _cached_device = torch.device("cuda:0")

    return _cached_device

def list_available_devices(prefer_memory: bool = True) -> list[torch.device]:
    """
    Returns a list of all available devices:
    - If no CUDA, returns [cpu]
    - If CUDA and prefer_memory=True, returns GPUs sorted by free memory desc
    - If CUDA and prefer_memory=False, returns [cuda:0, cuda:1, ...]
    """
    global _cached_all
    if _cached_all is not None:
        return _cached_all

    if not torch.cuda.is_available():
        _cached_all = [torch.device("cpu")]
        return _cached_all

    n = torch.cuda.device_count()
    if not prefer_memory:
        _cached_all = [torch.device(f"cuda:{i}") for i in range(n)]
    else:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,nounits,noheader"],
                encoding="utf-8"
            )
            free_memories = [int(l) for l in out.strip().splitlines()]
            # GPU indices sorted by free memory descending
            sorted_idx = sorted(range(n), key=lambda i: free_memories[i], reverse=True)
            _cached_all = [torch.device(f"cuda:{i}") for i in sorted_idx]
        except Exception:
            _cached_all = [torch.device(f"cuda:{i}") for i in range(n)]
    return _cached_all