import torch

# =================================================
# Device & reproducibility
# =================================================

seed: int = 42
device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch_dtype: torch.dtype = (
    torch.float32
    if (torch.cuda.is_available() or torch.backends.mps.is_available())
    else torch.float
)