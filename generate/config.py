CONFIG = {
    "model": "deepseek-v3",# qwen3 
    "devices": {
        "cpu": {
            "type": "CPU",
            "cores": 12,
            "threads": 64,
            "memory": "32GB",
            "available": True,
        },
        "gpu": {
            "type": " NVIDIA GPU RTX 3090",
            "compute_capability": 8.0,  # CUDA compute capability
            "memory": "24GB",
            "threads_per_block": 1024,
            "available": True,
        },
    }
}