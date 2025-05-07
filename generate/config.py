CONFIG = {
    #"model": "qwen2.5-coder-3b-instruct",
    #"model": "qwen-coder-plus",
    #"model": "deepseek-r1-distill-llama-70b",
    #"model": "llama-4-scout-17b-16e-instruct",
    #"model": "deepseek-r1",
    "model":"qwen-plus-latest",# qwen3 
    "devices": {
        "cpu": {
            "type": "CPU",
            "cores": 16,
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
        "time":"652ms ",
    }
}