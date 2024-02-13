import torch

def main():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: Name: {gpu_properties.name}, Compute Capability: {gpu_properties.major}.{gpu_properties.minor}, Memory: {gpu_properties.total_memory / 1e9} GB")

if __name__ == "__main__":
    main()
