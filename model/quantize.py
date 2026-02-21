import os
import time
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model
import argparse

def get_val_loader(data_dir, batch_size=32):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "Validation"), transform=val_transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def calibrate(model, loader, device, num_batches=10):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(images.to(device))

def benchmark(model, loader, device, name="Model"):
    model.eval()
    total_time = 0
    num_images = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= 5: # Benchmark first 5 batches
                break
            start = time.time()
            model(images.to(device))
            total_time += time.time() - start
            num_images += images.size(0)
    
    fps = num_images / total_time
    ms_per_image = (total_time / num_images) * 1000
    print(f"[{name}] Latency: {ms_per_image:.2f} ms/image | Throughput: {fps:.2f} fps")
    return ms_per_image

def main():
    parser = argparse.ArgumentParser(description="Post-Training Dynamic Quantization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FP32 checkpoint")
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--data-dir", type=str, default="../Dataset")
    parser.add_argument("--output", type=str, default="outputs/quantized_efficientnet_b0.pth")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"[INFO] Dynamically Quantizing {args.model} for CPU...")

    # 1. Load FP32 Model
    model_fp32 = get_model(args.model, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()

    # 2. Dynamic Quantization
    # We quantize only the Linear layers (best compatibility for EfficientNet)
    print("[INFO] Applying dynamic quantization to Linear layers...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # 3. Save Quantized Model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model_int8.state_dict(), args.output)
    print(f"[SUCCESS] Quantized model saved to: {args.output}")

    # 4. Benchmark & Comparison
    print("\n" + "="*40)
    print("BENCHMARKING (CPU)")
    print("="*40)
    
    fp32_size = os.path.getsize(args.checkpoint) / (1024*1024)
    int8_size = os.path.getsize(args.output) / (1024*1024)
    
    print(f"FP32 Model Size: {fp32_size:.2f} MB")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
    print(f"Size Reduction:  {fp32_size/int8_size:.1f}x")
    print("-" * 40)
    
    try:
        val_loader = get_val_loader(args.data_dir, batch_size=32)
        benchmark(model_fp32, val_loader, device, name="FP32")
        benchmark(model_int8, val_loader, device, name="INT8")
    except Exception as e:
        print(f"[WARNING] Benchmarking failed: {e}")
    print("="*40)

if __name__ == "__main__":
    main()
