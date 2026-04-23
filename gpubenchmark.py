import torch
import torch.nn.functional as F
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

if device == "cuda":
    _ = torch.rand(10, device=device)  # Warm-up to initialize GPU


def get_device_info():
    lines = [f"Using device: {device}", f"Device name: {device_name}"]
    if device == "cuda":
        lines.append("GPU warm-up completed.")
    return lines


def time_function(func, device):
    if device == "cuda":
        torch.cuda.synchronize()  # Ensure all previous CUDA operations are complete
    start_time = time.time()
    func()
    if device == "cuda":
        torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    return time.time() - start_time

def benchmark_vector(size, device, iterations=3):
    A = torch.rand(size, device=device)
    B = torch.rand(size, device=device)
    times = []
    for _ in range(iterations):
        times.append(time_function(lambda: A + B, device))
    return sum(times) / len(times), times


def benchmark_matmul(n, device, iterations=2):
    A = torch.rand(n, n, device=device)
    B = torch.rand(n, n, device=device)
    times = []
    for _ in range(iterations):
        times.append(time_function(lambda: torch.matmul(A, B), device))
    return sum(times) / len(times), times


def benchmark_conv2d(batch=1, in_channels=3, out_channels=16, height=256, width=256,
                     kernel_size=3, stride=1, padding=1, iterations=2, device="cpu"):
    input_tensor = torch.rand(batch, in_channels, height, width, device=device)
    weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size, device=device)
    times = []
    for _ in range(iterations):
        times.append(time_function(
            lambda: F.conv2d(input_tensor, weight, bias=None, stride=stride, padding=padding),
            device))
    return sum(times) / len(times), times


def run_vector_benchmark():
    sizes = [100000, 500000, 1000000, 5000000, 10000000]
    results = []
    for s in sizes:
        results.append(f"Benchmarking vector addition for size {s}...")
        avg, times = benchmark_vector(s, device)
        results.append(f"Average time over {len(times)} runs: {avg:.6f} seconds")
    return results


def run_matmul_benchmark():
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    results = []
    for s in sizes:
        results.append(f"Benchmarking matrix multiplication for size {s}x{s}...")
        try:
            avg, times = benchmark_matmul(s, device)
        except RuntimeError as e:
            results.append(f"Skipping {s}x{s}: {e}")
            continue
        results.append(f"Average time over {len(times)} runs: {avg:.6f} seconds")
    return results


def run_conv2d_benchmark():
    results = ["Benchmarking conv2d with 1x3x256x256 input and 16 filters..."]
    avg, times = benchmark_conv2d(device=device)
    results.append(f"Average conv2d time over {len(times)} runs: {avg:.6f} seconds")
    return results


def compare_devices(n):
    results = [f"Comparing CPU and GPU performance for matrix multiplication of size {n}x{n}..."]
    A_cpu = torch.rand(n, n)
    B_cpu = torch.rand(n, n)
    start_time = time.time()
    torch.matmul(A_cpu, B_cpu)
    cpu_time = time.time() - start_time
    results.append(f"CPU time: {cpu_time:.6f} seconds")

    if not torch.cuda.is_available():
        results.append("CUDA is not available. Skipping GPU benchmark.")
        return results

    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    torch.matmul(A_gpu, B_gpu)
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    results.append(f"GPU time: {gpu_time:.6f} seconds")
    results.append(f"Speedup (CPU/GPU): {cpu_time / gpu_time:.2f}x")
    return results


def unified_benchmark():
    results = []
    results.extend(get_device_info())
    results.append("")
    results.extend(run_vector_benchmark())
    results.append("")
    results.extend(run_matmul_benchmark())
    results.append("")
    results.extend(run_conv2d_benchmark())
    results.append("")
    results.extend(compare_devices(1024))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Benchmarking")
    parser.add_argument("--task", choices=["vector", "matmul", "conv2d", "compare", "all"], default="all", help="Task to benchmark")
    args = parser.parse_args()
    if args.task == "vector":
        results = run_vector_benchmark()
    elif args.task == "matmul":
        results = run_matmul_benchmark()
    elif args.task == "conv2d":
        results = run_conv2d_benchmark()
    elif args.task == "compare":
        results = compare_devices(1024)
    else:
        results = unified_benchmark()
    for line in results:
        print(line)
