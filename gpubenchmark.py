import torch
import time
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Running warm-up iterations...")
    _ = torch.rand(10, device=device)  # Warm-up to initialize GPU
    
def time_function(func):
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all previous CUDA operations are complete
    start_time = time.time()
    func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    return time.time() - start_time

def benchmark_vector(size, device):
    A = torch.rand(size, device=device)
    B = torch.rand(size, device=device)
    def run():
        C = A + B
    return time_function(run)

def benchmark_matmul(n, device):
    A = torch.rand(n, n, device=device)
    B = torch.rand(n, n, device=device)
    def run():
        C = torch.matmul(A, B)
    return time_function(run)

def run_vector_benchmark():
    sizes = [100000, 500000, 1000000, 5000000, 10000000]
    for s in sizes:
        print(f"Benchmarking vector addition for size {s}...")
        t = benchmark_vector(s, device)
        print(f"Time taken for vector addition of size {s}: {t:.6f} seconds")

def run_matmul_benchmark():
    sizes = [256,512,1024,2048,4096,8192]
    for s in sizes:
        print(f"Benchmarking matrix multiplication for size {s}x{s}...")
        try:
            t = benchmark_matmul(s, device)
        except RuntimeError as e:
            print(f"Skipping {s}x{s}: {e}")
            continue
        print(f"Time taken for matrix multiplication of size {s}x{s}: {t:.6f} seconds")
      
def compare_devices(n):
    print(f"Comparing CPU and GPU performance for matrix multiplication of size {n}x{n}...")
    # Benchmark on CPU
    A_cpu = torch.rand(n, n)
    B_cpu = torch.rand(n, n)
    start_time = time.time()
    C_cpu = torch.matmul(A_cpu, B_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.6f} seconds")

    # Benchmark on GPU
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping GPU benchmark.")
        return
    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)
    start_time = time.time()
    C_gpu = torch.matmul(A_gpu, B_gpu)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to finish
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.6f} seconds") 
    print(f"Speedup (CPU/GPU): {cpu_time / gpu_time:.2f}x")     
        
def unified_benchmark():
    run_vector_benchmark()
    run_matmul_benchmark()
    compare_devices(1024)  # Compare CPU and GPU performance for a 1024x1024 matrix
parser = argparse.ArgumentParser(description="GPU Benchmarking ")
parser.add_argument("--task", choices=["vector", "matmul","compare","all"], default="all", help="Task to benchmark")
args = parser.parse_args()
if args.task == "vector":
    run_vector_benchmark()
elif args.task == "matmul":
    run_matmul_benchmark()
elif args.task == "compare":
    compare_devices(1024)
else:
    unified_benchmark()
