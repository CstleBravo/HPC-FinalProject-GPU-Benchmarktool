# Portable Windows Build

This project can be packaged as a portable Windows folder that does not require Python to be installed on the target machine.

## Build environment

This repository uses a dedicated local build environment at:

```text
.venv-portable-build\
```

The current build environment contains:

- `PyInstaller`
- `torch 2.7.1+cu118`
- `numpy`

## Build

Run:

```bat
portable_build\build_portable.bat
```

That creates:

```text
dist\GPUBenchmark\
```

Copy the entire `GPUBenchmark` folder to the destination machine or a USB drive, then launch `GPUBenchmark.exe`.

## Current limitation

The build environment now bundles a CUDA-enabled PyTorch wheel (`torch 2.7.1+cu118`). That means the portable folder carries the CUDA runtime libraries with it.

GPU acceleration on the destination machine still depends on having a compatible NVIDIA GPU driver installed. On this packaging machine, `torch.cuda.is_available()` is currently `False`, so the bundle can be created here, but live CUDA execution still needs to be verified on a machine with an accessible NVIDIA driver.
