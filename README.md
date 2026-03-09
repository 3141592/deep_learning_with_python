# Working through Deep Learning With Python, 2 ed.

## Setup
- Python 3.10.6
- tensorflow 2.11
-- pip3 install tensorflow
- cupy
-- pip install cupy --no-cache-dir
- CUDA support
-- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

## Shared Data Repository

This project depends on the shared asset repository:

https://github.com/<yourname>/ai_shared_data

Datasets and model assets are **not stored in this repository**.  
They are managed through `ai_shared_data`.

Example setup:

```bash
git clone https://github.com/<yourname>/ai_shared_data
pip install -e ai_shared_data
```

See that repository for dataset download and configuration instructions.