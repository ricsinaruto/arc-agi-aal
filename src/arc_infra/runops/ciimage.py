"""The shared CPU test/preflight image: one Modal image with the third-party deps of every test in
main and the arm branches (PTH's lockstep test once could not run on Modal CPU: no transformers), used by
tools/ops/test_modal.py, tools/ops/ci.py and every arm's CPU preflight.

    from arc_infra.runops.ciimage import image
    img = image()                                   # the trainer stack's torch/transformers, CPU wheels
    img = image(extra=("rclone",), apt=("rclone",))  # a preflight's own additions

A test that needs a GPU-only package (triton, fla, vllm kernels) importorskips it; everything else must import here.
"""
TORCH = "2.14.0"  # the trainers' torch (heal / RL trainer images); CPU wheel
TORCHVISION = "0.29.0"
TRANSFORMERS = "5.18.0"
DEPS = ("aiohttp>=3.12,<4", "arc-agi==0.9.9", "httpx>=0.28,<0.29", "imageio>=2.37,<3", "matplotlib>=3.10,<4",
        "numpy>=2.2,<3", "pillow>=11,<13", "pydantic>=2,<3", "pyyaml>=6,<7", "requests>=2.32,<3", "scipy>=1.15,<2",
        "safetensors", "jinja2", "tokenizers", "huggingface_hub>=0.34", "accelerate==1.15.0", "peft==0.21.0",
        "python-dotenv", "modal==1.5.5", "pytest>=8,<10", "pytest-asyncio>=0.25,<2", "pytest-xdist>=3.6,<4",
        "pytest-timeout>=2.3,<3")


def image(torch=TORCH, transformers=TRANSFORMERS, torchvision=TORCHVISION, extra=(), apt=()):
    import modal

    return (modal.Image.debian_slim(python_version="3.12").apt_install("git", "procps", *apt)  # ps: stall dumps
            .pip_install(f"torch=={torch}", f"torchvision=={torchvision}",
                         index_url="https://download.pytorch.org/whl/cpu")
            .pip_install(*DEPS, f"transformers=={transformers}", *extra))
