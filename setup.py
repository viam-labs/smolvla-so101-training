"""SmolVLA training script for SO-101 sequence datasets.

Runs as a Viam custom training job in the "LeRobot SmolVLA for SO-101"
training container, which already has every dependency below installed at
these exact versions.
"""
from setuptools import find_packages, setup

setup(
    name="smolvla-so101",
    version="0.6.3",
    packages=find_packages(),
    install_requires=[
        "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot@ff7cc3de1de830f5f3276918a013d04bdf9ea4be",
        "viam-sequence-to-lerobot @ git+https://github.com/viam-labs/viam-sequence-to-lerobot@70ab3f253e8b",
        "pandas",
        "pyarrow",
    ],
)
