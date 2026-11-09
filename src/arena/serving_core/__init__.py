"""The arena's serving core: the native vLLM server start-up and its checks.

Moved from the retired ``qwen_baselines`` package (4 October cleanup): ``serving`` (``start``, ``preflight``,
``worker_probe``, ``verify_hardware``), the request layer (``model``), ``config``, ``clock`` and the sandbox
probe's ``jail`` / ``sandbox`` / ``worker`` / ``segmentation``. ``python -m arena.serving_core.cli serve-native``
replaces ``python -m qwen_baselines.cli serve-native``.
"""
