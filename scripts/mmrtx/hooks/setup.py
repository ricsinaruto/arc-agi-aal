from setuptools import setup
setup(name="mmrtx_hooks", version="0.1", packages=["mmrtx_hooks"],
      entry_points={"vllm.general_plugins": ["mmrtx_hooks = mmrtx_hooks:register"]})
