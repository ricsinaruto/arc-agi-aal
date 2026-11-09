"""GPU-efficiency test of the public Duck TAAF loop on the Pennyroyal SGLang build (Kaggle).

Separate from the vLLM serving code in ``arc_infra``: the runtime comes from its own dataset
(``richardcsaky/arc-sglang-runtime``, the ``ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.1``
image's ``/opt`` tree), the server is launched with the recipe's exact ``sglang serve`` argv
minus the NIXL host cache, and the public loop is pointed at it through its analyzer
environment at a 128K window.
"""
