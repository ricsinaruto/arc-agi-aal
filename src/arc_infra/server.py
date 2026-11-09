"""Hardware identification for GPU benchmark records."""

import subprocess


def hardware():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,power.limit,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1 or not (("RTX PRO 6000" in rows[0] and "Blackwell" in rows[0])
                              or "B200" in rows[0] or "B300" in rows[0]):
        raise RuntimeError(f"Expected exactly one RTX PRO 6000 Blackwell, B200 or B300; got {rows}")
    name, memory, power, driver = [part.strip() for part in rows[0].split(",")]
    if float(memory) < 90000:
        raise RuntimeError(f"Expected 96 GB GPU; reported only {memory} MiB")
    return {
        "gpu_name": name,
        "memory_mib": float(memory),
        "power_limit_w": power,
        "driver": driver,
        "gpu_count": 1,
    }
