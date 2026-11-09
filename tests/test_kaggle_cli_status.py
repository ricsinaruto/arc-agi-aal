"""A rejected Kaggle launch must not be recorded as successful; no credentials read."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "kaggle_wrapper", Path(__file__).resolve().parents[1] / "scripts/kaggle_cli.py"
)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_zero_exit_capacity_rejection_is_failure(stream):
    output = {"stdout": b"", "stderr": b""}
    output[stream] = b"Kernel push error: Maximum batch GPU session count of 2 reached.\n"
    result = subprocess.CompletedProcess([], 0, **output)
    assert wrapper.command_status(["kernels", "push", "-p", "bundle"], result) == 1


def test_success_and_original_failures_keep_their_status():
    result = subprocess.CompletedProcess([], 0, b"Kernel version 1 successfully pushed.\n", b"")
    assert wrapper.command_status(["kernels", "push"], result) == 0
    result.returncode = 7
    assert wrapper.command_status(["kernels", "push"], result) == 7


def test_unrelated_commands_are_not_scanned_for_push_messages():
    result = subprocess.CompletedProcess([], 0, b"Kernel push error: historical log\n", b"")
    assert wrapper.command_status(["kernels", "logs", "owner/kernel"], result) == 0


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("operation,prefix", [
    ("create", "Dataset creation error:"),
    ("version", "Dataset version creation error:"),
])
def test_zero_exit_dataset_upload_rejection_is_failure(stream, operation, prefix):
    output = {"stdout": b"", "stderr": b""}
    output[stream] = (prefix + " The size of your files exceeds remaining private quota\n").encode()
    result = subprocess.CompletedProcess([], 0, **output)
    assert wrapper.command_status(["datasets", operation, "-p", "bundle"], result) == 1
    # Reading an archived error message must not change an unrelated command's status.
    assert wrapper.command_status(["datasets", "status", "owner/dataset"], result) == 0


@pytest.mark.parametrize("operation", ["create", "version"])
def test_successful_dataset_upload_and_nonzero_exit_preserved(operation):
    result = subprocess.CompletedProcess([], 0, b"Starting upload for model.safetensors\n", b"")
    assert wrapper.command_status(["datasets", operation], result) == 0
    result.returncode = 9
    assert wrapper.command_status(["datasets", operation], result) == 9
