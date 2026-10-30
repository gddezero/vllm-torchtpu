# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion-worker collectives under DEFER_AND_FUSE on real TPU ranks.

Diffusion workers run eager under DEFER_AND_FUSE while vllm-torchtpu turns
TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS off process-wide. This pins:

  1. With OmniTpuPlatform._materialize_collective_inputs() applied, int64
     broadcast / all-gather and broadcast_object_list deliver the source
     values on every rank.
  2. Without it (the positive control), the same int64 broadcast reaches the
     other ranks corrupted (seen on v6e-8). Where it does not corrupt (another
     TPU generation, or a torch_tpu that fixed it), the control arm skips: the
     fixed arm alone then cannot tell the workaround from a no-op there.

Each arm is its own worker group: torch_tpu reads the variable once. Both arms
import the same modules; only the fixed arm applies the workaround.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
]

WORLD_SIZE = 8
LAUNCH_TIMEOUT_SECONDS = 600
SOURCE = [42, -7, 2**40 + 5]


def _prepare_worker_env() -> dict[str, str]:
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # What sitecustomize.py sets in every vllm-torchtpu process.
    env["TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"] = "false"
    return env


def _run_worker(result_dir: Path, arm: str) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank, "arm": arm}
    try:
        # vllm_omni first, as in a diffusion worker: importing the platform
        # module on its own runs into vllm_omni's plugin import cycle.
        import vllm_omni.diffusion.ipc  # noqa: F401

        from vllm_torchtpu.omni.platform import OmniTpuPlatform

        if arm == "fixed":
            OmniTpuPlatform._materialize_collective_inputs()
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from torch_tpu._internal import execution_mode

        # The mode TpuPlatform.check_and_update_config sets in diffusion workers.
        execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE

        src = torch.tensor(
            SOURCE if rank == 0 else [0] * len(SOURCE), dtype=torch.long, device="tpu"
        )
        dist.broadcast(src, src=0)
        result["broadcast"] = src.cpu().tolist()
        if arm == "fixed":
            mine = torch.tensor([100 + rank], dtype=torch.long, device="tpu")
            gathered = torch.zeros(world, dtype=torch.long, device="tpu")
            dist.all_gather_into_tensor(gathered, mine)
            result["all_gather"] = gathered.cpu().tolist()
            payload: list[dict[str, int] | None] = [{"a": 1} if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            result["object"] = payload[0]
    except BaseException:
        result["error"] = traceback.format_exc()[-4000:]
    (result_dir / f"rank_{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    # Skip process-group teardown: with corrupted collectives in flight (the
    # control arm) the source rank can hang there, and the result is written.
    os._exit(0)


def _launch(result_dir: Path, arm: str) -> tuple[int | None, str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--arm",
        arm,
        "--result-dir",
        str(result_dir),
    ]
    log_path = result_dir / "workers.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=_prepare_worker_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=LAUNCH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode, log_path.read_text(errors="replace")


def _results(result_dir: Path) -> list[dict]:
    out = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


def assert_fixed(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    for r in results:
        assert r["broadcast"] == SOURCE, r
        assert r["all_gather"] == [100 + i for i in range(WORLD_SIZE)], r
        assert r["object"] == {"a": 1}, r


def check_control_corrupts(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    if all(r["broadcast"] == SOURCE for r in results):
        pytest.skip(
            "positive control did not reproduce the int64 corruption here; the "
            "fixed arm alone does not show the workaround is needed on this "
            "TPU / torch_tpu (it was on v6e-8 with torch_tpu 0.2.0)"
        )


@pytest.mark.parametrize("arm", ["fixed", "control"])
def test_diffusion_worker_collectives_under_defer_and_fuse(tmp_path, arm):
    result_dir = tmp_path / arm
    result_dir.mkdir()
    returncode, output = _launch(result_dir, arm)
    results = _results(result_dir)
    print(json.dumps(results, indent=1))
    assert returncode == 0 or any("error" in r for r in results), output[-12000:]
    if arm == "fixed":
        assert_fixed(results)
    else:
        check_control_corrupts(results)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=["fixed", "control"], required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker:
        raise ValueError("expected --worker")
    _run_worker(args.result_dir, args.arm)


if __name__ == "__main__":
    _main()
