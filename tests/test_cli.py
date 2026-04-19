from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from hust_ascend_manager import cli


def test_env_install_hook_dispatches(capsys):
    with (
        patch(
            "hust_ascend_manager.cli.install_conda_env_hook",
            return_value=(
                Path("/tmp/conda/etc/conda/activate.d/hust-ascend-manager.sh"),
                Path("/tmp/conda/etc/conda/deactivate.d/hust-ascend-manager.sh"),
            ),
        ),
        patch.object(
            sys,
            "argv",
            ["hust-ascend-manager", "env", "--install-hook", "--conda-prefix", "/tmp/conda"],
        ),
    ):
        rc = cli.main()

    captured = capsys.readouterr()
    assert rc == 0
    assert "Installed activate hook:" in captured.out
    assert "Installed deactivate hook:" in captured.out


def test_runtime_repair_dispatches_plugin_options():
    with (
        patch("hust_ascend_manager.cli.repair_vllm_runtime", return_value=0) as repair_mock,
        patch.object(
            sys,
            "argv",
            [
                "hust-ascend-manager",
                "runtime",
                "repair",
                "--repo",
                "/workspace/vllm-hust",
                "--install-plugin",
                "--plugin-repo",
                "/workspace/vllm-ascend-hust",
                "--plugin-package",
                "vllm-ascend==0.13.0",
            ],
        ),
    ):
        rc = cli.main()

    assert rc == 0
    repair_mock.assert_called_once_with(
        "/workspace/vllm-hust",
        None,
        skip_torch_install=False,
        skip_build_deps=False,
        skip_rebuild=False,
        install_plugin=True,
        plugin_repo="/workspace/vllm-ascend-hust",
        plugin_package="vllm-ascend==0.13.0",
    )
