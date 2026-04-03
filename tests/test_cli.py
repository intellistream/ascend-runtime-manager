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
