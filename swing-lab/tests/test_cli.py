"""명령줄 오류 처리."""

from __future__ import annotations

import shutil
from pathlib import Path

from conftest import random_walk_prices
from typer.testing import CliRunner

from swinglab.cli import app
from swinglab.pipeline.storage import clean_path, write_prices

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_experiments_run_reports_non_mapping_yaml_as_refusal(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy(CONFIG_DIR / "stage1.yaml", config_dir / "stage1.yaml")
    shutil.copy(CONFIG_DIR / "backtest.yaml", config_dir / "backtest.yaml")
    (config_dir / "universe.yaml").write_text("- SPY\n- XLK\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    write_prices(random_walk_prices(["SPY", "XLK"], 30), clean_path(data_dir))

    result = CliRunner().invoke(
        app,
        [
            "experiments", "run", "-f", "A",
            "--config-dir", str(config_dir),
            "--data-dir", str(data_dir),
            "--results-dir", str(tmp_path / "results"),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # 처리되지 않은 TypeError가 아님
    assert "실행 거부:" in result.output
    assert "YAML 최상위는 매핑이어야 합니다" in result.output
    assert not (tmp_path / "results").exists()
