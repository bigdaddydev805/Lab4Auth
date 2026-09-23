import pytest

from helpco.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """A mock-brain, lockstep config with its own save directory."""
    return load_config(overrides={
        "world": {"save_dir": str(tmp_path / "save"), "start": "2027-03-15", "timezone": "America/Los_Angeles",
                  "seed": 3},
        "llm": {"provider": "mock"},
        "clock": {"mode": "lockstep"},
    })
