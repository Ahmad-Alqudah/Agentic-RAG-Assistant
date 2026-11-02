import pytest
import app.api.main as api_module


@pytest.fixture(autouse=True)
def disable_real_model_loading(monkeypatch):

    monkeypatch.setattr(
        api_module,
        "get_engine",
        lambda: None,
    )

    monkeypatch.setattr(
        api_module,
        "build_graph",
        lambda: api_module.graph,
    )