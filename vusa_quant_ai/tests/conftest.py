import os
import sys
import tempfile
import warnings
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="vusa_tests_")
os.environ["VUSA_DATA_DIR"] = _TMP  # must be set before the package is imported
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")  # headless CI often runs as root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def settings():
    from vusa_quant_ai.config.settings import Settings

    s = Settings()
    s.models.history_years = 9
    s.models.n_splits = 3
    s.monte_carlo_paths = 1000
    return s


@pytest.fixture(scope="session")
def bundle(settings):
    from vusa_quant_ai.data.service import DataService

    return DataService(settings).load(synthetic=True)


@pytest.fixture(scope="session")
def features(bundle):
    from vusa_quant_ai.features.builder import build_features

    return build_features(bundle)


@pytest.fixture(scope="session")
def regimes(bundle, features):
    from vusa_quant_ai.regimes.detector import detect_regimes

    return detect_regimes(bundle.target["close"], features.frame, bundle.assets["VIX"]["close"])
