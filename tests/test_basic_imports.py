"""Basic import tests for Cross-Asset Alpha Engine.

This module tests that all main components can be imported successfully
and that the package structure is correct.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_main_package_import():
    """Test that main package can be imported."""
    try:
        import cross_asset_alpha_engine
        assert hasattr(cross_asset_alpha_engine, '__version__')
        assert cross_asset_alpha_engine.__version__ == "0.1.0"
    except ImportError as e:
        pytest.fail(f"Failed to import main package: {e}")


def test_config_import():
    """Test that config module can be imported."""
    try:
        from cross_asset_alpha_engine import config
        assert hasattr(config, 'POLYGON_API_KEY')
        assert hasattr(config, 'DEFAULT_CACHE_DIR')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")


def test_data_layer_imports():
    """Test that data layer components can be imported."""
    try:
        from cross_asset_alpha_engine.data import PolygonClient, DataCache, AssetUniverse
        from cross_asset_alpha_engine.data import load_daily_bars, load_intraday_bars
        
        # Test that classes can be instantiated (without API key for PolygonClient)
        cache = DataCache()
        universe = AssetUniverse()
        
        assert cache is not None
        assert universe is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import data layer: {e}")


def test_feature_engines_import():
    """Test that feature engines can be imported."""
    try:
        from cross_asset_alpha_engine.features import (
            IntradayFeatureEngine, 
            DailyFeatureEngine, 
            CrossAssetFeatureEngine
        )
        
        # Test that engines can be instantiated
        intraday_engine = IntradayFeatureEngine()
        daily_engine = DailyFeatureEngine()
        cross_asset_engine = CrossAssetFeatureEngine()
        
        assert intraday_engine is not None
        assert daily_engine is not None
        assert cross_asset_engine is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import feature engines: {e}")


def test_regime_detection_import():
    """Test that regime detection components can be imported."""
    try:
        from cross_asset_alpha_engine.regimes import RegimeHMM, RegimeFeatureEngine
        
        # Test that components can be instantiated
        regime_engine = RegimeFeatureEngine()
        hmm_model = RegimeHMM()
        
        assert regime_engine is not None
        assert hmm_model is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import regime detection: {e}")


def test_models_import():
    """Test that model components can be imported."""
    try:
        from cross_asset_alpha_engine.models import AlphaModel, ModelUtils
        
        # Test that components can be instantiated
        alpha_model = AlphaModel()
        model_utils = ModelUtils()
        
        assert alpha_model is not None
        assert model_utils is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import models: {e}")


def test_utils_import():
    """Test that utility modules can be imported."""
    try:
        from cross_asset_alpha_engine.utils import (
            setup_logger, 
            normalize_timezone,
            plot_equity_curve
        )
        
        # Test that functions are callable
        assert callable(setup_logger)
        assert callable(normalize_timezone)
        assert callable(plot_equity_curve)
        
    except ImportError as e:
        pytest.fail(f"Failed to import utils: {e}")


def test_optional_dependencies():
    """Test that optional dependencies are available."""
    optional_deps = [
        'numpy',
        'pandas', 
        'requests',
        'sklearn',  # scikit-learn imports as sklearn
        'hmmlearn',
        'matplotlib',
        'plotly'
    ]
    
    missing_deps = []
    
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        pytest.fail(f"Missing optional dependencies: {missing_deps}")


def test_package_structure():
    """Test that package has expected structure."""
    try:
        import cross_asset_alpha_engine
        
        # Check main subpackages exist
        subpackages = ['data', 'features', 'regimes', 'models', 'utils']
        
        for subpackage in subpackages:
            try:
                __import__(f'cross_asset_alpha_engine.{subpackage}')
            except ImportError as e:
                pytest.fail(f"Subpackage {subpackage} not found: {e}")
                
    except ImportError as e:
        pytest.fail(f"Package structure test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_main_package_import()
    test_config_import()
    test_data_layer_imports()
    test_feature_engines_import()
    test_regime_detection_import()
    test_models_import()
    test_utils_import()
    test_optional_dependencies()
    test_package_structure()
    
    print("All import tests passed! ✅")
