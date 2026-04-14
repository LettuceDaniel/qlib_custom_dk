from setuptools import setup

setup(
    name="qlib_research",
    version="0.1.0",
    description="qlib-based model training and backtesting pipeline",
    author="DK",
    packages=[
        "qlib_research",
        "qlib_research.train_valid_backtest",
        "qlib_research.train_valid_backtest.data",
        "qlib_research.train_valid_backtest.model",
        "qlib_research.train_valid_backtest.backtest",
        "qlib_research.train_valid_backtest.evaluation",
        "qlib_research.train_valid_backtest.workflow",
        "qlib_research.config",
        "qlib_research.qlib",
    ],
    package_dir={"qlib_research": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "h5py>=3.0.0",
        "pyyaml>=6.0",
        "toml>=0.10.0",
    ],
)
