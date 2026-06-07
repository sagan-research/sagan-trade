from setuptools import setup, find_packages

setup(
    name='sagan-trade',
    version='0.1.1',
    description='Sagan High Frequency Trading Engine',
    py_modules=['backtester', 'infinite_trading_daemon', 'ipc_parameter_writer', 'moe_model', 'run', 'sagan_combinatorial_generator', 'simulator', 'symbolic'],
    install_requires=[
        'torch',
        'numpy',
        'pandas'
    ],
    python_requires='>=3.8',
)
