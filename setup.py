from setuptools import setup, find_packages

setup(
    name="rl-production-optimizer",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=2.3.0",
        "numpy>=1.24.0",
    ],
    author="Ing. Kelvin Cabrera",
    description="Reinforcement Learning system for optimizing oil production",
    python_requires=">=3.8",
)
