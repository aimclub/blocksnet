from setuptools import setup, find_packages

setup(
    name="blocksnet",
    version="0.1.0",
    description="A library for urban service optimization.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torch-geometric",
        "pymoo",
        "pandas",
        "numpy",
        "scipy"
    ],
    include_package_data=True,
)