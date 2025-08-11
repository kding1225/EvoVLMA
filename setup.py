from setuptools import setup, find_packages

setup(
    name="eoh",
    version="0.1",
    author="Kun Ding",
    author_email="kun.ding@ia.ac.cn",
    description="EvoVLMA: Evolutionary Vision-Language Model Adaptation",
    packages=find_packages(include=['eoh', 'eoh.*']),
    package_dir={"": "."},
    install_requires=[
        "numpy",
        "numba",
        "gunicorn",
        "joblib",
        "optunahub",
        "optuna",
        "flask",
        "torch",
        "pyyaml",
        "torchvision",
        "scikit-learn",
        "gdown",
        "ftfy",
        "regex",
        "openai"
    ],
    python_requires="==3.9.0",
    include_package_data=True,
)