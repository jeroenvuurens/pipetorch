from setuptools import setup, find_packages
from pipetorch import __version__

assert 'a' in __version__, "should be a alpha version"

alpha = [
         "Development Status :: 3 - Alpha",
         "Intended Audience :: Education",
         "Intended Audience :: Science/Research",
         "Topic :: Scientific/Engineering :: Artificial Intelligence",
         "Topic :: Software Development :: Libraries :: Python Modules",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]

setup(
     name='pipetorch',
     packages=find_packages(),
     #packages=['pipetorch', 'pipetorch.data'],
     package_data={'pipetorch': ['data/datasets/*']},
     include_package_data=True,
     version=__version__,
     author="Jeroen Vuurens",
     author_email="jbpvuurens@gmail.com",
     description="A data pipeline library for PyTorch and Machine Learning projects",
     url="https://github.com/jeroenvuurens/pipetorch",
     download_url="https://github.com/jeroenvuurens/pipetorch/archive/refs/tags/v0.1a.tar.gz",
     keywords=['PyTorch', 'SKLearn', 'Machine Learning', 'Neural Network', 'Predictive Analytics'],
     install_requires=['torch', 'torchvision', 'numpy', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm', 'iterative-stratification', 'optuna', 'seaborn', 'kaggle'],
     classifiers=alpha
 )

