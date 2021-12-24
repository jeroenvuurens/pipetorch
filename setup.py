from setuptools import setup, find_packages
from pipetorch import __version__

setup(
     name='pipetorch',
     packages=find_packages(),
     #packages=['pipetorch', 'pipetorch.data'],
     version=__version__,
     author="Jeroen Vuurens",
     author_email="jbpvuurens@gmail.com",
     description="A data pipeline library for PyTorch and Machine Learning projects",
     url="https://github.com/jeroenvuurens/pipetorch",
     download_url="https://github.com/jeroenvuurens/pipetorch",
     keywords=['PyTorch', 'SKLearn', 'Machine Learning', 'Neural Network', 'Predictive Analytics'],
     install_requires=['torch', 'torchvision', 'numpy', 'sklearn', 'matplotlib', 'pandas', 'pathlib', 'tqdm', 'statistics'],
     classifiers=[
         "Development Status :: 3 - Alpha",
         "Intended Audience :: Education",
         "Intended Audience :: Science/Research",
         "Topic :: Scientif/Enginering :: Artificial Intelligence",
         "Topic :: Software Development :: Libraries :: Python Modules",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

