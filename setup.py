import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='pipetorch',  
     version='0.1',
     #packages=['jtorch', 'jtorch.tabular', 'jtorch.image']
     author="Jeroen Vuurens",
     author_email="jbpvuurens@gmail.com",
     description="A data pipeline library for PyTorch and Machine Learning projects",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/jeroenvuurens/pipetorch",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

