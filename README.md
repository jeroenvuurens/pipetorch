# PipeTorch

This library provides a toolset for PyTorch and SKLearn projects, to efficiently setup data, use a general purpose trainer, tune hyperparameters, validate the model and evaluate its performance.

This library was initially created for educational purposes at The Hague University of Applied Sciences. Especially, for teaching students how to optimize Neural Networks, the learning curve may be quite steep. Therefore, we adopted the 'Fast.AI philosophy' of creating a toolset to code experiments in four simple steps: Data, Model, Train, Evaluate. We start our teachings to-down, using just one or two lines of code for each step to demonstrate the idea behind Predictive Analytics, Neural Networks and how to avoid problems like non-convergence and overfitting. In phases, our students have to code a single step themselves until they master the whole spectrum and code everything in PyTorch. We created PipeTorch in a way that you can decide for yourself which steps you would like to code yourself and which steps you would like to use the PipeTorch tools.

Disclaimer: the support that PipeTorch will provide is limited. We do not intent this library to be better than Fast.AI or TorchLightning. Our focus is more on a transparent toolkit that provides easy access to some state-of-the-art algorithms, most of which are already built into PyTorch. 

Disclaimer2: although we very much liked and recommend the Fast.AI lectures, we found that the library itself heavily relied on the latest packages and broke frequently. For our intents, that is not workable. We also aim to use 'standard' libraries like Pandas as much as possible and extend them instead of replace them. Therefore, we decided to put together a simple framework. We actually found that setting it up in a way that it never breaks is a delusion, the code does break sometimes because the libraries we use make backwards incompatible updates. Regardless, in its current state it is stable enough to work with.

PipeTorch provides four separate toolsets, that work together, or can be used as a separate tool

# Data

The core is an extension to a Pandas DataFrame, that you can manipulate like any normal DataFrame. Additionally, our DataFrame has methods that simplify data preparation, like split(), category(), scale(), balance(), columny(), polynomials(), dummies(), sequence(). All configurations are lazily executed and you will therefore work with the original data until you require processing like train_X, valid_y, to_dataset() or to_databunch. You can also view the train, valid and test DataSets that are created by split and use several methods to visualize the data. Eventually, the predictions from a model can be added back to the original DataFrame and transformations like scaling are inverted to allow for a comprehensible evaluation.

The DataFrame facilitates transformation of the data to Numpy and PyTorch using the exact same pipeline, to allow a fair comparison of SKLearn and PyTorch models.

The DataFrames also support learning on sequences such as time-series that are generated using a sliding window (per group).

Since images and text are not easily represented in DataFrames, there are two custom classes that will load these from files, using a similar interface to the one we have for DataFrames.

# Model

There is some support for creating simple Perceptron's, CNN's and transfer learning.

# Trainer

Perhaps the best part. The general purpose trainer will train any model, with any (custom) loss function, on two dataloaders (or a databunch). It current only works with the AdamW optimizer, however, it is to some extend configurable:
- collects a configurable list of (SKLearn) metrics during training
- there are three learning rate schedules: uniform, cyclic and decaying
- can easily add weight_decay for regularization

Additionally:
- can easily switch between gpu and cpu
- there is a learning rate finder
- can commit, checkout, revert, and lowest to switch to earlier versions of the model, making early termination easier
- save and load models to/from file
- simplifies diagnostics through a learning_curve and validation_curve
- provides some support for transfer learning

# Evaluator

This is mostly used by the trainer to store the training results in a DataFrame, with additional support to visualize data and compare models. 

