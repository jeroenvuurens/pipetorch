# PipeTorch Data module

The most common way to prepare data is to use Pandas DataFrame, however, data preparation and visualization can be very repetitive. Therefore, we designed an extension to a Pandas DataFrame (called a DFrame), that adds a quick way to do the most common data preprocessing, preparation and visualization.

The additional functions are divided in:
- [Data loading](#Data-Loading)
- [Out-of-sample validation](#Out-of-sample-validation): [split()](#split()) and [folds()](#folds())
- [Preprocessing](#Data-preprocessing): [scale()](#Scale()), [balance()](#Balance()), [polynomials()](#polynomials()), [category()](#catorgy()], [dummies()](#dummies())
- [Visualization](#Visualization): e.g. `df.train.scatter`
- [Data preparation](#Data-preparation): through the `.train_X`, ... , `.valid_y` properties, and `to_datasets()`, `to_dataloader()` and `to_databunch()` methods.

Two important things about the way the PipeTorch data pipeline works:
- all operations are `lazily executed`; e.g. scaling is not done until the data preparation is called. A dataframe `df` therefore still shows the original data while `df.train_X` shows the result after splitting and scaling.
- therefore, a call to any data preparation should come last, the order of the other functions is irrevelant.
- all PipeTorch operations are `non-destructive`, i.e. calling a function on a DataFrame `df` will not alter `df` but return a new version that is configured accordingly. There is one exception, when any data preparation function is called, the exact data split is stored to allow subsequent actions to consistently work with the same data.

We will provide some examples below, for more explanations and advanced options you can check the docstring for these functions (e.g. ?df.split).


```python
from pipetorch.data import read_from_kaggle, read_csv, create_kaggle_authentication
import numpy as np
```

    using gpu 3


# Data Loading

The data pipeline often starts by loading a dataset. The most basic way is to use [read_csv](#read_csv) to . Kaggle is a great resource for these, therefore, there is also a [read_from_kaggle](#read_from_kaggle) function to download a dataset directly from Kaggle.

### read_csv()

Uses pd.read_csv to read a csv from file or url. The difference with Pandas is that a DFrame is returned and it allows downloaded files to be automatically stored with `save=True` in `path / filename`, so that when read_csv is called with the same parameters the stored file is used. The `kwargs` are passed to pd.read_csv. When a (multichar) delimiter is used, this is also used to save the file so that the original delimiter is kept.

*Args:*
```    
url: str
    the url to download or a full path pointing to a .csv file
filename: str (None)
    the filename to store the downloaded file under. If None, the filename is extracted from the url.
path: str (None)
    the path in which the file is stored. If None, it will first check the ~/.pipetorch (for sharing
    dataset between users) and then ~/.pipetorchuser (for user specific caching of datasets).
save: bool (True)
    whether to save a downloaded .csv
**kwargs:
    additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
    you will have to set engine='python'.
```

*Returns*: DFrame


```python
wine = read_csv('https://osf.io/8fwaj/download', save=False)
wine
```

    Downloading https://osf.io/8fwaj/download





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quality</th>
      <th>pH</th>
      <th>volatile acidity</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>3.51</td>
      <td>0.700</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>3.20</td>
      <td>0.880</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>3.26</td>
      <td>0.760</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3.16</td>
      <td>0.280</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3.51</td>
      <td>0.700</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1594</th>
      <td>5</td>
      <td>3.45</td>
      <td>0.600</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>6</td>
      <td>3.52</td>
      <td>0.550</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>6</td>
      <td>3.42</td>
      <td>0.510</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>1597</th>
      <td>5</td>
      <td>3.57</td>
      <td>0.645</td>
      <td>10.2</td>
    </tr>
    <tr>
      <th>1598</th>
      <td>6</td>
      <td>3.39</td>
      <td>0.310</td>
      <td>11.0</td>
    </tr>
  </tbody>
</table>
<p>1599 rows × 4 columns</p>
</div>



### read_from_kaggle()

Reads a DFrame from a Kaggle dataset. The downloaded dataset is automatically stored so that the next time it is read from file rather than downloaded. See `read_csv`. The dataset is stored by default in a folder with the dataset name in `~/.pipetorchuser`.

If the dataset is not cached, this functions requires a valid .kaggle/kaggle.json file, that you can create manually or with the function `create_kaggle_authentication()`.

Note: there is a difference between a Kaggle dataset and a Kaggle competition. For the latter, you have to use `read_from_kaggle_competition`.
    
```
Example:
    read_from_kaggle('uciml/autompg-dataset')
        to read/download `https://www.kaggle.com/datasets/uciml/autompg-dataset`
    read_from_kaggle('robmarkcole/occupancy-detection-data-set-uci', 'datatraining.txt', 'datatest.txt')
        to combine a train and test set in a single DFrame
```
    
```
Args:
    dataset: str
        the username/dataset part of the kaggle url, e.g. uciml/autompg-dataset     
    train: str (None)
        the filename that is used as the train set, e.g. 'train.csv'
    test: str (None)
        the filename that is used as the test set, e.g. 'test.csv'
    shared: bool (False)
        save the dataset in ~/.pipetorch instead of ~/.pipetorchuser, allowing to share downloaded
        files between users.
    force: bool (False)
        when True, the dataset is always downloaded
    **kwargs:
        additional parameters passed to pd.read_csv. For example, when a multichar delimiter is used
        you will have to set engine='python'.

Returns: DFrame
```
            



```python
# if you want to be able to download yourself, register (free), create and register a token.
# create_kaggle_authentication('username', 'tokenstring')
```


```python
mpg = read_from_kaggle('uciml/autompg-dataset') 
```

### train/test sets

It is also possible to combine separate train and test files through `read_from_kaggle(dataset, train, test)` or combine train/test DataFrame through `DFrame.from_train_test()`. The data is then combined in a single DataFrame, but the test data will never leak into training. Any PipeTorch function will only use the train part, thus `split()`, `scale()`, `dummies`, `category`, etc. will only apply/fit on the train part. When data preparation is called, the test data (accessible through `.test`) will be transformed exactly like the train set. This allows you to configure the data processing pipeline for the entire set.

# DataFrame

The returned object is an extension of a Pandas DataFrame (called a DFrame). This means you can use Pandas for cleaning the data. One convenient function we added is `inspect()` to get an overview of the data.


```python
mpg.drop(columns='cylinders').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mpg</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>model year</th>
      <th>origin</th>
      <th>car name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>307.0</td>
      <td>130</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>350.0</td>
      <td>165</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>318.0</td>
      <td>150</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>304.0</td>
      <td>150</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>302.0</td>
      <td>140</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>




```python
mpg.inspect()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing (#)</th>
      <th>Missing (%)</th>
      <th>Datatype</th>
      <th>Range</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mpg</th>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
      <td>[9.0, 46.6]</td>
      <td>(13.0, 14.0, ...)</td>
    </tr>
    <tr>
      <th>cylinders</th>
      <td>0</td>
      <td>0.0</td>
      <td>int64</td>
      <td>[3, 8]</td>
      <td>(4, 8, ...)</td>
    </tr>
    <tr>
      <th>displacement</th>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
      <td>[68.0, 455.0]</td>
      <td>(97.0, 98.0, ...)</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>0</td>
      <td>0.0</td>
      <td>object</td>
      <td>#94</td>
      <td>(150, 90, ...)</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>0</td>
      <td>0.0</td>
      <td>int64</td>
      <td>[1613, 5140]</td>
      <td>(1985, 2130, ...)</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>0</td>
      <td>0.0</td>
      <td>float64</td>
      <td>[8.0, 24.8]</td>
      <td>(14.5, 15.5, ...)</td>
    </tr>
    <tr>
      <th>model year</th>
      <td>0</td>
      <td>0.0</td>
      <td>int64</td>
      <td>[70, 82]</td>
      <td>(73, 78, ...)</td>
    </tr>
    <tr>
      <th>origin</th>
      <td>0</td>
      <td>0.0</td>
      <td>int64</td>
      <td>[1, 3]</td>
      <td>(1, 3, ...)</td>
    </tr>
    <tr>
      <th>car name</th>
      <td>0</td>
      <td>0.0</td>
      <td>object</td>
      <td>#305</td>
      <td>(ford pinto, toyota corolla, ...)</td>
    </tr>
  </tbody>
</table>
</div>



# Out-of-sample validation

For Machine Learning, you have to use out-of-sample validation and test sets. PipeTorch provides two easy functions to do that:

### folds()

Divide the data in folds to setup n-Fold Cross Validation in a reproducible manner. 
        
By combining folds() with split(0 < test_size < 1) , a single testset is split before 
dividing the remainder in folds that are used for training and validation. 
When used without split, by default a single fold is used for testing.

The folds assigned to the validation and test-set rotate differently, 
giving 5x4 combinations for 5-fold cross validation. You can apply exhaustive cross-validation
over all 20 combinations by calling fold(0) through fold(19), or less exhaustive cross-validation 
by calling fold(0) through fold(4) to use every fold for validation and testing once. 

```
Example:
    df.folds(5)
        creates 5 equally sized folds. Calls to df.fold(i) will set the train/valid/test sets
        to one of the permutations over the folds. The sets are generated when data preparation
        is called, e.g. df.train, df.valid or df.test.

Arguments:
    folds: int (None)
        The number of times the data will be split in preparation for n-fold cross validation. The
        different splits can be used through the fold(n) method.
        SKLearn's SplitShuffle is used, therefore no guarantee is given that the splits are
        different nor that the validation splits are disjoint. For large datasets, that should not
        be a problem.
    shuffle: bool (None)
        shuffle the rows before splitting. None means True unless sequence() is called to process
        the data as a (time) series.
    random_state: int (None)
        set a random_state for reproducible results
    stratify: str or [ str ] (None)
        apply stratified sampling. Per value for the given column, the rows are sampled. When a list
        of columns is given, multi-label stratification is applied.
    test: bool (None)
        whether to use one fold as a test set. The default None is interpreted as True when
        split is not used. Often for automated n-fold cross validation studies, the validation set
        is used for early termination, and therefore you should use an out-of-sample
        test set that was not used for optimizing.

Returns: copy of DFrame 
    schedules the data to be split in folds.
```


```python
wine = wine.folds(5)
print(wine.fold(0).valid.head())
print(wine.fold(1).valid.head())  # fold(0) - fold(4) all have unique validation examples
```

        quality    pH  volatile acidity  alcohol
    9         5  3.35              0.50     10.5
    11        5  3.35              0.50     10.5
    19        6  3.04              0.32      9.2
    24        6  3.43              0.40      9.7
    28        5  3.47              0.71      9.4
       quality    pH  volatile acidity  alcohol
    1        5  3.20              0.88      9.8
    2        5  3.26              0.76      9.8
    3        6  3.16              0.28      9.8
    4        5  3.51              0.70      9.4
    8        7  3.36              0.58      9.5


### split() 

Split the data in a train/valid/(test) set. If the DFrame was loaded with a separate test part, then
split only applies to the train part.

```
Example:
    df.split(0.2, stratify='city')
        splits the train data in a 80%/20% train/valid part. Stratify attempts to populate both parts 
        with the same proportion over each value of the column city
    df.split(0.2, 0.2)
        splits the train data in a 60%/20%/20% train/valid/test split

Arguments:
    valid_size: float (None)
        the fraction of the dataset that is used for the validation set.
    test_size: float (None)
        the fraction of the dataset that is used for the test set. When combined with folds
        if 1 > test_size > 0, the test set is split before the remainder is divided in folds 
        to apply n-fold cross validation.
    shuffle: bool (None)
        shuffle the rows before splitting. None means True unless sequence() is called to process
        the data as a (time) series.
    random_state: int (None)
        set a random_state for reproducible results
    stratify: str or [ str ] (None)
        apply stratified sampling. Per value for the given column, the rows are sampled. When a list
        of columns is given, multi-label stratification is applied.

Returns: copy of DFrame 
    schedules the rows to be split into a train, valid and (optionally) test set.
```


```python
wine = wine.split(0.2)
len(wine.train), len(wine.valid)
```




    (1279, 320)



# Data selection

By default, PipeTorch assumes that the last column is the target variable. All columns except the target columns will be the input features. You can use `columny()` and `columnx` to change this default bahavior. For filtering, you can just use Pandas.


```python
wine = wine.columny('quality')
```

# Data preprocessing

### Scale()

Configures feature scaling for the features and target variable in the DataFrame. This effect is not visible in the DataFrame, but applied when data preparation is called. Then scaler is fitted 
on the training set and used to transform the train, valid and test set. 

There is also a `scalex` to scale only the input features.

```
Arguments:
    columns: True, str or list of str (True)
        the columns to scale (True for all)
    scalertype: an SKLearn type scaler (StandardScaler)
    omit_interval: (-2,2) when colums is set to True
        all columns whose values lie outside the omit_interval,

Return: copy of DFrame 
```


```python
wine.scale().head() # scaling is configured, yet not visible in the DataFrame
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>quality</th>
      <th>pH</th>
      <th>volatile acidity</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>3.51</td>
      <td>0.70</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>3.20</td>
      <td>0.88</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>3.26</td>
      <td>0.76</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>3.16</td>
      <td>0.28</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3.51</td>
      <td>0.70</td>
      <td>9.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
wine.scalex().train_X # but it will be when you prepare the data
```




    array([[ 1.29927255,  0.7       , -0.95754073],
           [-0.71712237,  0.88      , -0.57978176],
           [-0.32685238,  0.76      , -0.57978176],
           ...,
           [ 0.71386757,  0.51      ,  0.55349517],
           [ 1.68954254,  0.645     , -0.20202278],
           [ 0.51873258,  0.31      ,  0.55349517]])



### balance()

Oversamples rows in the training set, so that the values of the target variable are better balanced. Only affects the train set. This effect is not visible in the DataFrame, but applied when data preparation is called.

```
Arguments:
    weights: True or dict
        when set to True, the target values of the training set are 
        uniformely distributed,
        otherwise a dictionary can be passed that map target values to the 
        desired fraction of the training set (e.g. {0:0.4, 1:0.6}).

Returns: copy of DFrame
```


```python
wine = wine.balance()
```

The original DFrame is not affected


```python
wine.groupby(by='quality').quality.count()
```




    quality
    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64



But the generated training split is. Roughly 80% was used for training, and rows where duplicated so that the minority classes match the rowcount. Alternatively, you can also supply a weight distribution for the classes to balance().


```python
wine.train.groupby(by='quality').quality.count()
```




    quality
    3    537
    4    537
    5    537
    6    537
    7    537
    8    537
    Name: quality, dtype: int64



### polynomials()

Adds (higher-order) polynomials to the data pipeline. This effect is not visible in the DataFrame, but applied when data preparation is called.

Note: the generated columns are nameless, therefore you cannot combine polynomials with column
specific scaling (only None, all, or only input features).
```
Example:
    df.polynomials(degree=2)
        Transforms X to include 2nd order polynomials, see SKLearn PolynomialFeatures

Args:
    degree: int
        degree of the higher-order polynomials 
    include_bias: bool (False)
        whether to generate a bias column

Returns: copy of DFrame 
```


```python
wine.columnx('pH').polynomials(degree=2).train_X
```




    array([[ 3.25  , 10.5625],
           [ 3.16  ,  9.9856],
           [ 3.38  , 11.4244],
           ...,
           [ 3.15  ,  9.9225],
           [ 3.23  , 10.4329],
           [ 3.15  ,  9.9225]])



### category()


