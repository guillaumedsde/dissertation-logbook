---
title: "Grid Search for finding optimal parameters"
date: 2020-02-03T14:04:04Z
draft: false
---

_note: this is a Markdown render of a Jupyter notebook whose [latest version is available here](https://gitlab.com/harpocrates-app/svm-grid-search)_

In this notebook, I'll go over my efforts to find the best parameters for my sensitive document classifier.

My [previous GridSearch](https://dissertation.guillaume.desusanne.com/posts/paragraphs/) of a small parameter space to try to quickly improve my model had a few problems:

- I only performed it over a small parameter space, and not even the most appropriate one
  - specifically I ran it mostly on my vectorizer's parameters, not the SVC classifier
- I only printed out best parameters, disregarding any output measure of accuracy and trusting the "default" ones from the GridSearch built in scikit learn.

So I need to do a couple of things:

- define a selection of classifiers to use
- build a parameter space to explore
- select which metrics to use
- export these metrics
- visualize them

## Initialization

I've included a Pipefile in this repository, I suggest you use it to create a virtualenv with all the dependencies.
Otherwise, you can run this cell to install the dependencies for your local user

```python
#!pip install numpy jupyter numpy scikit-learn beautifulsoup4 ipykernel lxml pandas
```

Lets initalize a couple of variables (you'll need to adjust the path to the dataset on your machine):

```python
from multiprocessing import cpu_count
from IPython.lib.display import YouTubeVideo


PROCESSES = cpu_count()
CLASS_NAMES = ["not sensitive", "sensitive"]
SEED = 1968

TRAIN_DATA_DIR = (
    "/mnt/data/architect/Documents/data/collection/cab_html_noboiler/"
)

TRAIN_LABELS = "/mnt/data/architect/Documents/data/full.collection.cables.path.gold"


```

## Extracting labels and data

Then we'll extract the raw HTML data from the files and parse it to extract the textual data and load them in a numpy array with the associated labels. Let's define the function for extracting

```python
from multiprocessing import Pool

import numpy as np
from bs4 import BeautifulSoup


def extract_labels():
    print("extracting labels from {}".format(TRAIN_LABELS))
    # load classification data
    return np.loadtxt(TRAIN_LABELS, dtype=int, usecols=1)


def extract_file_paths():
    # load labels (file paths)
    file_paths = []
    print("extracting file paths from {}".format(TRAIN_LABELS))
    with open(TRAIN_LABELS) as ground_truth_file:
        for line in ground_truth_file.read().splitlines():
            # build full file path (path prefix and file extension)
            file_paths.append(TRAIN_DATA_DIR + line.split(" ")[0] + ".html")
    return file_paths


def read_file(file_path):
    with open(file_path) as file:
        raw_html = file.read()

        bs = BeautifulSoup(raw_html, "lxml")
        # get the contents of the last <pre> tags
        pre = bs.find_all("pre")[-1]

        text = pre.get_text()
    return text


def extract_data(file_paths):
    texts = []
    print(
        "extracting data from {n_files} files with {n_processes} processes".format(
            n_files=len(file_paths), n_processes=PROCESSES
        )
    )
    pool = Pool(processes=PROCESSES)
    for file_path in file_paths:
        pool.apply_async(
            func=read_file,
            args=(file_path,),
            callback=texts.append,
            error_callback=print,
        )
    pool.close()
    pool.join()
    return np.array(texts)

```

Now lets extract run this code to get two arrays:

- one containing a label for each document
- the other containing the textual representation of each document

```python
train_labels = extract_labels()
file_paths = extract_file_paths()
train_data = extract_data(file_paths)
```

    extracting labels from /mnt/data/architect/Documents/data/full.collection.cables.path.gold
    extracting file paths from /mnt/data/architect/Documents/data/full.collection.cables.path.gold
    extracting data from 3801 files with 12 processes

We've extracted all 3801 documents into an array of string representation of the document:

```python
print(train_data.shape)
print(train_labels.shape)
```

    (3801,)
    (3801,)

The first document is 44 lines long and is classified as "not sensitive"

```python
print(len(train_data[0].splitlines()))
print(train_labels[0])
```

    44
    0

## Defining a parameter space

We'll now create a document vectorizer and select a classifier in order to define a parameter space to explore all combinations of the different parameters of the

### TF-IDF Vectorizer

```python
tfidf_parameters = {
    "vect__norm": ("l1",),
    "vect__use_idf": (True,),
    "vect__smooth_idf": (False,),
    "vect__sublinear_tf": (True,),
    "vect__analyzer": ["word"],
    "vect__stop_words": ["english"],
    "vect__strip_accents": ["unicode"],
    "vect__lowercase": (True,),
    "vect__max_df": (0.9,),
    "vect__min_df": (1,)
#     "ngram_range":[ (1,1), (1,2), (1,3), (1,4)]
}
```

### Support Vector Machines (SVM)

```python
# MIT lecture (long) on SVM
YouTubeVideo('_PwhiWxHK8o')
```

<iframe
    width="400"
    height="300"
    src="https://www.youtube.com/embed/_PwhiWxHK8o"
    frameborder="0"
    allowfullscreen
></iframe>

```python
# Good Explanation of the influence of the gamma parameter:
YouTubeVideo('m2a2K4lprQw')
```

<iframe
    width="400"
    height="300"
    src="https://www.youtube.com/embed/m2a2K4lprQw"
    frameborder="0"
    allowfullscreen
></iframe>

```python
# Good Explanation of the influence of the C parameter:
YouTubeVideo('joTa_FeMZ2s')
```

<iframe
    width="400"
    height="300"
    src="https://www.youtube.com/embed/joTa_FeMZ2s"
    frameborder="0"
    allowfullscreen
></iframe>

```python
# OvO vs OvR
YouTubeVideo('93AjE1YY5II')
```

<iframe
    width="400"
    height="300"
    src="https://www.youtube.com/embed/93AjE1YY5II"
    frameborder="0"
    allowfullscreen
></iframe>

```python
# General explanation for a kernel
from datetime import timedelta
start=int(timedelta(hours=0, minutes=4, seconds=4).total_seconds())
YouTubeVideo("Y6RRHw9uN9o", start=start)
```

<iframe
    width="400"
    height="300"
    src="https://www.youtube.com/embed/Y6RRHw9uN9o?start=244"
    frameborder="0"
    allowfullscreen
></iframe>

```python
svc_parameters = {
    # set fixed random state of comparison
    # does not matter as long as it is consistent
    "clf__random_state": [1984],
    # tradeoff between classifying training points correctly
    # and regular decision boundary
    "clf__C": [0.01, 0.1, 1, 10, 100, 1000],
    # low gamma => points away from hyperplane matter more
    # high gamma => points close to hyperplane matter more
#     "clf__gamma": [0.001, 0.01, 0.1, 1],
    # ovo: One vs One binary classification of two classes
    # ovr: One vs Rest One class vs "the Rest" classification
    "clf__decision_function_shape": ("ovo", "ovr"),
    # Kernel transforms data to higher dimensional space
    # for when data is not linearly seperable in current dimension
#     "clf__kernel": ("linear", "poly", "rbf", "sigmoid"),
    "clf__kernel": ["linear"],
    # Cache in MB, adjust according to available RAM
    "clf__cache_size": [100],
    "clf__probability": [True],
    # only for ThunderSVM
#     "clf__max_mem_size": [100]
}
```

## Grid Search

### Parameter space

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
from thundersvm import SVC


# Create the Pipeline
pipeline = Pipeline(
        steps=[
            ("vect", TfidfVectorizer()),
            ("clf", SVC()),
        ],
        verbose=False,
    )
```

```python
from pprint import pprint

# Unify parameters dictionary
parameters = {}
parameters.update(svc_parameters)
parameters.update(tfidf_parameters)

pprint(parameters)


test_parameters = {}
for k,v in parameters.items():
    test_parameters[k]= [v[0]]
pprint(test_parameters)
```

    {'clf__C': [0.01, 0.1, 1, 10, 100, 1000],
     'clf__cache_size': [100],
     'clf__decision_function_shape': ('ovo', 'ovr'),
     'clf__kernel': ['linear'],
     'clf__probability': [True],
     'clf__random_state': [1984],
     'vect__analyzer': ['word'],
     'vect__lowercase': (True, False),
     'vect__max_df': (0.9, 0.95, 1),
     'vect__min_df': (0.1, 0.05, 1),
     'vect__norm': ('l1', 'l2'),
     'vect__smooth_idf': (True, False),
     'vect__stop_words': ['english'],
     'vect__strip_accents': ['unicode'],
     'vect__sublinear_tf': (True, False),
     'vect__use_idf': (True, False)}
    {'clf__C': [0.01],
     'clf__cache_size': [100],
     'clf__decision_function_shape': ['ovo'],
     'clf__kernel': ['linear'],
     'clf__probability': [True],
     'clf__random_state': [1984],
     'vect__analyzer': ['word'],
     'vect__lowercase': [True],
     'vect__max_df': [0.9],
     'vect__min_df': [0.1],
     'vect__norm': ['l1'],
     'vect__smooth_idf': [True],
     'vect__stop_words': ['english'],
     'vect__strip_accents': ['unicode'],
     'vect__sublinear_tf': [True],
     'vect__use_idf': [True]}

### Running GridSearch

Now we're going to define the GridSearch, passing it our pipeline and the parameter space. We're going to tell it to use as many processes as there are CPU cores on the current machine (12 in my case).

#### RAM

However, running parallel fits implies holding many copies of the dataset in memory. Thus with my 8GB of RAM it is not feasable to hold 12 copies (one for each process) in RAM, this is defined with the `pre_dispatch` parameter of the `GridSearch`.

Quick testing shows that disk swapping is too slow beyond two copies of the dataset held in RAM at a time. I peformed this test with the test with the `test_parameters` space defined above, a small GridSearch mainly for measuring performance to find the optimal `pre_dispatch` parameter .

`clf__cache_size` is the size of the kernel cache in MB but I've not played around with it too much after I saw that it increase the total time.

| pre_dispatch | clf\_\_cache_size | elapsed time (s) |
| ------------ | ----------------- | ---------------- |
| 1            | 100               | 120.113          |
| 2            | 100               | 87.814           |
| 3            | 100               | 126.990          |
| 1            | 300               | 140.500          |
| 2            | 300               | 101.858          |

This works, however, lots of my 12 cores are now sitting idle because I can only load 3 copies of the dataset in RAM. Another idea came to my mind however: using [ZRAM](https://wiki.archlinux.org/index.php/Improving_performance#Zram_or_zswap), i.e. compressing RAM, essentially using extra CPU power to "grow" my RAM space. Using a tool like [zramswap](https://github.com/highvoltage/zram-tools). I created what is essentially a zstd compressed SWAP file of 75% of my 8GB loaded in RAM which, combined with my SWAP space, gives me 22 GB of "RAM" allowing me to load 5 copies of the dataset in RAM and use the remaining cores for compressing RAM.

Technically I could also [SWAP to the Video RAM](https://wiki.archlinux.org/index.php/Swap_on_video_RAM) of my GPU for an additional 3GB, but its slightly more complicated to configure.

#### Parallel SVM

Another trick to improve performance is to use parallel implementations of SVM. Specifically [ThunderSVM](https://github.com/Xtra-Computing/thundersvm) is capable of utilizing multiple CPU cores or even a GPU to perform the computations of the SVM classifier.

Although your CPU will probably be fully utilized either compressing RAM (like me) or testing a parameter combination, if you have a CUDA enabled GPU (= Nvidia GPU) you can leverage it to perform the GridSearch faster.

```python
from time import time

from sklearn.model_selection import GridSearchCV

# find the best parameters for both the feature extraction and the
# classifier

metrics = ["accuracy", "balanced_accuracy", "average_precision", "neg_brier_score", "f1", "f1_micro", "f1_macro", "f1_weighted", "neg_log_loss", "precision", "recall", "jaccard", "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"]

grid_search = GridSearchCV(pipeline,
                           # parameters dictionary
                           parameters,
                           # list of metrics
                           scoring=metrics,
                           # use all available CPUs
                           n_jobs=2,
                           # numbers of copies of dataset to keep in RAM
                           pre_dispatch=2,
                           # "Refit an estimator using the best
                           # found parameters on the whole dataset"
                           # refit optimizing for recall
                           refit="recall",
                           verbose=10)
```

And now lets run the Grid Search (this takes a while)

```python
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(train_data, train_labels)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

```

    Performing grid search...
    pipeline: ['vect', 'clf']
    parameters:
    {'clf__C': [0.01, 0.1, 1, 10, 100, 1000],
     'clf__cache_size': [100],
     'clf__decision_function_shape': ('ovo', 'ovr'),
     'clf__kernel': ['linear'],
     'clf__probability': [True],
     'clf__random_state': [1984],
     'vect__analyzer': ['word'],
     'vect__lowercase': (True, False),
     'vect__norm': ('l1', 'l2'),
     'vect__smooth_idf': (True, False),
     'vect__stop_words': ['english'],
     'vect__strip_accents': ['unicode'],
     'vect__sublinear_tf': (True, False),
     'vect__use_idf': (True, False)}
    Fitting 5 folds for each of 384 candidates, totalling 1920 fits


    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done   3 tasks      | elapsed:   50.8s
    [Parallel(n_jobs=2)]: Done   6 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=2)]: Done  11 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=2)]: Done  16 tasks      | elapsed:  2.9min
    [Parallel(n_jobs=2)]: Done  23 tasks      | elapsed:  4.1min
    [Parallel(n_jobs=2)]: Done  30 tasks      | elapsed:  5.0min
    [Parallel(n_jobs=2)]: Done  39 tasks      | elapsed:  6.5min
    [Parallel(n_jobs=2)]: Done  48 tasks      | elapsed:  8.0min
    [Parallel(n_jobs=2)]: Done  59 tasks      | elapsed: 10.3min
    [Parallel(n_jobs=2)]: Done  70 tasks      | elapsed: 12.2min
    [Parallel(n_jobs=2)]: Done  83 tasks      | elapsed: 14.8min
    [Parallel(n_jobs=2)]: Done  96 tasks      | elapsed: 16.7min
    [Parallel(n_jobs=2)]: Done 111 tasks      | elapsed: 19.3min
    [Parallel(n_jobs=2)]: Done 126 tasks      | elapsed: 21.9min
    [Parallel(n_jobs=2)]: Done 143 tasks      | elapsed: 25.9min
    [Parallel(n_jobs=2)]: Done 160 tasks      | elapsed: 29.4min
    [Parallel(n_jobs=2)]: Done 179 tasks      | elapsed: 32.4min
    [Parallel(n_jobs=2)]: Done 198 tasks      | elapsed: 35.0min
    [Parallel(n_jobs=2)]: Done 219 tasks      | elapsed: 39.2min
    [Parallel(n_jobs=2)]: Done 240 tasks      | elapsed: 43.0min
    [Parallel(n_jobs=2)]: Done 263 tasks      | elapsed: 46.9min
    [Parallel(n_jobs=2)]: Done 286 tasks      | elapsed: 50.8min
    [Parallel(n_jobs=2)]: Done 311 tasks      | elapsed: 56.6min
    [Parallel(n_jobs=2)]: Done 336 tasks      | elapsed: 61.0min
    [Parallel(n_jobs=2)]: Done 363 tasks      | elapsed: 66.0min
    [Parallel(n_jobs=2)]: Done 390 tasks      | elapsed: 71.3min
    [Parallel(n_jobs=2)]: Done 419 tasks      | elapsed: 77.1min
    [Parallel(n_jobs=2)]: Done 448 tasks      | elapsed: 83.1min
    [Parallel(n_jobs=2)]: Done 479 tasks      | elapsed: 91.3min
    [Parallel(n_jobs=2)]: Done 510 tasks      | elapsed: 96.4min
    [Parallel(n_jobs=2)]: Done 543 tasks      | elapsed: 103.2min
    [Parallel(n_jobs=2)]: Done 576 tasks      | elapsed: 109.7min
    [Parallel(n_jobs=2)]: Done 611 tasks      | elapsed: 117.3min
    [Parallel(n_jobs=2)]: Done 646 tasks      | elapsed: 125.9min
    [Parallel(n_jobs=2)]: Done 683 tasks      | elapsed: 133.8min
    [Parallel(n_jobs=2)]: Done 720 tasks      | elapsed: 142.1min
    [Parallel(n_jobs=2)]: Done 759 tasks      | elapsed: 151.3min
    [Parallel(n_jobs=2)]: Done 798 tasks      | elapsed: 162.1min
    [Parallel(n_jobs=2)]: Done 839 tasks      | elapsed: 170.4min
    [Parallel(n_jobs=2)]: Done 880 tasks      | elapsed: 179.8min
    [Parallel(n_jobs=2)]: Done 923 tasks      | elapsed: 190.1min
    [Parallel(n_jobs=2)]: Done 966 tasks      | elapsed: 201.8min
    [Parallel(n_jobs=2)]: Done 1011 tasks      | elapsed: 212.5min
    [Parallel(n_jobs=2)]: Done 1056 tasks      | elapsed: 225.0min
    [Parallel(n_jobs=2)]: Done 1103 tasks      | elapsed: 239.2min
    [Parallel(n_jobs=2)]: Done 1150 tasks      | elapsed: 251.3min
    [Parallel(n_jobs=2)]: Done 1199 tasks      | elapsed: 265.2min
    [Parallel(n_jobs=2)]: Done 1248 tasks      | elapsed: 278.3min
    [Parallel(n_jobs=2)]: Done 1299 tasks      | elapsed: 293.6min
    [Parallel(n_jobs=2)]: Done 1350 tasks      | elapsed: 307.0min
    [Parallel(n_jobs=2)]: Done 1403 tasks      | elapsed: 322.4min
    [Parallel(n_jobs=2)]: Done 1456 tasks      | elapsed: 338.3min
    [Parallel(n_jobs=2)]: Done 1511 tasks      | elapsed: 353.5min
    [Parallel(n_jobs=2)]: Done 1566 tasks      | elapsed: 368.8min
    [Parallel(n_jobs=2)]: Done 1623 tasks      | elapsed: 387.3min
    [Parallel(n_jobs=2)]: Done 1680 tasks      | elapsed: 403.9min
    [Parallel(n_jobs=2)]: Done 1739 tasks      | elapsed: 424.3min
    [Parallel(n_jobs=2)]: Done 1798 tasks      | elapsed: 442.2min
    [Parallel(n_jobs=2)]: Done 1859 tasks      | elapsed: 461.1min
    [Parallel(n_jobs=2)]: Done 1920 out of 1920 | elapsed: 481.4min finished


    done in 28911.589s

    Best score: 0.210
    Best parameters set:
    	clf__C: 0.01
    	clf__cache_size: 100
    	clf__decision_function_shape: 'ovo'
    	clf__kernel: 'linear'
    	clf__probability: True
    	clf__random_state: 1984
    	vect__analyzer: 'word'
    	vect__lowercase: False
    	vect__norm: 'l1'
    	vect__smooth_idf: True
    	vect__stop_words: 'english'
    	vect__strip_accents: 'unicode'
    	vect__sublinear_tf: False
    	vect__use_idf: False

## Grid Search Results

Lastly, lets display the results and also save them as a CSV:

```python
from IPython.display import display, HTML
from pandas import DataFrame

# Convert results dictionary to dataframe
df = DataFrame.from_dict(grid_search.cv_results_)

# display dataframe as table
# display(HTML(df.to_html()))

# export results dataframe to CSV
df.to_csv('/mnt/btrfs/git_repositories/svm-grid-search/results/grid_search_results.csv', index=True)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-279395ff79ed> in <module>
          3
          4 # Convert results dictionary to dataframe
    ----> 5 df = DataFrame.from_dict(grid_search.cv_results_)
          6
          7 # display dataframe as table


    NameError: name 'grid_search' is not defined

### Simplified results

```python
import pandas
from IPython.display import display, HTML
from pandas import DataFrame

df = pandas.read_csv("./results/grid_search_stratified_results_simplified.csv", delimiter=";")


# display dataframe as table
display(HTML(df.to_html(index=False)))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>id</th>
      <th>roc_auc</th>
      <th>accuracy</th>
      <th>Precision</th>
      <th>f1</th>
      <th>recall</th>
      <th>settings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>180.0</td>
      <td>0.673658</td>
      <td>0.867930</td>
      <td>0.314556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>468.0</td>
      <td>0.673658</td>
      <td>0.867930</td>
      <td>0.314556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>176.0</td>
      <td>0.673310</td>
      <td>0.867930</td>
      <td>0.314278</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>464.0</td>
      <td>0.673310</td>
      <td>0.867930</td>
      <td>0.314278</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>224.0</td>
      <td>0.672614</td>
      <td>0.867930</td>
      <td>0.314804</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>512.0</td>
      <td>0.672614</td>
      <td>0.867930</td>
      <td>0.314804</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>228.0</td>
      <td>0.672493</td>
      <td>0.868982</td>
      <td>0.315317</td>
      <td>0.019306</td>
      <td>0.010000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>516.0</td>
      <td>0.672493</td>
      <td>0.868982</td>
      <td>0.315317</td>
      <td>0.019306</td>
      <td>0.010000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>209.0</td>
      <td>0.669304</td>
      <td>0.792403</td>
      <td>0.297952</td>
      <td>0.051908</td>
      <td>0.134653</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>213.0</td>
      <td>0.669304</td>
      <td>0.792403</td>
      <td>0.297952</td>
      <td>0.051908</td>
      <td>0.134653</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>497.0</td>
      <td>0.669304</td>
      <td>0.792403</td>
      <td>0.297952</td>
      <td>0.051908</td>
      <td>0.134653</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>501.0</td>
      <td>0.669304</td>
      <td>0.792403</td>
      <td>0.297952</td>
      <td>0.051908</td>
      <td>0.134653</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>230.0</td>
      <td>0.669201</td>
      <td>0.868982</td>
      <td>0.324387</td>
      <td>0.103085</td>
      <td>0.066000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>518.0</td>
      <td>0.669201</td>
      <td>0.868982</td>
      <td>0.324387</td>
      <td>0.103085</td>
      <td>0.066000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>226.0</td>
      <td>0.668616</td>
      <td>0.784772</td>
      <td>0.324016</td>
      <td>0.131320</td>
      <td>0.176000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>514.0</td>
      <td>0.668616</td>
      <td>0.784772</td>
      <td>0.324016</td>
      <td>0.131320</td>
      <td>0.176000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>178.0</td>
      <td>0.668301</td>
      <td>0.771614</td>
      <td>0.325860</td>
      <td>0.125080</td>
      <td>0.184000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>466.0</td>
      <td>0.668301</td>
      <td>0.771614</td>
      <td>0.325860</td>
      <td>0.125080</td>
      <td>0.184000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>182.0</td>
      <td>0.667079</td>
      <td>0.838719</td>
      <td>0.321449</td>
      <td>0.156526</td>
      <td>0.142000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>470.0</td>
      <td>0.667079</td>
      <td>0.838719</td>
      <td>0.321449</td>
      <td>0.156526</td>
      <td>0.142000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>177.0</td>
      <td>0.663895</td>
      <td>0.868719</td>
      <td>0.311865</td>
      <td>0.040100</td>
      <td>0.022000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>181.0</td>
      <td>0.663895</td>
      <td>0.868719</td>
      <td>0.311865</td>
      <td>0.040100</td>
      <td>0.022000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>465.0</td>
      <td>0.663895</td>
      <td>0.868719</td>
      <td>0.311865</td>
      <td>0.040100</td>
      <td>0.022000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>469.0</td>
      <td>0.663895</td>
      <td>0.868719</td>
      <td>0.311865</td>
      <td>0.040100</td>
      <td>0.022000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>225.0</td>
      <td>0.663235</td>
      <td>0.867930</td>
      <td>0.312608</td>
      <td>0.014815</td>
      <td>0.008000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>229.0</td>
      <td>0.663235</td>
      <td>0.867930</td>
      <td>0.312608</td>
      <td>0.014815</td>
      <td>0.008000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>513.0</td>
      <td>0.663235</td>
      <td>0.867930</td>
      <td>0.312608</td>
      <td>0.014815</td>
      <td>0.008000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>517.0</td>
      <td>0.663235</td>
      <td>0.867930</td>
      <td>0.312608</td>
      <td>0.014815</td>
      <td>0.008000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>227.0</td>
      <td>0.662184</td>
      <td>0.602403</td>
      <td>0.317502</td>
      <td>0.173736</td>
      <td>0.424238</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>231.0</td>
      <td>0.662184</td>
      <td>0.602403</td>
      <td>0.317502</td>
      <td>0.173736</td>
      <td>0.424238</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>515.0</td>
      <td>0.662184</td>
      <td>0.602403</td>
      <td>0.317502</td>
      <td>0.173736</td>
      <td>0.424238</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>519.0</td>
      <td>0.662184</td>
      <td>0.602403</td>
      <td>0.317502</td>
      <td>0.173736</td>
      <td>0.424238</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>161.0</td>
      <td>0.659445</td>
      <td>0.867930</td>
      <td>0.269449</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>165.0</td>
      <td>0.659445</td>
      <td>0.867930</td>
      <td>0.269449</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>449.0</td>
      <td>0.659445</td>
      <td>0.867930</td>
      <td>0.269449</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>453.0</td>
      <td>0.659445</td>
      <td>0.867930</td>
      <td>0.269449</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>179.0</td>
      <td>0.658535</td>
      <td>0.764509</td>
      <td>0.317548</td>
      <td>0.051637</td>
      <td>0.123921</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>183.0</td>
      <td>0.658535</td>
      <td>0.764509</td>
      <td>0.317548</td>
      <td>0.051637</td>
      <td>0.123921</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>467.0</td>
      <td>0.658535</td>
      <td>0.764509</td>
      <td>0.317548</td>
      <td>0.051637</td>
      <td>0.123921</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>471.0</td>
      <td>0.658535</td>
      <td>0.764509</td>
      <td>0.317548</td>
      <td>0.051637</td>
      <td>0.123921</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>803.0</td>
      <td>0.655694</td>
      <td>0.867930</td>
      <td>0.244797</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.1, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>807.0</td>
      <td>0.655694</td>
      <td>0.867930</td>
      <td>0.244797</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.1, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>1091.0</td>
      <td>0.655694</td>
      <td>0.867930</td>
      <td>0.244797</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.1, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>1095.0</td>
      <td>0.655694</td>
      <td>0.867930</td>
      <td>0.244797</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.1, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': False, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>160.0</td>
      <td>0.650157</td>
      <td>0.867930</td>
      <td>0.294609</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>448.0</td>
      <td>0.650157</td>
      <td>0.867930</td>
      <td>0.294609</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.9, 'vect__min_df': 0.05, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': True}</td>
    </tr>
    <tr>
      <td>193.0</td>
      <td>0.648121</td>
      <td>0.868193</td>
      <td>0.244444</td>
      <td>0.003960</td>
      <td>0.002000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>197.0</td>
      <td>0.648121</td>
      <td>0.868193</td>
      <td>0.244444</td>
      <td>0.003960</td>
      <td>0.002000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovo', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>481.0</td>
      <td>0.648121</td>
      <td>0.868193</td>
      <td>0.244444</td>
      <td>0.003960</td>
      <td>0.002000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.1, 'vect__norm': 'l1', 'vect__smooth_idf': True, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
    <tr>
      <td>485.0</td>
      <td>0.648121</td>
      <td>0.868193</td>
      <td>0.244444</td>
      <td>0.003960</td>
      <td>0.002000</td>
      <td>{'clf__C': 0.01, 'clf__cache_size': 100, 'clf__decision_function_shape': 'ovr', 'clf__kernel': 'linear', 'clf__probability': True, 'clf__random_state': 1984, 'vect__analyzer': 'word', 'vect__lowercase': False, 'vect__max_df': 0.95, 'vect__min_df': 0.1, 'vect__norm': 'l1', 'vect__smooth_idf': False, 'vect__stop_words': 'english', 'vect__strip_accents': 'unicode', 'vect__sublinear_tf': True, 'vect__use_idf': False}</td>
    </tr>
  </tbody>
</table>

### What happened?

And... disapointment, I guess every data scientist will come accross this at some point: for a runtime of 8 hours I get very little usable results.

The problem here is that most recall, precision and F1 values are of 0.0. Why? well it seems like a lot of precision and recall calculations involved a division by zero.

Lets think about why: given the formula for calculating precision:

$\text{Precision}=\frac{truePositive}{truePositive + falsePositive}$

If we no true positives or false positives, in our case, no sensitive documents or no insensitive documents "wrongly" classified as sensitive in our test set, then we have a division by zero. And the same if true of Recall if there are no sensitive documents or no sensitive documents "wrongly" classified as insensitive:

$\text{Recall}=\frac{truePositive}{truePositive+falseNegatives}$

#### Why?

Well the answer here is that my document collection is too small, or more specifically, that it holds a very small share of actually sensitive documents. So given randomly defined K-folds the odds of some not having any true positives, false positives or false negatives are quite high.

#### So?

The solution to this problem lies in the [stratified k-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) cross-validation technique, essentially the same as a k-fold cross validation except that the data is split into folds that preserve the share of each class .

I am going to have to re-run the Grid Search using stratified k-fold cross validation which should remove this problem and yield actually useful results. I am probably going to try and find better hardware somehow, which might allow me to search a bigger parameter space.

## Again

```python
from time import time

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# find the best parameters for both the feature extraction and the
# classifier

metrics = ["accuracy", "balanced_accuracy", "average_precision", "neg_brier_score", "f1", "f1_micro", "f1_macro", "f1_weighted", "neg_log_loss", "precision", "recall", "jaccard", "roc_auc", "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted", "roc_auc_ovo_weighted"]

grid_search = GridSearchCV(pipeline,
                           # parameters dictionary
                           parameters,
                           # list of metrics
                           scoring=metrics,
                           # use stratified k-folds
                           cv=StratifiedKFold(),
                           # use all available CPUs
                           n_jobs=2,
                           # numbers of copies of dataset to keep in RAM
                           pre_dispatch=2,
                           # "Refit an estimator using the best
                           # found parameters on the whole dataset"
                           # refit optimizing for recall
                           refit="recall",
                           verbose=10)
```

```python
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(train_data, train_labels)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

```

```python
from IPython.display import display, HTML
from pandas import DataFrame

# Convert results dictionary to dataframe
df = DataFrame.from_dict(grid_search.cv_results_)

# display dataframe as table
# display(HTML(df.to_html()))

# export results dataframe to CSV
df.to_csv('/mnt/btrfs/git_repositories/svm-grid-search/results/grid_search_stratified_results.csv', index=True)
```

## XGBoost

### Tree boosting

Tree boosting is the process of combining many _weak learner_ classifier trees into a _strong learner_ boosted tree.

### Grabient Boosting parameters

- Tree-Specific Parameters: These affect each individual tree in the model.
- Boosting Parameters: These affect the boosting operation in the model.
- Miscellaneous Parameters: Other parameters for overall functioning.

### Sources

- https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

```python
xgboost_parameters = {
    # set fixed random state of comparison
    # does not matter as long as it is consistent
    "clf__random_state": [1984],
    # gbtree and dart use tree based models
    # while gblinear uses linear functions.
    "clf__booster": ["gbtree", "gblinear"],
    # learning_rate
    "clf__learning_rate": [0.1, 0.9],
    # Minimum loss reduction required to make a further
    # partition on a leaf node of the tree.
    # The larger gamma is, the more conservative the algorithm will be.
    "clf__min_split_loss": [0, 10],
    "clf__max_depth": [2, 3],
    # Setting it to 0.5 means that XGBoost would randomly sample half of the training
    # data prior to growing trees. and this will prevent overfitting.
    # Subsampling will occur once in every boosting iteration.
    "clf__subsample": [0.2, 0.8],
    # L2 regularization term on weights. Increasing this value will make model more conservative.
    "clf__reg_lambda": [1, 10],
    # L1 regularization term on weights. Increasing this value will make model more conservative.
    "clf__reg_alpha": [0, 10],
    # Control the balance of positive and negative weights, useful for unbalanced classes.
    # A typical value to consider: sum(negative instances) / sum(positive instances).
    "clf__scale_pos_weight": [1, 3299/502 ],
    # Specify the learning task and the corresponding learning objective.
    "clf__objective": ["binary:logistic", "reg:logistic"],
    "clf__n_jobs" : [cpu_count()]
}
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# Create the Pipeline
pipeline = Pipeline(
        steps=[
            ("vect", TfidfVectorizer()),
            ("clf", XGBClassifier()),
        ],
        verbose=False,
    )

# Unify parameters dictionary
parameters = {}
parameters.update(xgboost_parameters)
parameters.update(tfidf_parameters)
```

```python
from time import time

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# find the best parameters for both the feature extraction and the
# classifier

metrics = ["accuracy", "balanced_accuracy", "average_precision", "f1", "f1_micro", "f1_macro", "f1_weighted", "precision", "recall", "roc_auc"]

grid_search = GridSearchCV(pipeline,
                           # parameters dictionary
                           parameters,
                           # list of metrics
                           scoring=metrics,
                           # use stratified k-folds
                           cv=StratifiedKFold(),
                           # use all available CPUs
                           n_jobs=-1,
                           # numbers of copies of dataset to keep in RAM
                           pre_dispatch=2,
                           # "Refit an estimator using the best
                           # found parameters on the whole dataset"
                           # refit optimizing for recall
                           refit="roc_auc",
                           verbose=10)
```

```python
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
t0 = time()
grid_search.fit(train_data, train_labels)
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

```

```python
from IPython.display import display, HTML
from pandas import DataFrame

# Convert results dictionary to dataframe
df = DataFrame.from_dict(grid_search.cv_results_)

# display dataframe as table
# display(HTML(df.to_html()))

# export results dataframe to CSV
df.to_csv('/mnt/btrfs/git_repositories/svm-grid-search/results/xgboost_grid_search_stratified_results.csv', index=True)
```
