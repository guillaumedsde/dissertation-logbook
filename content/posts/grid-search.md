---
title: "Grid Search for finding optimal parameters"
date: 2020-02-03T14:04:04Z
draft: false
---

_note: this is a Mardown render of a Jupyter notebook whose [latest version is available here](https://gitlab.com/harpocrates-app/svm-grid-search)_

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
    "vect__norm": ("l1", "l2"),
    "vect__use_idf": (True, False),
    "vect__smooth_idf": (True, False),
    "vect__sublinear_tf": (True, False),
    "vect__analyzer": ["word"],
    "vect__stop_words": ["english"],
    "vect__strip_accents": ["unicode"],
    "vect__lowercase": (True, False),
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
    [Parallel(n_jobs=2)]: Done   3 tasks      | elapsed:   34.8s
    [Parallel(n_jobs=2)]: Done   6 tasks      | elapsed:   52.3s
    [Parallel(n_jobs=2)]: Done  11 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=2)]: Done  16 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=2)]: Done  23 tasks      | elapsed:  3.3min
    [Parallel(n_jobs=2)]: Done  30 tasks      | elapsed:  4.1min

### Results

Lastly, lets display the results and also save them as a CSV:

```python
from IPython.display import display, HTML
from pandas import DataFrame

# Convert results dictionary to dataframe
df = DataFrame.from_dict(grid_search.cv_results_)

# display dataframe as table
display(HTML(df.to_html()))

# export results dataframe to CSV
df.to_csv('/mnt/btrfs/git_repositories/svm-grid-search/results/grid_search_results.csv', index=True)
```

```python

```
