---
title: "Paragraphs and GridSearch"
date: 2020-01-22T13:34:16Z
draft: true
---

Let's try and classify documents at a paragraph level to see if the extra level of granularity is

## What is a Paragraph?

When trying to formally define problems, one can end up questioning event the seemingly most obvious seemingly established truths. This time, I set out to answer: what is a paragraph? Take this nonsensical document for example:

```
Date: 04/05/1997

Something about Star Wars

I am writing something about Star Wars.
I am doing this because I need an example paragraph, the content itself does not matter, I just need text to explain my point and Lorem Lipsums are a bit "overused"

Now This is a second paragraph.
I need it to differentiate it from the second paragraph so I am yet again writing another paragraph.

```

The first approach of "a paragraph is surround by double line returns" brings up more questions: is `Date: 04/05/1997` a paragraph? is `Something about Star Wars`?

<!-- TODO develop on this -->

## Classifying at document and paragraph level

Going back t

## GridSearch

I've had a glimpse at the [GridSearch scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) and have also found [an example](https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py) of it in use for finding the best parameters for a text classifier. Its fairly straightforward to implement and I was curious so I've modified this algorithm slightly, I've tried running it on my personal server, however despite having 12 cores, it only has 8GB of RAM. Instead (and in order to avoid taking over my personal infrastructure and stopping the apps running on it) I've provisioned a DigitalOcean Droplet with 4 cores and 8GB of RAM. My 50\$ DigitalOcean Credits should be enough to run this VM for just under a month, I've set it to explore the following parameter space. We'll see how long it takes, if its too slow I can run it on my personal desktop (6 cores 8GB RAM) but with 4 cores, RAM usage already exceeds 8GB and uses SWAP space, so I might by more RAM.

```python
parameters = {
    "vect__norm": ("l1", "l2"),
    "vect__use_idf": (True, False),
    "vect__smooth_idf": (True, False),
    "vect__sublinear_tf": (True, False),
    "vect__analyzer": ["word"],
    "vect__stop_words": ["english"],
    "vect__strip_accents": ["unicode"],
    "vect__lowercase": (True, False),
    "clf__random_state": [1984],
    "clf__decision_function_shape": ("ovo", "ovr"),
    "clf__kernel": ("linear", "poly", "rbf", "sigmoid"),
    "clf__cache_size": [100],
}
```

After close to five hours, the GridSearch on my particular dataset given the parameter space defined above returned the following optimal parameters:

```
clf__cache_size: 100
clf__decision_function_shape: 'ovo'
clf__kernel: 'linear'
clf__random_state: 1984
vect__analyzer: 'word'
vect__lowercase: True
vect__norm: 'l1'
vect__smooth_idf: True
vect__stop_words: 'english'
vect__strip_accents: 'unicode'
vect__sublinear_tf: True
vect__use_idf: True
```
