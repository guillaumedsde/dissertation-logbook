---
title: "Paragraphs and GridSearch"
date: 2020-01-22T13:34:16Z
draft: false
---

Let's try and classify documents at a paragraph level to see if the extra level of granularity is

## What is a Paragraph?

When trying to formally define problems, one can end up questioning event the seemingly most obvious seemingly established truths. This time, I set out to answer: what is a paragraph? I have tried to develop a regular expression for "a paragraph" and ended up with this expression for matching a paragraph delimiter: `(?=\n\s*\n)` with this (simplified here) code:

```python
import re

text = "..."

r = re.compile(r"(?=\n\s*\n)")
paragraphs = r.split(text)
```

The only issue is that some "paragraphs" consist of only the match for the delimiter. My non-optimal solution to this is to then iterate over paragraphs to append those consisting only of whitespaces to the previous one, it is probably possible to write regex to avoid this but I have not figured out how.

## Classifying a document with arbitrary granularity

Classifying at a paragraph level did not yield very satisfactory levels of "document overview", most paragraphs have very similar classifications making it hard/meaningless to differentiate between them. My model could potentially be an issue, so I've tried to tweak it (see GridSearch below) but I need to look into it more.

So I tried classifiying each line within a document which yielded what seems like more interesting information with some lines clearly being classified as more sensitive than others. To what extent this is true: I do not know, I need word level gold standard annotations for the document in order to be able to determine that.

Furthermore, currently, my implementation displays line or paragraph sensitivity using a "progress bar" component I developed using Material UI ReactJS framework. For now it is the only visualization I could come up with, it works, but I am not completely satisfied with it. Especially with line level classifications, it is a bit verbose and hard to read.

Lastly, implementing this word and paragraph level classifications broke the document wide feature highlighting as well as the redaction functionality. I have started to "generalize" the code to allow me to split the text at an arbitrary level of granularity (even no split, i.e. document level granularity) and retain these features.

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
