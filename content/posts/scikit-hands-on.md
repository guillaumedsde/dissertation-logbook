---
title: "Scikit Hands On"
date: 2019-09-24T15:53:10+01:00
draft: false
---

## scikit learn

*Notes from the [scikit tutorial](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)*

* **Supervised learning**
  * predict some attributes of given data
  * classification "binning" of discreet data
  * regression predict continuous variable data
* **Unsupervised Learning**
  * set of input vectors without any target value
  * goal is to **cluster** data into groups
  * or estimate **data density**
  * or project higher dimensional data onto lower dimensional spaces (usually 2 or 3 for visualization)

## Sparse Matrices

Matrices that contain mostly `0` values. Sparsity of a matrix is calculated as such:

`sparsity = sum of '0' elements / total elements`

Large memory footprint

## Text pre-processessing
Some aspects of text pre-processessing:

* **Tokenization**
  * sentences to words
* remove punctuation
* remove stopwords
  * frequent words such as "the", "is" ...
* **Stemming**
  * reduce words to their roots
  * e.g. studying, studies -> study
  * works by cropping common suffixes from words to obtain its **stem**
* **Lemmatization**
  * _"smarter"_  reducing of words to their roots
  * uses a mapping of word to **lemma**


## Text processing
* Feature Extraction
  * **Bag of Words** frequency of all words in a text


## Resources
* [SciKit introduction to Machine Learning Classification](https://scikit-learn.org/0.18/tutorial/basic/tutorial.html)
  * [SciKit Practical text classification](https://scikit-learn.org/0.18/auto_examples/text/document_classification_20newsgroups.html)
* [Text Processing for Machine Learning](https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958)
