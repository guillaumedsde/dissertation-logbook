---
title: "Better Gridsearch"
date: 2020-01-30T15:12:05Z
draft: true
---

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
