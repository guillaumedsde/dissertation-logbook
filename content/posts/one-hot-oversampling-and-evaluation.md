---
title: "One Hot encoding and resampling"
date: 2020-02-15T13:18:43Z
draft: true
---

As I've [mentioned before](https://dissertation.guillaume.desusanne.com/posts/whats-next/) my GridSearch on a TF-IDF and SVC Pipeline parameter space did not yield very conclusive results, especially compared with those obtained on the same dataset by [McDonald et al. (2017)](https://doi.org/10.1007/978-3-319-56608-5_35). So, taking inspiration from this work, I'm going ahead with a couple of things. I'm going to try and use a "one-hot encoder" since this is what is used in the baseline in the aforementioned paper. I'm also going to try and resample the dataset ot remediate its imbalance.

## Resampling

- `class_weight` parameter on most scikit classifiers
- SMOTE _Synthetic Minority Over-Sampling Technique_
  - Boost mintority class by generating _fake_ instances of the minorty class using K-nearest neighbor clustering
- AdaBoost _adaptive boosting_
  - "meta-estimator"
  - adaptive because, fit a classifier, then fit weak learners adapted to correct previous classifier's misclassification
- Tomek link removal
  - remove borderline example
    - i.e. close neighbors that don't share the same label
- Condensed nearest neighbor
  - goal to remove instance from majority class that are distant from decision border

"Sensitive" documents in my dataset are quite rare and so the dataset is considerably imbalanced, skewed towards insensitive documents. To remediable this, [McDonald et al. (2017)](https://doi.org/10.1007/978-3-319-56608-5_35) uses random downsampling of the majority class. I'm also going to try and randomly downsample in the same way, but I'm also going to try and use **SMOTE** and **Condensed Nearest Neighbor** to try to resample in a "non-arbitrary" manner.

I'll be reusing my GridSearch code, we'll see how it goes.

## Evaluation

I need to start evaluating my application to evaluate how it performs and which part of it perform well. Here are a few of the question I am trying to answer

- How useful are the predicted sensitivity classifications?
- How useful are the sensitive feature explanations ?
- How useful are the non-sensitive feature explanations?
- How useful is the explanation features plot

There are multiple ways to evaluate this:

- questionnaire with Likert scale
- mouse position/click data
  - mouse click is easy to obtain, mouse position, not so much
- time review process with/without feature
  - time consuming since have to review more documents
