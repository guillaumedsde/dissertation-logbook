---
title: "One Hot encoding resampling and user evaluation"
date: 2020-02-15T13:18:43Z
draft: false
---

## Resampling

As I've [mentioned before](https://dissertation.guillaume.desusanne.com/posts/whats-next/) my GridSearch on a TF-IDF and SVC Pipeline parameter space did not yield very conclusive results, especially compared with those obtained on the same dataset by [McDonald et al. (2017)](https://doi.org/10.1007/978-3-319-56608-5_35). So, taking inspiration from this work, I'm going ahead with a couple of things. I'm going to try and use a "one-hot encoder" since this is what is used in the baseline in the aforementioned paper. I'm also going to try and resample the dataset ot remediate its imbalance.

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

For some reason, I cannot reproduce the aforementioned paper's results. I tried a small Grid Search over a parameter space that includes the parameters used in the paper and could only achieve a F1 score of 0.1814, considerably lower than the research's 0.3520. All metrics (F2, precision, recall ad aur_roc) are also lower than expected, I am unsure why and need to investigate more.

## Evaluation

I need to start evaluating my application to evaluate how it performs and which part of it perform well. Here are a few of the question I am trying to answer

- How useful are the predicted sensitivity classifications?
- How useful are the sensitive feature explanations ?
- How useful are the non-sensitive feature explanations?
- How useful is the explanation features plot

There are multiple ways to evaluate this:

- questionnaire with Likert scale question on each feature
- time review process with/without feature
  - time consuming since have to review more documents
- mouse position/click data
  - mouse click is easy to obtain, mouse position, not so much

I am also going to need a document set for testers to review, in this case, a few problems arise: I'll probably not be able to use the dataset I trained the classifier on due to its sensitive nature. Assuming I can find labelled sensitive documents the classifier might perform differently due to the fact it has been trained in a specific "redaction context".

There are a few bugs that might hinder a user evaluation, I have spent some time fixing them and am almost done. Similarly there are some features which might be useful for a user study that are not currently implemented such as a "next document" button, ordering of document sets by sensitive documents, ordering documents by sensitivity ...

Another important issue is getting "users" (i.e. reviewers) to conduct the user study. In any case I need someone who is either familiar with the sensitive document review process, or someone knowledgeable enough in a domain area to be able to review a document for sensitivity, ideally: both. This is going to be hard to find and finding incentives for participation is also going to be hard.

I have started a handout for the user evaluation as [HTML](https://harpocrates-app.gitlab.io/notech/user_evaluation.md.html) or [PDF](https://harpocrates-app.gitlab.io/notech/user_evaluation.md.pdf) (this is an incomplete document I will continue updating it)
