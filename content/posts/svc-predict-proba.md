---
title: "SVC `predict_proba` reversed results in Binary classification"
date: 2020-02-09T14:20:15Z
draft: false
---

I noticed something strange. The reason I decided to spend some time running a GridSearch over the parameter space of my sensitive document classification pipeline is because I was getting what seemed like very poor classification results.

To get the probability of each class, I am using `svm.predict_proba()` in the scikit-learn API. The [documentation for this method](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict_proba) comes with a warning note:

> The probability model is created using cross validation, so the results can be slightly different than those obtained by predict. Also, it will produce meaningless results on very small datasets.

"Fair" I thought, especially since I [need](https%3A%2F%2Fmarcotcr.github.io%2Flime%2Ftutorials%2FLime%20-%20basic%20usage%252C%20two%20class%20case.html) the `predict_proba` method for computing the Lime interpretation

> this explainer works for any classifier you may want to use, as long as it implements predict_proba.

To calculate these probabilities, the `probability=True` needs to be passed the the `SVC` object. From what I understand, to calculate each class' probability for a given document, sklearn performs a logistic regression on the `decision_function`'s scores and then cross-validates them on the training data.
However, there is an odd behaviour from the SVC using the `probability=True` flag for binary classification: the probabilities are reversed. There is [an issue](https://github.com/scikit-learn/scikit-learn/issues/13662) documenting this specific reversal in the binary classification case. There is also [an issue](https://github.com/scikit-learn/scikit-learn/issues/13211) documenting the broader API inconsitency of `predict` and `predict_proba`. Furthermore, the [documentation suggests](https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities) avoiding the use of `probability=True` in binary classification cases.

This is problematic for me since I _need_ to have the probabilities for extracting Lime Explanations. I could try and see if using Shap explanations actually avoids this issue but since SVC is not a tree algorithm, Shap values are expensive to compute, I would need to investigate this further.

The other option would be to use another classifier, ideally I would find one with a parallel implementation like what ThunderSVM does in order to speed up computation. Which I will need if I do decide to switch classifier since I would need to do a new grid search.
