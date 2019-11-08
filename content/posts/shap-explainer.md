---
title: "Shap explanations"
date: 2019-11-08T18:27:46Z
draft: false
---

I'm going to start by [linking to another blog](https://christophm.github.io/interpretable-ml-book/shap.html) which does a far better job than I could at explaining SHAP values for explaning Machine Learning Predictions

This is my attempt at using an alternative to Lime for local Machine Learning prediction explanations. In other words, trying to explain a single prediction from a model rather than the model as a whole. The [shap python package](https://pypi.org/project/shap/) contains different explainers, the ones interesting to me are the [TreeExplainer](https://shap.readthedocs.io/en/latest/#shap.TreeExplainer) for explaining Tree based models, notably scikit-learn's, and the [KernelExplainer](https://shap.readthedocs.io/en/latest/#shap.KernelExplainer) which can explain "the output of any function", operating on models as "Black Boxes".

While trying to get a TreeExplainer to work, I've encountered an issue as [documented here](https://github.com/slundberg/shap/issues/314). I've also tried to get KernelExplainer working but also to no avail, here is my setup in the latter case:

```python
def shap_kernel_explanation(classifier, train_data, test_data):
    # transform data beforehand
    print("transforming training and test data for training KernelExplainer")
    transformed_training_data = classifier.steps[0][1].transform(train_data)
    transformed_test_data = classifier.steps[0][1].transform([test_data])

    # use kmeans, otherwise data too big
    print("Calculating kmeans on training data for KernelExplainer")
    kmeans_training_data = shap.kmeans(transformed_training_data.todense(), 10)

    # build explainer
    print("Training KernelExplainer")
    explainer = shap.KernelExplainer(
        classifier.steps[-1][1].predict, kmeans_training_data
    )

    # calculate SHAP values
    print("Calculating shap values")
    shap_values = explainer.shap_values(transformed_test_data)

    return shap_values
```

As you can see, I am trying to calculate the K means from my transformed data as advised for large datasets by the [SHAP documentation](https://shap.readthedocs.io/en/latest/#shap.KernelExplainer) however, I get an error I do not understand:

```python
transforming training and test data for training KernelExplainer
Calculating kmeans on training data for KernelExplainer
Traceback (most recent call last):
  File "main.py", line 55, in <module>
    print(shap_kernel_explanation(trained_classifier, train_data, to_predict))
  File "/home/architect/git_repositories/dissertation/text-classification/sensitivity_classifier/explainers.py", line 22, in shap_kernel_explanation
    kmeans_training_data = shap.kmeans(transformed_training_data, 10)
  File "/home/architect/.local/share/virtualenvs/text-classification-YBkDjDw-/lib/python3.7/site-packages/shap/explainers/kernel.py", line 48, in kmeans
    ind = np.argmin(np.abs(X[:,j] - kmeans.cluster_centers_[i,j]))
  File "/home/architect/.local/share/virtualenvs/text-classification-YBkDjDw-/lib/python3.7/site-packages/scipy/sparse/base.py", line 430, in __sub__
    raise NotImplementedError('subtracting a nonzero scalar from a '
NotImplementedError: subtracting a nonzero scalar from a sparse matrix is not supported
```

So I blindly try to convert the matrix to a dense matrix:

```python
kmeans_training_data = shap.kmeans(transformed_training_data.todense(), 10)
```

But then the entire python process is killed presumably because it used too much memory?

```python
loading found model for DecisionTreeClassifier from ./models/DecisionTreeClassifier.joblib
transforming training and test data for training KernelExplainer
Calculating kmeans on training data for KernelExplainer
[1]    486439 killed     python main.py
python main.py  159,86s user 26,19s system 124% cpu 2:28,85 total
```
