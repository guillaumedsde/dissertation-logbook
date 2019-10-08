---
title: "k-fold cross-validation and more explainers"
date: 2019-10-03T10:46:46+01:00
draft: true
---

# k-fold cross-validation and more explainers

## k-fold cross-validation

In order to evaluate the performance of my multiple classifiers, I have been advised to perform a [**K-fold cross-validation**](https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f) on them. Given my relatively small dataset, this cross validation (which I perform for k=5 splits of the dataset) is a good way to reflect all sensitivities in the dataset.

## Explainers

**Global Explainers** explain a _Machine Learning model globally_, highlighting which labels and which values of those labels matter most while making an arbitrary prediction.

**Local Explainers** explain a _single prediction_ by a Machine Learning model, highlighting Labels and their values which weighed in for that particular prediction.

Python modules for ML explainers:

- [Lime](https://github.com/marcotcr/lime)
- [ELI5](https://github.com/TeamHG-Memex/eli5)
  - slightly different implementation of Lime
  - and Permutation Importance
- [skater](https://github.com/oracle/Skater)
- [shap](https://github.com/slundberg/shap)

### Permutation Importance

//TODO

## Resources

- Introductions to k-fold cross-validation
  - https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
  - https://machinelearningmastery.com/k-fold-cross-validation/
- [Hands on Machine Learning explainers](https://towardsdatascience.com/explainable-artificial-intelligence-part-3-hands-on-machine-learning-model-interpretation-e8ebe5afc608) series of articles of Machine Learning Model explainers in theory and in practice
- [How to Explain the Prediction of a Machine Learning Model?](https://lilianweng.github.io/lil-log/2017/08/01/how-to-explain-the-prediction-of-a-machine-learning-model.html#beta-black-box-explanation-through-transparent-approximations)
- Permutation importance
  - https://christophm.github.io/interpretable-ml-book/feature-importance.html
  - http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html
