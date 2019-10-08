---
title: "k-fold cross-validation and more explainers"
date: 2019-10-03T10:46:46+01:00
draft: false
---

## k-fold cross-validation

In order to evaluate the performance of my multiple classifiers, I have been advised to perform a [**K-fold cross-validation**](https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f) on them. Given my relatively small dataset, this cross validation (which I perform for k=5 splits of the dataset) is a good way to reflect all sensitivities in the dataset.

If I wanted to keep training 5-10 classifiers to try them out, I had to start using multiple cores in order to speed things up. Especially since I have 5-10 classifiers with which I need to train 5 models each for k-fold cross-validation.

A problem I encoutered quickly was memory usage, especially since I am not using a Hashing Vectorizer in order to preserve the individual features' values within the model (in this case: words). Indeed, I need these values for inspecting the model and running the Lime local explainer. My first attempts at using a Python processes Pool in order to leverage multiple cores resulted in the my memory and SWAP space being overloaded in seconds. My first though was naturally

> Since my Vectorizer does not hash feature I am overloading my memory

However, model size was not the issue since using a Hashing Vectorizer results in the same memory overload. My theory is that my many "k-split" copies of the entire dataset is what is overloading my memory. So what I have done is that I train multiple models on seperate cores for a given data split in order to only maintain one copy in Memory.

## Explainers

I am trying to explore other Machine Learning model explainers in order to compare them. I have now come to understand the difference between _Local Explainers_ and _Global Explainers_:

**Global Explainers** explain a _Machine Learning model globally_, highlighting which labels and which values of those labels matter most while making an arbitrary prediction.

**Local Explainers** explain a _single prediction_ by a Machine Learning model, highlighting Labels and their values which weighed in for that particular prediction.

While searching I have also found a handful of Python modules for ML model explanation:

- [Lime](https://github.com/marcotcr/lime)
- [ELI5](https://github.com/TeamHG-Memex/eli5)
  - slightly different implementation of Lime
  - and Permutation Importance
- [skater](https://github.com/oracle/Skater)
- [shap](https://github.com/slundberg/shap)

I have already implemented Lime and have run a couple of explanations already: its API is quick simple to use.
I have also implemented ELI5 explanations, they are quite similar to lime but not quite the same: I do not know whether it is using Lime in the background or not but it seems that way.

I need to look into skater and shap in more depth.

## Resources

- Introductions to k-fold cross-validation
  - https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
  - https://machinelearningmastery.com/k-fold-cross-validation/
- [Hands on Machine Learning explainers](https://towardsdatascience.com/explainable-artificial-intelligence-part-3-hands-on-machine-learning-model-interpretation-e8ebe5afc608) series of articles of Machine Learning Model explainers in theory and in practice
- [How to Explain the Prediction of a Machine Learning Model?](https://lilianweng.github.io/lil-log/2017/08/01/how-to-explain-the-prediction-of-a-machine-learning-model.html#beta-black-box-explanation-through-transparent-approximations)
- Permutation importance
  - https://christophm.github.io/interpretable-ml-book/feature-importance.html
  - http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html
