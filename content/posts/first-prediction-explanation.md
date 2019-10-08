---
title: "First Prediction Explanation"
date: 2019-09-26T17:11:57+01:00
draft: false
---

# Explaining Classifier predictions with Lime

Adding Lime to text classification predicitons allows us to understand why the classifier made its decision by highlighting which text features influenced its choice. It is a particularly simple library to use, only requiring you to create an `explainer` instance and to pass it the parameters for your prediction to get an _explanation_ for that prediction. That explanation consists of an order list of words that were used to make the prediction. The [paper on lime](https://arxiv.org/abs/1602.04938) is quite interesting, but I have only skimmed over it and have only really used Lime as a _black box_, I will need to give it a proper look, notably since:

> this explainer works for any classifier you may want to use, as long as it implements predict_proba.

Futhermore I have yet to understand any of the classifiers I am using and their differences, I will need to explore them.

## References

- [Lime: Explaining the predictions of any machine learning classifier](https://github.com/marcotcr/lime)
