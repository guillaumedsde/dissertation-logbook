---
title: "Classifier Predictions Trust"
date: 2019-09-25T10:26:54+01:00
draft: false
---

# Classfifier Predictions and Trust

[_Ribeiro et al., 2016_](http://arxiv.org/abs/1602.04938)

Without trust, a tool will not be used. This applies to machine learning models which are typically used as **black boxes** with "blind trust". Furthermore, providing an _explanation_ for the decision of a model will help in diagnosing and fixing incorrect predicitons.

Complexity of some models and high data throughput make it hard to provide an understandable explanation for the model's output, this is why these explanations need to be an **understandable** of the reasons for the output.

_Global Fidelity_ describes an explanation which would accurately represent the reasons for the ouput for the entire data. In order for an explanation to be _understandable_, it must _summarize_ the choices of the model. This summary will not be have **Global Fidelity** but it will have **local fidelity**, i.e. it will correspond to the behavior of the model around the outputted solution.

By formally formulating this decrease in accuracy as a **loss function** finding the most accurate yet understandable explanation becomes a solveable optimization problem.

## Cummulative summary

Given a set of pre-redacted documents, we want to use it to train text analysis models to detect **features that characterize sensitive elements** of an unredacted document. We then seek to evaluate the accuracy of these models possibly using the LIME framework developed by [Ribeiro et al., 2016](http://arxiv.org/abs/1602.04938), also using that model to provide _explanations_ for the ouputted sensitive sections of the document.

## References

- [Ribeiro, M.T., Singh, S. and Guestrin, C. 2016. ‘Why Should I Trust You?’: Explaining the Predictions of Any Classifier. arXiv:1602.04938 [cs, stat].](http://arxiv.org/abs/1602.04938)
