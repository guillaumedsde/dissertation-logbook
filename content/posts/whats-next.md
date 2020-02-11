---
title: "Whats Next"
date: 2020-02-11T09:47:02Z
draft: false
---

First, I'll start by saying, the [GridSearch](https://dissertation.guillaume.desusanne.com/posts/grid-search/) I ran was fairly disapointing: First, I performed non-stratified K-folds and got meaningless results, then I obtained poor results with stratified K-folds which are probably linked to the relatively small parameter space I can explore with the hardware I have available. One potential solution would be to reuse the model from [McDonald et al. (2017)](doi.org/10/ggh8p5), however, it uses n-grams > 1 so I would definitely have to bear that in mind, especially paying attention to the behaviour of Lime explanations. In theory there should not be any technical issues since the entire Harpocrates app uses character offsets, regardless of whether they point to a word or multiple. However, it would be interesting to have a look at the quality of the explanations.

Then comes the issues of the [potential unreliablity](https://dissertation.guillaume.desusanne.com/posts/svc-predict-proba/) of the `predict_proba` method in LibSVM based SVC classifiers (observed in bother scikit-learn's SVC implementation as well as ThunderSVM's) which I also have to look into, potentially ignoring it, using the `decision_function` method or changing classifier.

I tried running a [grid search](https://dissertation.guillaume.desusanne.com/posts/grid-search/) on the XGBoost classifier parameters, but that did not yield very conclusive results either due to again, a large parameter space and limited hardware.

On the frontend side there are a couple of issues remaining. First, the document uploads are quite limited, only taking in raw textual data and posting it to the API, ignoring the filename, and it is also not currently possible to upload multiple files, to choose the classifier or the granularity level for the document analysis, so I have had to resort to using a population script, although I might have found a fix for file uploading.

Then I should also fix the Lime explanations, which are currently calculated at the level of granularity chosen for the document (line, paragraph, document) thus, at the line level, almost every feature is an explanation which is not legible, I should calculate explanations at the document level regardless of the granularity of analysis selected.
