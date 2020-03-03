---
title: "Verifying Classifier and more user study preparations"
date: 2020-03-03T08:51:15Z
draft: false
---

## Classifier fix

This week, with some much appreciated help, I've spent some time verifying the results of last week's fixed classifier. The balanced accuracy is now in the expected ranges, so I have got a working baseline classifier.

## More User study preparations

I have spent some time filtering the documents to get an appropriate selection for the user study. I have written multiple filters for the test data to recreate a confusion matrix amongst other things:

- True Positive
- False Positive
- True Negative
- False Negative
- Document Length
- Contains FOIA Section 27 sections

However, using a document length of 4000 I have not managed to obtain at least one document in each category of the confusion matrix. Without the maximum document length, it works, so I might just skip filtering by document length and the user study will be longer. Otherwise, I could manually find a True Negative that does not contain Section 27 sensitivities. This is the last remaining step as I have built the population script to setup a test environment.

I have also given some thought to the design of the user study: I will give the questionnaire after each batch of documents given an interface except for the questionnaire about specific components (Predicted classification, explanations etc...). I
have modified the questionnaire accordingly.

## Dissertation

I have also begun writing my dissertation paper, specifically the section about the user study, since it is "freshest" in my mind.
