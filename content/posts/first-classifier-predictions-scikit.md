---
title: "First Classifier Predictions Scikit"
date: 2019-09-25T17:02:39+01:00
draft: false
---

# First Predictions using scikit classifier

Following the scikit tutorial below I have managed to build my first Machine Learning (ML) Pipeline. Using the [20 Newsgroup dataset](http://qwone.com/~jason/20Newsgroups/) built into scitkit, I have trained multiple models using the following classifiers:
* RidgeClassifier
* Perceptron
* PassiveAggressiveClassifier
* MultinomialNB

I then ran them against 2 articles from the Guardian, [one about the new Beijing Airport](https://www.theguardian.com/world/2019/sep/25/daxing-international-airport-zaha-hadid-starfish-opens-beijing), the other about [Brexit and its impact on the Irish Border](https://www.theguardian.com/politics/2019/sep/25/boris-johnson-accused-of-seeking-to-create-no-mans-land-at-irish-border).

Interestingly, all four models classified the latter as `talk.politics.mideast`, thus showing some accuracy (*talk.politics*) but Interestingly classifying the article as related to the *Middle East*. Looking at the article, it mentions the IRA's operations against PM M. Tatcher, if I were to guess, the model has linked the Middle East to "terrorist attacks" which reflects an important issue in Machine Learning: the bias of the data reflected onto a model's output.

Now the former article about the new Beijing airport is clasified as:

```
sci.space
talk.politics.misc
sci.space
sci.space
```

This classification as `talk.politics.misc` is somewhat accurate, the article was from the politics section of The Guardian and it does mention the geopolitical implications of the opening of one of the biggest airports in the world for the People's Republic of China. Airport are somewhat "science" related, which is perhaps why it was classified as `sci.space`.

I have used these classifiers as *black boxes* in my Machine Learning Pipeline and do not understand their inner workings, in fact I still need to investigate more of the other steps in the Pipeline. These predictions are not very accurate, it would be interesting to understand why the classifier made such choices. To do that I could try and use the [LIME framework](https://github.com/marcotcr/lime) to try and understand the models' decision, I also need to further investigate each step of the ML pipeline.

## Resources
* [scikit training and running a classifier](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#training-a-classifier)
