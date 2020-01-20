---
title: "Sensitive Feature Density, Shap II and other tweaks"
date: 2020-01-15T15:06:01Z
draft: false
---

## Sensitive Feature Density

I wrote a [quick literature review](https://harpocrates-app.gitlab.io/notech/document_visualization.pdf) on document overview visualization techniques. For our usage, a document level overview that would allow reviewers to, at a glance, get an overview of a document's potential sensitive sections. One way of doing this could be to use the density of Lime explaining features as an indication of that paragraph's sensitivity. This rests upon the assumption that the more "sensitive explaining features" a paragraph contains the more likely it is to contain actually sensitive content.

I have touched upon this point in a previous post: Lime highlights features which contribute to the Classifier's prediction, whether these features are actually sensitive is yet to be determined. In this case, we need to explore whether a sensitive explanation at a particular location in the text is indicative of "nearby" (in the same paragraph) sensitive content. For this I will need actual document section redactions in order to evaluate whether that seems to be the case. I will also probably need to refactor the current application to operate at a paragraph level as opposed to document wide.

If feature explanation proves a valid indication of nearby sensitive section, I could use each paragraph's feature explanation count as a measure of its sensitivity in order to provide a high level overview of the document and its potentially sensitive sections.

## Shap, Second Attempt

I've explored around for more resources on the Shap framework for interpreting Machine Learning models and have found some useful resources for understanding how it works.

{{< youtube wjd1G5bu_TY >}}

> This is a simplified overview of the research from which Shap emerged

{{< youtube id="ngOBhhINWb8?start=712" autoplay="false">}}

> And this is a more in depth talk, I've set the video to start at the demo, but I suggest watching all of it

My previous attempts at implementing the SHAP explainer were not successful, I have detailed the issues I stumbled upon in previous posts. The official [README for Shap](https://github.com/slundberg/shap) uses the [XGBoost classifier](https://xgboost.readthedocs.io/en/latest/index.html), without knowing much about the details of how it works, I decided to try using it instead of the SVM classifier I used before in the hope of getting Shap to work. And it seemed to work, Shapley values were calculated without any problem using the `TreeExplainer` built in Shap on the XGBoost trained model.

I sorted them by absolute value to get the top n features with the biggest Shapley values (the ones that contribute most to or against the prediction) and matched them with their textual values. I modified the API and implemented the changes in the ReactJS frontend only to end up with less explaining features than I had asked for.

It seems like some of the to Shap values were those of features that were _not_ in the text I was trying to classify, perhaps my method actually only provides for a global model interpretation as opposed to a local prediction interpretation? But then, why are the top shapley values different for each document given the same model?
A possible explanation might be that the Shap values also underline the importance of the fact that a feature is missing from the text, which would explain why I am getting Shap values for features _not_ in the text.
I might need to look further into the code for the Shap package's [force plot of feature contribution](https://github.com/slundberg/shap/blob/master/shap/plots/force.py) and how it calculates the importance of each feature.
Indeed it seems like it normalizes Shapley values according to the "[base value](https://github.com/slundberg/shap/blob/52eeee48b3cb6754bd993d16e77dddc25f1292a3/shap/plots/force.py#L39)" but I am not sure what this base value actually is.

> This is the reference value that the feature contributions start from. For SHAP values it should be the value of explainer.expected_value.

Another thing I need to look into: it appears it is possible to "train" the Shap explainer by passing it labelled data. I don't have any idea what this entails though: in fact I've come to realize that I should probably try to further understand how Shap works in order to solve these issues, I'll need to think about whether this is really worth it or if the Lime explainer will suffice?

## Other improvements

I have spent quite some time implementing fixes and tweaks to the code I rushed in mid-December. While it was functional there were some bugs as well as some things that needed to be tweaked:

- Fix page layout to fit everything in viewport
- moved most of the feature importance calculation to the backend
- API documentation fixes
- return proper HTTP codes (not all of them are correct)
- Layout tweaks (i.e. navbar)
- added document sensitivity in document set view
- use more theme color as opposed to manually setting them
- round feature weight value
- added classifier string to predictedClassification API object
- added explainer string to explanation API object
- improve population script performance
- centralized all documentation in one URL: [harpocrates-app.gitlab.io/harpocrates](https://harpocrates-app.gitlab.io/harpocrates/)
- other code cleanup and optimizations

<!-- https://youtu.be/ngOBhhINWb8?t=712 -->
