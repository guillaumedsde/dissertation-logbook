---
title: "Sensitive Feature Density"
date: 2020-01-15T15:06:01Z
draft: false
---

I wrote a [quick literature review](https://harpocrates-app.gitlab.io/notech/document_visualization.pdf) on document overview visualization techniques. For our usage, a document level overview that would allow reviewers to, at a glance, get an overview of a document's potential sensitive sections. One way of doing this could be to use the density of Lime explaining features as an indication of that paragraph's sensitivity. This rests upon the assumption that the more "sensitive explaining features" a paragraph contains the more likely it is to contain actually sensitive content.

I have touched upon this point in a previous post: Lime highlights features which contribute to the Classifier's prediction, whether these features are actually sensitive is yet to be determined. In this case, we need to explore whether a sensitive explanation at a particular location in the text is indicative of "nearby" (in the same paragraph) sensitive content. For this I will need actual document section redactions in order to evaluate whether that seems to be the case. I will also probably need to refactor the current application to operate at a paragraph level as opposed to document wide.

If feature explanation proves a valid indication of nearby sensitive section, I could use each paragraph's feature explanation count as a measure of its sensitivity in order to provide a high level overview of the document and its potentially sensitive sections.
