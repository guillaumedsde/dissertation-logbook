---
title: "December Update"
date: 2019-12-18T15:49:04+01:00
draft: false
---

I have explored a couple of issues/topics over the past two weeks. First Having an overview of the document as well as its sensitive sections could definitely be useful for reviewing sensitive documents, I have written research on this in LaTeX, I will convert it to a web format and make a post about it.

I have added a chart for visualizing features explaining the classification, showing the weights of each feature, that is, how much each contributes to the classification of the document. I've encountered a couple issues I have yet to resolve while doing that.

First, I've converted the weights, representing features that contribute to sensitivity as "positive" and features that contribute to "non-sensitivity" as "negative" since our goal is mainly to identify "sensitive" documents. The way Lime naturally represents these weights is as a positive value for features that contribute to the classification, **regardless of whether that classification is "sensitive" or "non-sensitive"** and features that contribute to the converse of the classification as negative values. While it makes more sense in our specific case, I am sceptical about "alterting" the value of the Lime output.

Secondly, I am currently using a "two class" Lime explainer:

{{< figure src="/december-update/twoclass.png" caption="Two class Lime explainer" >}}

An interesting visualization on the

{{< figure src="/december-update/twoclass.png" caption="Two class Lime explainer" >}}

I have also done various layout tweaks, notably displaying the current "path" i.e. the collection and the document set, a quick dark/light theme, a homepage, I've removed the empty side drawer and I've enlarged the main content container to avoid large margins on each side.

## UI Demo

_(right click -> view image for bigger image)_

{{< figure src="/december-update/home-file-upload.gif" caption="Homepage and document upload" >}}

{{< figure src="/december-update/redacting.gif" caption="Document view with classification, explanations and redactions" >}}
