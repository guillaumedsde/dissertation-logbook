---
title: "Fixing Redactions"
date: 2019-11-26T13:04:04Z
draft: false
---

This week I've focused on trying to implement the text redact feature. My previous attept at getting character offsets and using [Mozilla's react-content-marker](https://github.com/mozilla/react-content-marker) were not very fruitful. This required using an ordered set of character ranges that must not overlap, these requirements were too painful to implement.

I have however switched to using character offset in the backend for displaying explanations instead of matching text in the body of the document.

My next attempt was using [Facebook's DraftJS](https://github.com/facebook/draft-js). It is more of a text editor, as such, the text is editable by default. As detailed in [this issue](https://github.com/facebook/draft-js/issues/690) making the editor read only while still supporting markup is not built in, I've tried a couple of workarounds: without success, one worked, but text could still be "edited" if the text selection was dragged around.

My next step which is where I currently stand is using a ReactJS library for text annotation (designed for ML classification from what I can tell), it is called [react-annotate](https://www.npmjs.com/package/react-text-annotate). Here is where I stand:

{{< figure src="/redacting/buggy2_redactions.gif" caption="Document sensitivity redactions wth annotation" >}}
