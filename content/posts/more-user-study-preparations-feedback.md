---
title: "More user study preparations, testing and feedback demo"
date: 2020-03-05T10:23:31Z
draft: false
---

## Feedback demo

I gave a demo of the application which resulted in very interesting feedback on the UI, the classifier, and sensitivity redaction as a whole.
The notes I took during this demo are [available here](https://harpocrates-app.gitlab.io/notech/meetings.md.html#special-ui-feedback-meeting).

## CI/CD

### Javascript API client testing

The OpenAPI code generator generates a test suite for the API client by default, I have implemented it in my CI pipeline and coupled it with a code coverage analysis.

### FOSSA License check

I have found an interesting tool, an automated dependency license compliance check called [FOSSA](https://fossa.com/).
It analyses whether your dependencies' licenses are compatible with your projects' license automatically.
It was a quick and interesting addon so I added it to my CI pipeline.

### FOSSA Dependency updater bot

Another interesting quick addon I found is [Dependabot](https://dependabot.com/) a bot that automatically creates and closes Merge Requests for updating dependencies.

## Targeted classifier

Part of the discussion during the demo was around targeted classifiers, in my case, the suggestion came up during a prior meeting: lets try and run the classifier on documents containing particular FOI section redactions.
Thus, I built a "FOI Section 27 classifier" that does indeed perform slightly better than the general classifier with a balanced accuracy of 0.5576 as opposed to 0.5526 for the general classifier.
Furthermore, and perhaps more interesting, the intuition is that explanation features, like their parent classifier, will be more targeted, however, I doubt I will have the time to properly test this.

## Last Test UI tweaks

For the users study I have done some last modifications such as removing all redaction labels except the one of interest: "Personal Information".
I have also removed the per paragraph classifications from the bare test UI.

## Final document selection

I have completed my test setup and document selection script that populates and classifies documents as needed for the user study in a reproducible way.
The documents are now selected for the identification of sensitive personal information.
The best solution I have found once the reviewer is finished is to dump the mongo database, this is done in my script.

## Continued dissertation

I have also continued to write an outline to my dissertation, writing about wireframing, testing and related products. The always up to date PDF can be [found here](https://harpocrates-app.gitlab.io/dissertation/dissertation.pdf)
