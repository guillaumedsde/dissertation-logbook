---
title: "Wireframing II and inspiration from other products"
date: 2019-10-29T18:47:16Z
draft: false
---

## Adobe Reader Pro DC

I found out that Adobe Reader Pro DC has a document redaction functionality, sidenote: this PDF tool, while expensive is actually impressively powerful, I will need to compare it to Foxit PhantomPDF.

First note of interest, it has a redaction repeat functionality, however, it does not redact other instances of the selected text, rather it redacts the highlighted position on the page accross other pages of the document. Interesting feature, I think it is there because this is a PDF redaction toolkit and not all textual information is included in some PDFs (i.e. can't select text on some PDFs).

{{< figure src="/wireframing2/repeat-redactions.png" caption="Repeating redactions across a document" >}}

Comments might definitely be useful when I make the frontend, but for now i don't think its as important as redaction itself.

{{< figure src="/wireframing2/comments.png" caption="Adobe reader Comments can also contain the status of a redaction: rejected, accepted etc..." >}}

Exemptions are only visible when the redaction is selected, this is not great for a quick overview, but on the other hand, where to put them?

I think having a toggle for multiple elements will be useful:

- hide sensitivity explanations
- hide suggested redactions
- show redacted view (blacked out text with exception printed on it)
- show semi redacted view (black background and white text for redactions)
- hide all redactions

{{< figure src="/wireframing2/categorize-redactions.png" caption="Exemptions can be easily annotated to a redaction" >}}

## Foxit PhantomPDF Business

Foxit sells a competing software to Adobe's , it bears a striking resemblance in terms of redaction features

{{< figure src="/wireframing2/foxit.png" caption="Redaction Context menu is quite similar" >}}

It does however, have a search feature which makes it easy to redact text instances across an entire document, which adobe does not have.

{{< figure src="/wireframing2/foxit-search.png" caption="Foxit's search feature" >}}

## Redacted.ai

This is a web applcation that is no publically accessible, a [demo video](https://www.youtube.com/watch?v=hLgwEs1KCdQ) does however gie an outline of how it works.

{{< figure src="/wireframing2/entity-extraction.png" caption="redacted.ai performs entity extraction, identifying names, places and other potentially interesting features for redaction" >}}

{{< figure src="/wireframing2/redactedai.png" caption="redacted.ai redaction view" >}}

Couple of things to note here:

- Document outline is useful
- skip to next file button
- label might be a desirable simplification of comments + FOI exceptions
- search text
- draw (but for us its probably out of scope)

## Conclusions

Drawing from these different softwares, and previous design meeting, there's a couple of things I need to add to the wireframes.

- Document outline
- skip to next file button
- add label
- search text
  - possibly redact results?
- toggles
  - hide sensitivity explanations
  - hide suggested redactions
  - show redacted view (blacked out text with exception printed on it)
  - show semi redacted view (black background and white text for redactions)
  - hide all redactions
- get rid of hamburger menu

## Wireframing II

{{< figure src="/wireframing2/doc-view.jpg" caption="Document view with minimized menu" >}}

{{< figure src="/wireframing2/doc-view-menu.jpg" caption="Document view with expanded menu" >}}

{{< figure src="/wireframing2/tooltip.jpg" caption="Updated tooltip" >}}

{{< figure src="/wireframing2/tooltip-comment.jpg" caption="Tooltip comment popup" >}}
