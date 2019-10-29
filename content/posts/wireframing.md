---
title: "Wireframing"
date: 2019-10-21T17:57:41+01:00
draft: false
---

Our redaction UI is basically going to be a simplified text editor, as such, there is plenty of inspiration to be taken from the vast landscape of text editing software.

There are things we need to keep in mind however, notably _free form redaction_, that is the ability to draw a shape to redact is going to be out of the scope of this project. In part because it is _generally_ hard to implement and in part because it is _especially_ hard to implement such free form selection in web applications. A notable example is LibreOffice's redaction functionality which is heavily reliant on grid "guidelines" to help redactors create somewhat consistent shapes. Interestingly, it does not acknowledge text at all, essentially all redactions rely on the redactor drawing a shape as opposed to selecting text.

{{% figure src="/wireframing/2.png" caption="LibreOffice Redaction functionality" %}}

Since the scope of our application is focused on textual redactions (we are working on plaintext document) any such freeform redaction is out of scope. Instead, the application will rely on redactors selecting text to be redacted. As such, the number of actions on a given text selection will be minimal:

- Redact once
- Redact in document
- Redact in document set
- Remove redaction

A tooltip style "popup" menu with these options fading in like what is done on the Ghost CMS or on Medium would be an elegant way to display possible interactions on the text selection:

{{% figure src="/wireframing/ghost.png" caption="Ghost tooltip" %}}

{{% figure src="/wireframing/medium.png" caption="Medium tooltip" %}}

I like the idea of seperating Document wide functionality in a side menu and keeping selection relevant options near the selection for quicker actions. Google Documents does not have a popup menu, so it keeps all its functionalities (in part because they are numerous) "up there":

{{% figure src="/wireframing/docs_menu.png" caption="Google Docs menu bar" %}}

However, I don't think I want to keep this format, this long bar with lots of button is not inviting as it contains too many possible options. Furthermore, considering how little possible interactions there are in the scope of this application, having space for so many functionalities is not needed.

Exploring Google Docs' UI I was also reminded of the Context Menu. It is useful and universal, people expect a right click to bring up interactivity on what is under the mouse cursor. While it is intuitive, I do not like the idea of taking over the context menu of people's browsers, it is invasive, it overrides the browser's built in useful functionality.

{{% figure src="/wireframing/1.png" caption="Google Docs's takeover of the context menu" %}}

The set of documents I will be working from will be HTML. As such, these documents will have some formatting that I should try to preserve. On the other hand, this formatting is minimal and the application is more geared towards plaintext document redactions. As such, I would like to try and present the text on an imitation of an A4 sheet of paper drawn on the webpage, upon which will be inscribed the text to be redacted.

I mentioned above that I did not want a long options toolbar like Google Docs' however, while I do have few interactions on a selection, I do need to have a few document wide interactions, notably:

- Go back to document set
- Download Redacted Copy of document
- (Save functionality?)
- Delete document from set

So I still need some form of menu, I like the idea of a minimal _hamburger_ menu like this one:

{{% figure src="/wireframing/burger.gif" caption="Hamburger Menu gotten from [here](https://github.com/mblode/burger)" %}}

Keeping the hamburger menu away from the cursor (usually top left of the screen) is acceptable because it will only contain "document wide" actions which will not be used often.

In summary, the final editing interface would display the text to be redacted within an A4 sheet of paper rendered on screen with the only other element being a hamburger menu on the top left. When the redactor release the mouse button after having selected text, a tooltip style menu will popup with at most 5-7 actions portrayed by icons which can be hovered to get a more detailed description.

## Wireframes

{{% figure src="/wireframing/home-wireframe.jpg" caption="Home page wireframe" %}}

{{% figure src="/wireframing/set-wireframe.jpg" caption="Set view wireframe" %}}

{{% figure src="/wireframing/set-menu-wireframe.jpg" caption="Set menu view wireframe" %}}

{{% figure src="/wireframing/page-wireframe.png" caption="Document View wireframe" %}}

{{% figure src="/wireframing/menu-wireframe.png" caption="Hamburger Menu wireframe" %}}

{{% figure src="/wireframing/tooltip-wireframe-cropped.jpg" caption="Text highlight Tooltip wireframe" %}}

## Edits

_23/10/2019_

Now having given this more thought, the idea of a hamburger menu is not great. Having to click twice to access any feature in the menu is not a good UX. Rather, having a sidebar with icons and keeping a hamburger menu that expands the sidebar to also show text next to the icons is a better idea because it keeps the functions accessible while not cluttering too much space. This works better than a top menu like Google Docs' because we can still expand the menu to get more information about the icons as opposed to having to wait for an information tooltip to appear.

Furthermore, a couple more actions on a redaction came to my mind while I was exploring possibilities, so now we could have:

- Redact once
- Redact in document
- Redact in document set
- Remove this redaction
- Remove redaction in document
- Remove redaction in document set

And for an entire document:
- Go back to document set
- _seperator_
- undo
- redo
- _seperator_
- Download redacted document
- Download unredacted document
- (Save functionality?)
- _seperator_
- Delete document from set