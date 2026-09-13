/*!
 * figures.js — captions and a feedback handle for images and Mermaid diagrams.
 *
 * Highlight comments (js/annotations.js) anchor on *text*, so a picture has
 * nothing to select. This gives every block image and every Mermaid diagram a
 * <figcaption> — 「图 N」 + the image's alt / the diagram's `title:` (Mermaid
 * front matter) or first `%%` comment — and a small corner button (next to the
 * copy button on diagrams) that selects the caption's title, which pops the
 * usual 赞 / 存疑 / 评论 / 建议修改 toolbar. The caption title is the passage:
 * stable as long as the alt stays, readable in the GitHub comment and the brief.
 *
 *   <p><img alt="…"></p>        ->  <figure class="post-figure"><img><figcaption class="post-figcaption">
 *                                     <span class="fig-no">图 N</span><span class="fig-title">…</span></figcaption>
 *                                     <div class="fig-tools"><button class="code-copy fig-feedback">…</button></div></figure>
 *   <div class="mermaid">…</div> ->  unchanged, + a .fig-tools strip inside holding code-copy's button and
 *                                     ours (32 px targets, padded dead zone), the <figcaption> as its next sibling
 *
 * Diagrams render asynchronously (rich-content.html), so they are picked up by
 * a MutationObserver; the number counts images and diagrams in document order
 * (a diagram is either its source <pre> or its .mermaid at any moment).
 */
(function () {
  'use strict';

  var ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M1.75 1h8.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0 1 10.25 10H7.061l-2.574 2.573A1.458 1.458 0 0 1 2 11.543V10h-.25A1.75 1.75 0 0 1 0 8.25v-5.5C0 1.784.784 1 1.75 1ZM1.5 2.75v5.5c0 .138.112.25.25.25h1a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h3.5a.25.25 0 0 0 .25-.25v-5.5a.25.25 0 0 0-.25-.25h-8.5a.25.25 0 0 0-.25.25Zm13 2a.25.25 0 0 0-.25-.25h-.5a.75.75 0 0 1 0-1.5h.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0 1 14.25 12H14v1.543a1.458 1.458 0 0 1-2.487 1.03L9.22 12.28a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215l2.22 2.22v-2.19a.75.75 0 0 1 .75-.75h1a.25.25 0 0 0 .25-.25Z"></path></svg>';
  var FIGURE_SELECTOR = 'figure.post-figure, .mermaid, code.language-mermaid';

  var container;

  function norm(s) { return String(s || '').replace(/\s+/g, ' ').trim(); }

  // Ordinal of an image / diagram among all of them in the article.
  function figureNo(el) {
    var all = container.querySelectorAll(FIGURE_SELECTOR), n = 0;
    for (var i = 0; i < all.length; i++) {
      var f = all[i];
      // a diagram's <pre><code> and its later .mermaid never coexist; skip code inside a figure we made
      if (f.tagName === 'CODE' && f.closest('.mermaid')) continue;
      n++;
      if (f === el || f.contains(el) || el.contains(f)) return n;
    }
    return n + 1;
  }

  // `---\ntitle: …\n---` front matter, else the first `%% …` comment line.
  function mermaidTitle(source) {
    var m = /^\s*---\s*\n([\s\S]*?)\n---/.exec(source || '');
    if (m) { var t = /^\s*title:\s*(.+?)\s*$/m.exec(m[1]); if (t) return norm(t[1].replace(/^["']|["']$/g, '')); }
    // only a comment on the very first line counts as a title — mid-source comments are just comments
    var first = String(source || '').replace(/^\s*---[\s\S]*?\n---\s*\n/, '').replace(/^\s+/, '').split('\n')[0];
    var c = /^%%\s*(?!\{)(.+?)\s*$/.exec(first || '');
    return c ? norm(c[1].replace(/^(图|title)\s*[:：]\s*/i, '')) : '';
  }

  function caption(no, title) {
    var cap = document.createElement('figcaption');
    cap.className = 'post-figcaption' + (title ? '' : ' is-untitled');
    var num = document.createElement('span'); num.className = 'fig-no'; num.textContent = '图 ' + no;
    cap.appendChild(num);
    if (title) { var t = document.createElement('span'); t.className = 'fig-title'; t.textContent = title; cap.appendChild(t); }
    return cap;
  }

  function button(cap, besideCopy) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'code-copy fig-feedback';
    b.title = '对这张图评论 / 存疑（会选中图题，再从工具条里选）';
    b.setAttribute('aria-label', '对这张图评论');
    b.innerHTML = ICON;
    b.addEventListener('click', function (e) {
      e.preventDefault(); e.stopPropagation();
      var target = cap.querySelector('.fig-title') || cap.querySelector('.fig-no');
      // a tall picture puts its caption below the fold: bring it (and the toolbar
      // annotations.js will hang over it) into view before selecting
      var r = cap.getBoundingClientRect(), vh = window.innerHeight || document.documentElement.clientHeight;
      if (r.top < 90 || r.bottom > vh - 20) cap.scrollIntoView({ block: 'center', behavior: 'smooth' });
      var range = document.createRange(); range.selectNodeContents(target);
      var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
      cap.setAttribute('tabindex', '-1'); cap.focus({ preventScroll: true });
      cap.classList.add('is-picked'); setTimeout(function () { cap.classList.remove('is-picked'); }, 1800);
      // annotations.js listens to selectionchange and shows its toolbar at the selection
    });
    return b;
  }

  // Corner strip that holds the buttons. Its padding is a dead zone: clicking a
  // few px off a button must not open the zoom lightbox (js/diagram-zoom.js
  // listens for clicks on the whole figure).
  function tools(host) {
    var t = host.querySelector(':scope > .fig-tools');
    if (!t) {
      t = document.createElement('div'); t.className = 'fig-tools';
      t.addEventListener('click', function (e) { e.stopPropagation(); });
      host.appendChild(t);
    }
    // code-copy.js may add its button before or after us: keep it in the strip
    Array.prototype.forEach.call(host.querySelectorAll(':scope > .code-copy:not(.fig-feedback)'), function (c) { t.insertBefore(c, t.firstChild); });
    return t;
  }

  function decorateImages() {
    Array.prototype.forEach.call(container.querySelectorAll('p > img'), function (img) {
      var p = img.parentNode;
      if (p.tagName !== 'P' || p.children.length !== 1 || norm(p.textContent)) return;
      if (img.closest('figure, a, .comment, .annotation-panel, .series-toc, .related-posts')) return;
      var fig = document.createElement('figure');
      fig.className = 'post-figure code-copy-anchor';
      p.parentNode.replaceChild(fig, p);
      fig.appendChild(img);
      var cap = caption(figureNo(fig), norm(img.getAttribute('alt')));
      fig.appendChild(cap);
      tools(fig).appendChild(button(cap, false));
    });
  }

  function decorateDiagrams() {
    Array.prototype.forEach.call(container.querySelectorAll('.mermaid[data-mermaid-source]'), function (d) {
      if (!d.querySelector('svg') && !d.classList.contains('mermaid-error')) return; // still rendering
      var strip = tools(d);
      if (strip.querySelector('.fig-feedback')) return;
      var cap = caption(figureNo(d), mermaidTitle(d.getAttribute('data-mermaid-source')));
      d.parentNode.insertBefore(cap, d.nextSibling);
      d.classList.add('code-copy-anchor');
      strip.appendChild(button(cap, true));
    });
  }

  function init() {
    container = document.querySelector('.post-container');
    if (!container) return;
    decorateImages();
    decorateDiagrams();
    if (window.MutationObserver) {
      new MutationObserver(function () { decorateDiagrams(); }).observe(container, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
