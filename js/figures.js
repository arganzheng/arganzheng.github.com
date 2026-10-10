/*!
 * figures.js — captions and a feedback handle for images, Mermaid diagrams and tables.
 *
 * Highlight comments (js/annotations.js) anchor on *text*, so a picture has
 * nothing to select. This gives every block image and every Mermaid diagram a
 * <figcaption> — 「图 N：」 + the image's alt / the diagram's `title:` (Mermaid
 * front matter) or first `%%` comment (「图 N」 alone when there is no title) — and a corner strip with a 放大 button
 * (js/diagram-zoom.js lightbox; pictures do not zoom on click) and a feedback
 * button that selects the caption's title, which pops the usual 点赞 /
 * 评论 toolbar. The caption title is the passage:
 * stable as long as the alt stays, readable in the GitHub comment and the brief.
 *
 *   <p><img alt="…"></p>        ->  <figure class="post-figure"><span class="fig-media"><img>
 *                                       <div class="fig-tools"><button class="code-copy fig-zoom">…</button><button class="code-copy fig-feedback">…</button></div></span>
 *                                     <figcaption class="post-figcaption"><span class="fig-no">图 N：</span><span class="fig-title">…</span></figcaption></figure>
 *   <div class="mermaid">…</div> ->  its <svg> wrapped in the same .fig-media (sized to the svg's max-width), the
 *                                     <figcaption> as the next sibling, and the .fig-tools strip (code-copy's button,
 *                                     放大, ours) floating on the .fig-media's top-right corner, shown on hover
 *   <pre> / .highlighter-rouge  ->  .code-block with a header bar on top (<div class="code-header">: 「代码块 N：」 /
 *                                     「文本块 N」 + the ```lang title="…" title + language tag + the .fig-tools strip);
 *                                     the handle selects the title, or the first line when there is none
 *   <table>                     ->  the same caption (<div class="post-figcaption table-caption">, 「表 N：」 +
 *                                     title) as the next sibling of the table / its .table-responsive wrapper;
 *                                     the title is the table's <caption> (Pandoc-style `Table: …` paragraph after
 *                                     the table, folded in by _plugins/table_captions.rb), moved into the caption.
 *                                     Tables have their own numbering (表 N), independent of the figures'.
 *
 * Diagrams render asynchronously (rich-content.html), so they are picked up by
 * a MutationObserver; the number counts images and diagrams in document order
 * (a diagram is either its source <pre> or its .mermaid at any moment).
 */
(function () {
  'use strict';

  var ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M1.75 1h8.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0 1 10.25 10H7.061l-2.574 2.573A1.458 1.458 0 0 1 2 11.543V10h-.25A1.75 1.75 0 0 1 0 8.25v-5.5C0 1.784.784 1 1.75 1ZM1.5 2.75v5.5c0 .138.112.25.25.25h1a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h3.5a.25.25 0 0 0 .25-.25v-5.5a.25.25 0 0 0-.25-.25h-8.5a.25.25 0 0 0-.25.25Zm13 2a.25.25 0 0 0-.25-.25h-.5a.75.75 0 0 1 0-1.5h.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0 1 14.25 12H14v1.543a1.458 1.458 0 0 1-2.487 1.03L9.22 12.28a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215l2.22 2.22v-2.19a.75.75 0 0 1 .75-.75h1a.25.25 0 0 0 .25-.25Z"></path></svg>';
  // octicon screen-full: the 放大 button (js/diagram-zoom.js lightbox)
  var ZOOM_ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M1.75 10a.75.75 0 0 1 .75.75v2.75h2.75a.75.75 0 0 1 0 1.5h-3.5a.75.75 0 0 1-.75-.75v-3.5a.75.75 0 0 1 .75-.75Zm12.5 0a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-.75.75h-3.5a.75.75 0 0 1 0-1.5h2.75v-2.75a.75.75 0 0 1 .75-.75ZM2.5 2.5v2.75a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 1.75 1h3.5a.75.75 0 0 1 0 1.5Zm8.25-.75a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0V2.5h-2.75a.75.75 0 0 1-.75-.75Z"></path></svg>';
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

  // `---\ntitle: …\n---` front matter, else the first `%% …` comment line
  // (leading `%%{init: …}%%` directives do not count as lines).
  function mermaidTitle(source) {
    var m = /^\s*---\s*\n([\s\S]*?)\n---/.exec(source || '');
    if (m) { var t = /^\s*title:\s*(.+?)\s*$/m.exec(m[1]); if (t) return norm(t[1].replace(/^["']|["']$/g, '')); }
    // only a comment on the very first line counts as a title — mid-source comments are just comments
    var first = String(source || '').replace(/^\s*---[\s\S]*?\n---\s*\n/, '').replace(/^\s*(%%\{[\s\S]*?\}%%\s*)*/, '').split('\n')[0];
    var c = /^%%\s*(?!\{)(.+?)\s*$/.exec(first || '');
    return c ? norm(c[1].replace(/^(图|title)\s*[:：]\s*/i, '')) : '';
  }

  // `kind` = 「图」 (default, a <figcaption>) or 「表」 (a <div>, since it sits outside a <figure>)
  function caption(no, title, kind) {
    var cap = document.createElement(kind === '表' ? 'div' : 'figcaption');
    cap.className = 'post-figcaption' + (title ? '' : ' is-untitled');
    var num = document.createElement('span'); num.className = 'fig-no'; num.textContent = (kind || '图') + ' ' + no + (title ? '：' : '');
    cap.appendChild(num);
    if (title) { var t = document.createElement('span'); t.className = 'fig-title'; t.textContent = title; cap.appendChild(t); }
    return cap;
  }

  // Select `target`'s text as if the reader had dragged over it; annotations.js
  // listens to selectionchange and shows its toolbar (点赞 / 评论 /
  // 复制 / 搜一搜 / 分享) at the selection. `focusEl` (a caption) is scrolled into
  // view first — a tall picture puts its caption below the fold — focused and
  // flashed, so the reader sees what got picked.
  function pick(target, focusEl) {
    if (focusEl) {
      var r = focusEl.getBoundingClientRect(), vh = window.innerHeight || document.documentElement.clientHeight;
      if (r.top < 90 || r.bottom > vh - 20) focusEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
    var range = document.createRange(); range.selectNodeContents(target);
    var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    if (focusEl) {
      focusEl.setAttribute('tabindex', '-1'); focusEl.focus({ preventScroll: true });
      focusEl.classList.add('is-picked'); setTimeout(function () { focusEl.classList.remove('is-picked'); }, 1800);
    }
  }

  function button(title, onClick, cls, icon) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'code-copy ' + (cls || 'fig-feedback');
    b.title = title;
    b.setAttribute('aria-label', title);
    b.innerHTML = icon || ICON;
    b.addEventListener('click', function (e) { e.preventDefault(); e.stopPropagation(); onClick(); });
    return b;
  }

  // 放大: opens the lightbox on the picture / diagram. Pictures used to open on
  // click; now only this button does, so selecting a caption never zooms.
  function zoomButton(el) {
    var b = button('放大查看（滚轮缩放、拖动平移）', function () { if (window.DiagramZoom) window.DiagramZoom.open(el); }, 'fig-zoom', ZOOM_ICON);
    if (el.tagName === 'IMG') {
      var check = function () { if (window.DiagramZoom && !window.DiagramZoom.zoomable(el) && b.parentNode) b.parentNode.removeChild(b); };
      if (el.complete) check(); else el.addEventListener('load', check);
    }
    return b;
  }

  var FIG_TITLE = '对这张图评论（会选中图题，再从工具条里选）';
  function figureButton(cap) {
    return button(FIG_TITLE, function () { pick(cap.querySelector('.fig-title') || cap.querySelector('.fig-no'), cap); });
  }

  // Strip that holds the buttons (copy · 放大 · feedback). `host` is the
  // element code-copy.js appends its button to; `mount` (default host) is
  // where the strip lives. For pictures and diagrams the mount is the
  // shrink-wrapped .fig-media, so the strip floats on the picture's own
  // top-right corner (hover-revealed, see less/annotations.less); code blocks
  // keep their always-on corner strip.
  function tools(host, mount) {
    var t = host.querySelector(':scope > .fig-tools, :scope > .fig-media > .fig-tools') || (mount && mount.querySelector(':scope > .fig-tools'));
    if (!t) {
      t = document.createElement('div'); t.className = 'fig-tools';
      t.addEventListener('click', function (e) { e.stopPropagation(); });
      (mount || host).appendChild(t);
    }
    // code-copy.js may add its button before or after us: keep it in the strip
    Array.prototype.forEach.call(host.querySelectorAll(':scope > .code-copy:not(.fig-feedback)'), function (c) { t.insertBefore(c, t.firstChild); });
    return t;
  }

  // Inline-block wrapper that shrink-wraps the picture (block display, natural
  // width, centred by the parent); the tool strip sits on its corner.
  function media(el, width) {
    var w = document.createElement('span');
    w.className = 'fig-media';
    if (width) w.style.width = width;
    el.parentNode.insertBefore(w, el);
    w.appendChild(el);
    // An <img> of an SVG that has a viewBox but no width/height has no intrinsic
    // size, and inside a shrink-wrapped wrapper it lays out at 0 × 0 (the
    // 2026-09-22 「图片没有正确显示」 reports). Give it the wrapper's full width.
    if (el.tagName === 'IMG') {
      var fit = function () {
        if (el.naturalWidth > 0 && el.getBoundingClientRect().width < 2) { w.style.width = '100%'; el.style.width = '100%'; }
      };
      if (el.complete) fit(); else el.addEventListener('load', fit, { once: true });
    }
    return w;
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
      var strip = tools(fig, media(img));
      strip.appendChild(zoomButton(img));
      strip.appendChild(figureButton(cap));
    });
  }

  function decorateDiagrams() {
    Array.prototype.forEach.call(container.querySelectorAll('.mermaid[data-mermaid-source]'), function (d) {
      var svg = d.querySelector(':scope > svg, :scope > .fig-media > svg');
      if (!svg && !d.classList.contains('mermaid-error')) return; // still rendering
      // Mermaid gives the svg width=100% + max-width=<natural>px; the wrapper takes that width
      if (svg && !svg.parentNode.classList.contains('fig-media')) media(svg, svg.style.maxWidth);
      var cap = d.nextElementSibling;
      if (!cap || !cap.classList.contains('post-figcaption')) {
        cap = caption(figureNo(d), mermaidTitle(d.getAttribute('data-mermaid-source')));
        d.parentNode.insertBefore(cap, d.nextSibling);
      }
      var strip = tools(d, svg && svg.parentNode);
      if (strip.querySelector('.fig-feedback')) return;
      d.classList.add('code-copy-anchor');
      if (svg) strip.appendChild(zoomButton(svg));
      strip.appendChild(figureButton(cap));
    });
  }

  var TABLE_EXCLUDE = '.comment, .annotation-panel, .series-toc, .related-posts';
  function tableAnchor(table) { return table.parentNode.classList.contains('table-responsive') ? table.parentNode : table; }

  // Ordinal of a table among the article's tables (own sequence: 表 N).
  function tableNo(table) {
    var all = container.querySelectorAll('table'), n = 0;
    for (var i = 0; i < all.length; i++) {
      if (all[i].closest(TABLE_EXCLUDE)) continue;
      n++;
      if (all[i] === table) return n;
    }
    return n + 1;
  }

  // Title of a table: its <caption> (_plugins/table_captions.rb folds a
  // `Table: …` / `表：…` paragraph after the table into one at build time; the
  // same paragraph is also read here for pages the plugin does not cover).
  // The source node is removed — the caption rendered under the table
  // replaces it, inline markup kept — and any hand-written number is dropped.
  var TITLE_PREFIX = /^\s*(Table|表)\s*\d*\s*[:：]\s*/i;
  function tableTitle(table) {
    var src = table.querySelector(':scope > caption');
    if (!src) {
      var next = tableAnchor(table).nextElementSibling;
      if (next && next.tagName === 'P' && TITLE_PREFIX.test(next.textContent) && norm(next.textContent).replace(TITLE_PREFIX, '')) src = next;
    }
    if (!src) return null;
    src.parentNode.removeChild(src);
    var first = src.firstChild;
    if (first && first.nodeType === 3) first.nodeValue = first.nodeValue.replace(TITLE_PREFIX, '');
    return src;
  }

  var TABLE_TITLE = '对这张表评论（会选中表格标题，再从工具条里选）';
  function decorateTables() {
    Array.prototype.forEach.call(container.querySelectorAll('table'), function (table) {
      if (table.closest(TABLE_EXCLUDE)) return;
      var anchor = tableAnchor(table);
      var cap = anchor.nextElementSibling;
      if (!cap || !cap.classList.contains('table-caption')) {
        var src = tableTitle(table);
        cap = caption(tableNo(table), src ? norm(src.textContent) : '', '表');
        cap.classList.add('table-caption');
        if (src) { var t = cap.querySelector('.fig-title'); t.textContent = ''; while (src.firstChild) t.appendChild(src.firstChild); }
        anchor.parentNode.insertBefore(cap, anchor.nextSibling);
      }
      // the strip floats on the table's top-right corner (code-copy.js put it there), hover-revealed
      var tools = anchor.querySelector(':scope > .table-tools');
      if (!tools || tools.querySelector('.table-feedback')) return;
      // Untitled: 「表 N」 renumbers when a table is inserted, so the header row
      // is the stable passage; only a table without a header uses the whole table.
      var target = cap.querySelector('.fig-title') || table.querySelector('thead > tr') || table;
      var feedback = button(TABLE_TITLE, function () { pick(target, target.closest('.post-figcaption') || target); });
      feedback.classList.add('table-feedback');
      tools.appendChild(feedback);
    });
  }

  // Fenced blocks get a header bar (VitePress style) instead of a caption:
  //   <div class="code-block [highlighter-rouge]" data-title?>
  //     <div class="code-header"><span class="fig-no">代码块 N：</span><span class="fig-title">…</span>
  //       <span class="code-lang">cpp</span><div class="fig-tools">copy · feedback</div></div>
  //     <div class="highlight"><pre>… | <pre>…
  // 「代码块 N」 for a fence with a language, 「文本块 N」 for an untyped / ```text
  // one (shell output, logs, trees); separate sequences. The title comes from
  // ```lang title="…" (_plugins/code_titles.rb → data-title on the rouge
  // wrapper or the bare <pre>) and is the passage the feedback button picks —
  // stable across edits to the code, readable in the comment. Untitled blocks
  // fall back to the block number in the header (「代码块 N」 / 「文本块 N」),
  // not the first line: this keeps the selection on the block's own metadata,
  // matching figures/tables. Authors should add a title for stable anchors.
  var TEXT_LANGS = { text: 1, txt: 1, plain: 1, plaintext: 1 };
  var CODE_EXCLUDE = '.mermaid, .comment, .annotation-panel, .series-toc, .related-posts';
  function codeLang(anchor, code) {
    var m = /(?:^|\s)language-([\w+#-]+)/.exec(anchor.className + ' ' + code.className);
    return m ? m[1].toLowerCase() : '';
  }
  function codeKind(lang) { return !lang || TEXT_LANGS[lang] ? '文本块' : '代码块'; }
  // Ordinal of a block among the article's blocks of the same kind. The panels
  // of a code-tabs group (js/code-tabs.js) are one block in several languages
  // and share one number.
  function codeNo(anchor, kind) {
    var all = container.querySelectorAll('.code-block'), n = 0, group = anchor.closest('.code-tabs');
    for (var i = 0; i < all.length; i++) {
      var b = all[i], g = b.closest('.code-tabs');
      if (b.getAttribute('data-kind') !== kind) continue;
      if (g && g === group) return n + 1;
      if (g && i && all[i - 1].closest('.code-tabs') === g) continue;
      n++;
      if (b === anchor) return n;
    }
    return n + 1;
  }
  var CODE_TITLE = '对这段代码评论（会选中它的标题或「代码块 N」，再从工具条里选）';
  function decorateCode() {
    Array.prototype.forEach.call(container.querySelectorAll('pre'), function (pre) {
      if (pre.closest(CODE_EXCLUDE) || pre.closest('.code-block') || pre.querySelector('code.language-mermaid')) return;
      var code = pre.querySelector('code') || pre;
      var anchor = pre.closest('.highlighter-rouge');
      if (!anchor) {   // untyped fence: a bare <pre>, wrapped so the header can sit above it
        anchor = document.createElement('div');
        pre.parentNode.insertBefore(anchor, pre);
        anchor.appendChild(pre);
        if (pre.hasAttribute('data-title')) anchor.setAttribute('data-title', pre.getAttribute('data-title'));
      }
      var lang = codeLang(anchor, code), kind = codeKind(lang);
      anchor.classList.add('code-block', 'code-copy-anchor');
      anchor.setAttribute('data-kind', kind);
      var title = norm(anchor.getAttribute('data-title'));
      var head = document.createElement('div');
      head.className = 'code-header' + (title ? '' : ' is-untitled');
      var num = document.createElement('span'); num.className = 'fig-no'; num.textContent = kind + ' ' + codeNo(anchor, kind) + (title ? '：' : '');
      head.appendChild(num);
      if (title) { var t = document.createElement('span'); t.className = 'fig-title'; t.textContent = title; head.appendChild(t); }
      if (lang && kind === '代码块') { var l = document.createElement('span'); l.className = 'code-lang'; l.textContent = lang; head.appendChild(l); }
      anchor.insertBefore(head, anchor.firstChild);
      var strip = tools(anchor, head);
      // code-copy.js put its button on the wrapper, or inside the bare <pre>
      Array.prototype.forEach.call(pre.querySelectorAll(':scope > .code-copy'), function (c) { strip.insertBefore(c, strip.firstChild); });
      var target = head.querySelector('.fig-title') || head.querySelector('.fig-no') || code;
      strip.appendChild(button(CODE_TITLE, function () { pick(target, head); }));
    });
  }

  // 随笔 小视频: full-screen black player (朋友圈); tap outside the video / Esc / × closes.
  function playVideo(src) {
    if (!src) return;
    var box = document.createElement('div');
    box.className = 'moment-player';
    box.innerHTML = '<video controls autoplay playsinline></video><button type="button" class="moment-player-close" aria-label="关闭">×</button>';
    var v = box.querySelector('video');
    v.src = src;
    document.body.appendChild(box);
    document.documentElement.classList.add('moment-playing');
    var played = v.play();
    if (played && played.catch) played.catch(function () {});
    function onKey(e) { if (e.key === 'Escape') close(); }
    function close() {
      v.pause(); v.removeAttribute('src'); box.remove();
      document.documentElement.classList.remove('moment-playing');
      document.removeEventListener('keydown', onKey);
    }
    box.addEventListener('click', function (e) { if (e.target !== v) close(); });
    document.addEventListener('keydown', onKey);
  }

  function init() {
    container = document.querySelector('.post-container');
    if (!container) return;
    // 随笔 (/moments/): pictures sit in a 朋友圈 grid — no 「图 N」 caption, no
    // tool strip; a tap opens the zoom overlay (the link is the no-JS fallback).
    if (container.classList.contains('moments')) {
      container.addEventListener('click', function (e) {
        var pic = e.target.closest('.moment-pic');
        if (pic && pic.classList.contains('moment-video')) { e.preventDefault(); return playVideo(pic.getAttribute('href')); }
        if (!pic || !window.DiagramZoom) return;
        e.preventDefault();
        var gallery = pic.closest('.moment-gallery');
        var pictures = gallery ? Array.prototype.slice.call(gallery.querySelectorAll('.moment-pic')) : [pic];
        var list = pictures.map(function (item) { return { src: item.getAttribute('href') }; });
        window.DiagramZoom.open(pic.querySelector('img'), { list: list, index: pictures.indexOf(pic) });
      });
    } else decorateImages();
    decorateDiagrams();
    decorateCode();
    decorateTables();
    if (window.MutationObserver) {
      new MutationObserver(function () { decorateDiagrams(); }).observe(container, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
