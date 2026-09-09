/**
 * annotations.js — highlight annotations ("划线批注") on blog posts.
 *
 * Readers select any passage of the article, click "评论" in the floating
 * toolbar and write a Markdown note in an inline composer. The note is posted
 * as a normal comment of the post's giscus / GitHub Discussions thread, in
 * the form
 *
 *     > quoted passage
 *     >
 *     > <sub>[§ 原文位置](https://arganzheng.life/<slug>.html#annot-<hash>:~:text=prefix-,start,end,-suffix)</sub>
 *
 *     the note
 *
 * so it reads naturally on GitHub too. On load, the thread is fetched through
 * the secret-free relay in tools/annotations-worker/, comments of that shape
 * are parsed back into a W3C TextQuoteSelector { exact, prefix, suffix } and
 * anchored in the article — exactly first, then fuzzily (approx-string-match,
 * the algorithm Hypothesis uses) so highlights survive small edits — and
 * rendered as <mark> elements with a hover/tap card (window.InlinePopover
 * from js/inline-popups.js).
 *
 * Posting reuses the reader's giscus session (GitHub login stored by giscus'
 * client.js in localStorage["giscus-session"]): the relay exchanges it for a
 * GitHub token and the browser calls GitHub's GraphQL API directly, the same
 * way the giscus iframe does. If anything in that path fails we fall back to
 * copying the quote to the clipboard so it can be pasted into giscus.
 *
 * Config comes from data-* attributes on `section.comment` (see
 * _includes/comments.html); `localStorage.annotationsApi` overrides the API
 * base for local development.
 */

(function () {
  'use strict';

  var GISCUS_ORIGIN = 'https://giscus.app';
  var GITHUB_GRAPHQL = 'https://api.github.com/graphql';
  var GITHUB_MARKDOWN = 'https://api.github.com/markdown';
  var SESSION_KEY = 'giscus-session';
  var CONTEXT_CHARS = 32;
  var FRAGMENT_SPLIT_AT = 150;
  var FRAGMENT_EDGE_CHARS = 20;
  var EXCLUDE_SELECTOR = '.comment, .pager, .related-posts, .share, .footnotes, .reversefootnote, sup[id^="fnref"], a.footnote, ' +
    'script, style, noscript, svg, .katex, .mermaid, button, .anchorjs-link, .annotation-toolbar, .annotation-composer';
  var HOVER_OWNERS = '.inline-tip, sup.has-popup-footnote'; // hover belongs to these; annotation card is click-only inside them
  var GHOST = { login: 'ghost', url: 'https://github.com/ghost', avatarUrl: 'https://avatars.githubusercontent.com/u/10137?s=64&v=4' };

  var cfg = null;
  var container = null;
  var index = null;          // { text, nodes:[{node, charIdx:[]}] }
  var annotations = [];      // parsed + anchored annotations
  var discussion = null;     // { id, url, totalCommentCount }
  var token = null;
  var viewer = null;
  var toolbar = null;
  var composer = null;
  var toast = null;
  var pendingSelector = null;
  var pendingOffsets = null;
  var selectionTimer = null;

  // ------------------------------------------------------------------ init

  function init() {
    var section = document.querySelector('section.comment[data-annotations-api]');
    container = document.querySelector('.post-container');
    if (!section || !container || !window.InlinePopover) return;

    var api = (localStorage.getItem('annotationsApi') || section.getAttribute('data-annotations-api') || '').replace(/\/$/, '');
    if (!api) return;

    cfg = {
      api: api,
      path: section.getAttribute('data-page-path') || location.pathname,
      siteUrl: (section.getAttribute('data-site-url') || location.origin).replace(/^http:/, 'https:').replace(/\/$/, ''),
      repoId: section.getAttribute('data-repo-id') || '',
      categoryId: section.getAttribute('data-category-id') || '',
      section: section
    };

    bindSelection();
    bindGiscusMessages();
    whenRichContentSettled(function () {
      loadAnnotations(false).then(function () { restoreDraft(); focusFromHash(); });
    });
  }

  // Mermaid / KaTeX rewrite the DOM asynchronously; anchor after they finish
  // (they fire richcontent:rendered), or straight away when there is nothing
  // to render. Re-anchor on every later event as well.
  function whenRichContentSettled(fn) {
    var needsWait = container.querySelector('.mermaid, pre code.language-mermaid') || hasRawMath(container);
    var done = false;
    function run() { if (!done) { done = true; fn(); } }
    document.addEventListener('richcontent:rendered', function () {
      if (done) reanchorAll(); else run();
    });
    if (!needsWait) run(); else setTimeout(run, 6000);
  }

  function hasRawMath(root) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    while (walker.nextNode()) {
      var t = walker.currentNode.nodeValue;
      if ((t.indexOf('\\(') !== -1 || t.indexOf('\\[') !== -1) && !walker.currentNode.parentNode.closest('pre, code')) return true;
    }
    return false;
  }

  // ------------------------------------------------------------ text index

  function isExcluded(node) {
    var el = node.nodeType === 1 ? node : node.parentNode;
    return !!(el && el.closest && el.closest(EXCLUDE_SELECTOR));
  }

  // `gaps` records normalised positions preceded by excluded content (e.g. a
  // footnote marker) — the browser's own Text Fragment matcher *does* see that
  // content, so fragments must not span a gap.
  function buildIndex() {
    var text = '';
    var nodes = [];
    var gaps = {};
    var lastWasSpace = true;
    var gapPending = false;
    var walker = document.createTreeWalker(container, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        if (n.nodeType === 1) {
          if (n.matches(EXCLUDE_SELECTOR)) { gapPending = true; return NodeFilter.FILTER_REJECT; }
          return NodeFilter.FILTER_SKIP;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    while (walker.nextNode()) {
      var node = walker.currentNode;
      var value = node.nodeValue;
      var charIdx = new Array(value.length);
      for (var i = 0; i < value.length; i++) {
        var ch = value.charAt(i);
        if (/\s/.test(ch)) {
          if (lastWasSpace) { charIdx[i] = -1; continue; }
          ch = ' ';
          lastWasSpace = true;
        } else {
          lastWasSpace = false;
        }
        if (gapPending) { gaps[text.length] = true; gapPending = false; }
        charIdx[i] = text.length;
        text += ch;
      }
      nodes.push({ node: node, charIdx: charIdx });
    }
    index = { text: text, nodes: nodes, gaps: gaps };
    return index;
  }

  // Normalised index range of a DOM Range (or null if it is outside the article).
  function rangeToOffsets(range) {
    if (!index) buildIndex();
    var start = -1, end = -1;
    for (var n = 0; n < index.nodes.length; n++) {
      var entry = index.nodes[n];
      if (!range.intersectsNode(entry.node)) continue;
      var node = entry.node, len = node.nodeValue.length;
      if (start === -1) {
        for (var k = 0; k <= len; k++) {
          if (range.comparePoint(node, k) >= 0) {
            start = firstIndexAtOrAfter(entry, k);
            break;
          }
        }
      }
      for (var m = len; m >= 0; m--) {
        if (range.comparePoint(node, m) <= 0) {
          var e = lastIndexBefore(entry, m);
          if (e !== -1) end = e + 1;
          break;
        }
      }
    }
    if (start === -1 || end === -1 || end <= start) return null;
    // trim surrounding whitespace
    while (start < end && index.text.charAt(start) === ' ') start++;
    while (end > start && index.text.charAt(end - 1) === ' ') end--;
    return end > start ? { start: start, end: end } : null;
  }

  function firstIndexAtOrAfter(entry, k) {
    for (var i = k; i < entry.charIdx.length; i++) if (entry.charIdx[i] !== -1) return entry.charIdx[i];
    // node ends before any kept char: next kept char in the document
    var pos = index.nodes.indexOf(entry);
    for (var n = pos + 1; n < index.nodes.length; n++) {
      var c = index.nodes[n].charIdx;
      for (var j = 0; j < c.length; j++) if (c[j] !== -1) return c[j];
    }
    return index.text.length;
  }

  function lastIndexBefore(entry, m) {
    for (var i = m - 1; i >= 0; i--) if (entry.charIdx[i] !== -1) return entry.charIdx[i];
    var pos = index.nodes.indexOf(entry);
    for (var n = pos - 1; n >= 0; n--) {
      var c = index.nodes[n].charIdx;
      for (var j = c.length - 1; j >= 0; j--) if (c[j] !== -1) return c[j];
    }
    return -1;
  }

  function selectorFromOffsets(o) {
    var t = index.text;
    return {
      exact: t.slice(o.start, o.end),
      prefix: t.slice(Math.max(0, o.start - CONTEXT_CHARS), o.start),
      suffix: t.slice(o.end, o.end + CONTEXT_CHARS)
    };
  }

  // ---------------------------------------------------------------- anchor

  function contextScore(text, start, end, sel) {
    var score = 0;
    var p = sel.prefix || '', s = sel.suffix || '';
    for (var i = 1; i <= p.length && start - i >= 0; i++) {
      if (text.charAt(start - i) === p.charAt(p.length - i)) score++; else break;
    }
    for (var j = 0; j < s.length && end + j < text.length; j++) {
      if (text.charAt(end + j) === s.charAt(j)) score++; else break;
    }
    return score;
  }

  function anchor(sel) {
    if (!index) buildIndex();
    var text = index.text;
    var exact = (sel.exact || '').replace(/\s+/g, ' ').trim();
    if (!exact) return null;

    var hits = [], from = 0, at;
    while ((at = text.indexOf(exact, from)) !== -1) { hits.push(at); from = at + 1; }
    if (hits.length) {
      var best = hits[0], bestScore = -1;
      for (var i = 0; i < hits.length; i++) {
        var sc = contextScore(text, hits[i], hits[i] + exact.length, sel);
        if (sc > bestScore) { bestScore = sc; best = hits[i]; }
      }
      return { start: best, end: best + exact.length, fuzzy: false };
    }

    if (exact.length < 4 || typeof window.approxStringMatch !== 'function') return null;
    var maxErrors = Math.min(40, Math.max(2, Math.round(exact.length * 0.2)));
    var matches = window.approxStringMatch(text, exact, maxErrors);
    if (!matches.length) return null;
    var chosen = null, chosenScore = -Infinity;
    for (var m = 0; m < matches.length; m++) {
      var mt = matches[m];
      var lengthPenalty = Math.abs((mt.end - mt.start) - exact.length);
      var score = -mt.errors * 3 - lengthPenalty + contextScore(text, mt.start, mt.end, sel);
      if (score > chosenScore) { chosenScore = score; chosen = mt; }
    }
    return chosen ? { start: chosen.start, end: chosen.end, fuzzy: true } : null;
  }

  // ------------------------------------------------------------- highlight

  // Character offsets of [start, end) inside each text node of the current index.
  function segmentsFor(start, end) {
    var segs = [];
    for (var n = 0; n < index.nodes.length; n++) {
      var entry = index.nodes[n], c = entry.charIdx, from = -1, to = -1;
      for (var i = 0; i < c.length; i++) {
        if (c[i] !== -1 && c[i] >= start && c[i] < end) { if (from === -1) from = i; to = i + 1; }
      }
      if (from !== -1) segs.push({ node: entry.node, from: from, to: to });
    }
    return segs;
  }

  function unwrapAll() {
    var marks = container.querySelectorAll('mark.annotation-hl');
    for (var i = 0; i < marks.length; i++) {
      var m = marks[i], parent = m.parentNode;
      while (m.firstChild) parent.insertBefore(m.firstChild, m);
      parent.removeChild(m);
    }
    container.normalize();
  }

  // One pass over a single index: every text node is cut at all highlight
  // boundaries and each piece gets ONE <mark> carrying every annotation id that
  // covers it (overlaps therefore never nest). Pieces are wrapped back-to-front
  // so splitText() never invalidates offsets still to be processed.
  function applyHighlights() {
    unwrapAll();
    buildIndex();
    var orphans = [], perNode = [], byNode = new Map();
    for (var i = 0; i < annotations.length; i++) {
      var a = annotations[i];
      a.range = anchor(a.selector);
      a.marks = [];
      if (!a.range) { orphans.push(a); continue; }
      var segs = segmentsFor(a.range.start, a.range.end);
      for (var s = 0; s < segs.length; s++) {
        var bucket = byNode.get(segs[s].node);
        if (!bucket) { bucket = { node: segs[s].node, segs: [] }; byNode.set(segs[s].node, bucket); perNode.push(bucket); }
        bucket.segs.push({ from: segs[s].from, to: segs[s].to, id: a.id });
      }
    }
    for (var n = 0; n < perNode.length; n++) {
      var node = perNode[n].node, nodeSegs = perNode[n].segs;
      var bounds = [];
      nodeSegs.forEach(function (sg) { if (bounds.indexOf(sg.from) === -1) bounds.push(sg.from); if (bounds.indexOf(sg.to) === -1) bounds.push(sg.to); });
      bounds.sort(function (x, y) { return x - y; });
      for (var b = bounds.length - 2; b >= 0; b--) {
        var from = bounds[b], to = bounds[b + 1];
        var ids = nodeSegs.filter(function (sg) { return sg.from <= from && sg.to >= to; }).map(function (sg) { return sg.id; });
        if (!ids.length) continue;
        var piece = node;
        if (to < piece.nodeValue.length) piece.splitText(to);
        if (from > 0) piece = piece.splitText(from);
        if (!piece.nodeValue.trim()) continue;
        var mark = document.createElement('mark');
        mark.className = 'annotation-hl' + (ids.length > 1 ? ' is-multi' : '');
        mark.setAttribute('data-annotation-ids', ids.join(' '));
        piece.parentNode.insertBefore(mark, piece);
        mark.appendChild(piece);
        bindMark(mark);
        ids.forEach(function (id) { var an = findAnnotation(id); if (an) an.marks.push(mark); });
      }
    }
    annotations.forEach(function (an) {
      an.marks.sort(function (x, y) { return x.compareDocumentPosition(y) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1; });
    });
    buildIndex();
    renderOrphans(orphans);
  }

  function reanchorAll() {
    if (annotations.length) applyHighlights();
    else buildIndex();
  }

  function bindMark(mark) {
    var hoverOwner = mark.closest(HOVER_OWNERS);
    var isLink = !!mark.closest('a[href]');
    if (!hoverOwner) {
      mark.addEventListener('mouseenter', function () {
        InlinePopover.scheduleShow(mark, function () { return buildCard(idsFor(mark)); }, { className: 'annotation-card', delay: 250 });
      });
      mark.addEventListener('mouseleave', function () { InlinePopover.scheduleHide(); });
    }
    mark.addEventListener('click', function (e) {
      if (isLink && !hoverOwner) return; // let external links navigate; hover shows the card
      e.preventDefault();
      e.stopPropagation();
      if (InlinePopover.isActiveFor(mark)) InlinePopover.hide();
      else InlinePopover.show(mark, buildCard(idsFor(mark)), { className: 'annotation-card' });
    });
  }

  function idsFor(mark) {
    var ids = [], el = mark;
    while (el && el !== container) {
      if (el.nodeType === 1 && el.classList.contains('annotation-hl')) {
        var own = (el.getAttribute('data-annotation-ids') || '').split(' ');
        for (var i = 0; i < own.length; i++) if (own[i] && ids.indexOf(own[i]) === -1) ids.push(own[i]);
      }
      el = el.parentNode;
    }
    return ids;
  }

  function findAnnotation(id) {
    for (var i = 0; i < annotations.length; i++) if (annotations[i].id === id) return annotations[i];
    return null;
  }

  // #annot-<hash> (the part of the permalink before `:~:`) -> scroll, flash, open card.
  function focusFromHash() {
    var m = /^#annot-([0-9a-f]{8})/.exec(location.hash || '');
    if (!m) return;
    for (var i = 0; i < annotations.length; i++) {
      var a = annotations[i];
      if (annotHash(a.selector.exact) !== m[1] || !a.marks.length) continue;
      // Instant scroll (the browser may already have jumped for the Text
      // Fragment), then open the card once layout has settled.
      var mark = a.marks[0];
      window.scrollTo({ top: Math.max(0, mark.getBoundingClientRect().top + window.pageYOffset - 160), behavior: 'instant' });
      flashMarks(a.marks, true);
      setTimeout(function () { InlinePopover.show(mark, buildCard(idsFor(mark)), { className: 'annotation-card' }); }, 200);
      return;
    }
  }

  function flashMarks(marks, noScroll) {
    for (var i = 0; i < marks.length; i++) {
      marks[i].classList.add('is-new');
      (function (m) { setTimeout(function () { m.classList.remove('is-new'); }, 2500); })(marks[i]);
    }
    if (marks[0] && !noScroll) InlinePopover.scrollToTargetWithOffset(marks[0]);
  }

  // ------------------------------------------------------------------ card

  function authorHtml(author) {
    return '<a class="annotation-author" href="' + escapeAttr(author.url) + '" target="_blank" rel="noopener noreferrer">' +
      '<img src="' + escapeAttr(author.avatarUrl) + '" alt="" width="20" height="20">' +
      '<span>' + escapeHtml(author.login) + '</span></a>';
  }

  function timeHtml(iso) {
    return '<time datetime="' + escapeAttr(iso) + '" title="' + escapeAttr(new Date(iso).toLocaleString()) + '">' + relativeTime(iso) + '</time>';
  }

  function buildCard(ids) {
    var wrap = document.createElement('div');
    wrap.className = 'annotation-list';
    for (var i = 0; i < ids.length; i++) {
      var a = findAnnotation(ids[i]);
      if (a) wrap.appendChild(buildItem(a));
    }
    if (!wrap.childNodes.length) wrap.textContent = '批注已删除或不可用。';
    return wrap;
  }

  function buildItem(a) {
    var item = document.createElement('div');
    item.className = 'annotation-item';
    item.setAttribute('data-annotation-id', a.id);
    var replies = a.replies || [];
    item.innerHTML =
      '<div class="annotation-meta">' + authorHtml(a.author) + timeHtml(a.createdAt) + '</div>' +
      '<div class="annotation-note"></div>' +
      '<div class="annotation-actions">' +
        (a.upvoteCount ? '<span class="annotation-upvotes" title="赞同"><i class="fa fa-caret-up"></i> ' + a.upvoteCount + '</span>' : '') +
        (replies.length ? '<button type="button" class="annotation-toggle-replies"><i class="fa fa-comments-o"></i> ' + replies.length + ' 条回复</button>' : '') +
        '<button type="button" class="annotation-reply-btn"><i class="fa fa-reply"></i> 回复</button>' +
        '<a href="' + escapeAttr(a.url) + '" target="_blank" rel="noopener noreferrer" title="在 GitHub 上查看"><i class="fa fa-github"></i></a>' +
      '</div>' +
      '<div class="annotation-replies" style="display:none"></div>' +
      '<div class="annotation-reply-box" style="display:none"></div>';
    item.querySelector('.annotation-note').appendChild(sanitizeHtml(a.noteHTML));

    var repliesEl = item.querySelector('.annotation-replies');
    replies.forEach(function (r) {
      var el = document.createElement('div');
      el.className = 'annotation-reply';
      el.innerHTML = '<div class="annotation-meta">' + authorHtml(r.author) + timeHtml(r.createdAt) + '</div><div class="annotation-note"></div>';
      el.querySelector('.annotation-note').appendChild(sanitizeHtml(r.bodyHTML));
      repliesEl.appendChild(el);
    });
    var toggle = item.querySelector('.annotation-toggle-replies');
    if (toggle) toggle.addEventListener('click', function () {
      var open = repliesEl.style.display !== 'none';
      repliesEl.style.display = open ? 'none' : 'block';
      toggle.classList.toggle('is-open', !open);
    });
    item.querySelector('.annotation-reply-btn').addEventListener('click', function () {
      var box = item.querySelector('.annotation-reply-box');
      if (box.style.display !== 'none') { box.style.display = 'none'; return; }
      renderReplyBox(box, a);
      box.style.display = 'block';
      var ta = box.querySelector('textarea');
      if (ta) ta.focus();
    });
    return item;
  }

  function renderReplyBox(box, a) {
    box.innerHTML = '';
    if (!getSession()) {
      box.innerHTML = '<button type="button" class="annotation-login"><i class="fa fa-github"></i> 使用 GitHub 登录后回复</button>';
      box.querySelector('.annotation-login').addEventListener('click', login);
      return;
    }
    box.innerHTML =
      '<textarea rows="2" placeholder="回复 @' + escapeAttr(a.author.login) + '…（支持 Markdown）"></textarea>' +
      '<div class="annotation-reply-actions"><span class="annotation-reply-status"></span>' +
      '<button type="button" class="annotation-reply-send"><i class="fa fa-paper-plane"></i> 发送</button></div>';
    var ta = box.querySelector('textarea');
    var send = box.querySelector('.annotation-reply-send');
    var status = box.querySelector('.annotation-reply-status');
    function doSend() {
      var text = ta.value.trim();
      if (!text) return;
      send.disabled = true; ta.disabled = true; status.textContent = '发送中…';
      graphql(ADD_COMMENT, { body: text, discussionId: discussion.id, replyToId: a.id }).then(function (data) {
        var c = data.addDiscussionComment.comment;
        a.replies = (a.replies || []).concat([{ id: c.id, url: c.url, author: c.author, createdAt: c.createdAt, bodyHTML: c.bodyHTML }]);
        a.replyCount = a.replies.length;
        var trigger = InlinePopover.currentTrigger();
        if (trigger) {
          InlinePopover.show(trigger, buildCard(idsFor(trigger)), { className: 'annotation-card' });
          var item = InlinePopover.element().querySelector('[data-annotation-id="' + a.id + '"]');
          var t = item && item.querySelector('.annotation-toggle-replies');
          if (t) t.click();
        }
        showToast('回复已发送');
        refreshGiscus();
      }).catch(function (err) {
        send.disabled = false; ta.disabled = false;
        status.textContent = err.message || String(err);
        if (!getSession()) renderReplyBox(box, a);
      });
    }
    send.addEventListener('click', doSend);
    ta.addEventListener('keydown', function (e) { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); doSend(); } });
  }

  function sanitizeHtml(html) {
    var doc = new DOMParser().parseFromString('<div>' + (html || '') + '</div>', 'text/html');
    var root = doc.body.firstChild;
    var bad = root.querySelectorAll('script, style, iframe, object, embed, link, meta, form, input, textarea, button');
    for (var i = 0; i < bad.length; i++) bad[i].parentNode.removeChild(bad[i]);
    var all = root.querySelectorAll('*');
    for (var j = 0; j < all.length; j++) {
      var el = all[j], attrs = el.attributes;
      for (var k = attrs.length - 1; k >= 0; k--) {
        var name = attrs[k].name, value = attrs[k].value;
        if (/^on/i.test(name) || ((name === 'href' || name === 'src') && /^\s*javascript:/i.test(value))) el.removeAttribute(name);
      }
      if (el.tagName === 'A') { el.setAttribute('target', '_blank'); el.setAttribute('rel', 'noopener noreferrer nofollow'); }
    }
    var frag = document.createDocumentFragment();
    while (root.firstChild) frag.appendChild(document.adoptNode(root.firstChild));
    return frag;
  }

  function renderOrphans(orphans) {
    var old = cfg.section.querySelector('.annotation-orphans');
    if (old) old.parentNode.removeChild(old);
    if (!orphans.length) return;
    var box = document.createElement('p');
    box.className = 'annotation-orphans';
    box.innerHTML = '<i class="fa fa-unlink"></i> ' + orphans.length + ' 条划线批注未能定位到原文（原文可能已修改）：';
    for (var i = 0; i < orphans.length; i++) {
      var a = document.createElement('a');
      a.href = orphans[i].url; a.target = '_blank'; a.rel = 'noopener noreferrer';
      a.textContent = '@' + orphans[i].author.login;
      a.title = orphans[i].selector.exact;
      box.appendChild(a);
      if (i < orphans.length - 1) box.appendChild(document.createTextNode('、'));
    }
    var hint = cfg.section.querySelector('.comment-hint');
    (hint || cfg.section).insertAdjacentElement(hint ? 'afterend' : 'afterbegin', box);
  }

  // --------------------------------------------------------- data loading

  function api(path, opts) {
    opts = opts || {};
    var headers = opts.headers || {};
    if (opts.body) headers['Content-Type'] = 'application/json';
    return fetch(cfg.api + path, {
      method: opts.method || 'GET',
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined
    }).then(function (r) {
      return r.json().then(function (data) {
        if (!r.ok) throw new Error((data && data.error) || ('HTTP ' + r.status));
        return data;
      });
    });
  }

  function loadAnnotations(fresh) {
    var qs = '?term=' + encodeURIComponent(cfg.path) + (fresh ? '&t=' + Date.now() : '');
    return api('/discussions' + qs).then(function (data) {
      var d = data && data.discussion;
      discussion = d ? { id: d.id, url: d.url, totalCommentCount: d.totalCommentCount } : null;
      annotations = d ? parseComments(d.comments || []) : [];
      applyHighlights();
      return annotations;
    }).catch(function (err) {
      console.warn('[annotations] load failed:', err.message);
      buildIndex();
      return [];
    });
  }

  function parseComments(comments) {
    var out = [];
    for (var i = 0; i < comments.length; i++) {
      var a = parseComment(comments[i]);
      if (a) out.push(a);
    }
    return out;
  }

  function parseComment(c) {
    if (!c || c.deletedAt || c.isMinimized || !c.bodyHTML) return null;
    var doc = new DOMParser().parseFromString('<div>' + c.bodyHTML + '</div>', 'text/html');
    var root = doc.body.firstChild;
    var quote = root.firstElementChild;
    if (!quote || quote.tagName !== 'BLOCKQUOTE') return null;
    var links = quote.querySelectorAll('a[href*=":~:text="]');
    var link = null;
    for (var i = 0; i < links.length; i++) {
      try {
        var u = new URL(links[i].getAttribute('href'));
        if (u.pathname === cfg.path) { link = links[i]; break; }
      } catch (e) { /* ignore */ }
    }
    if (!link) return null;
    var fragment = parseTextFragment(link.getAttribute('href'));
    var linkBlock = link.closest('p, sub') || link;
    while (linkBlock.parentNode !== quote && linkBlock.parentNode !== root) linkBlock = linkBlock.parentNode;
    linkBlock.parentNode.removeChild(linkBlock);
    var exact = quote.textContent.replace(/\s+/g, ' ').trim();
    if (!exact) return null;
    root.removeChild(quote);
    return {
      id: c.id,
      url: c.url,
      author: c.author || GHOST,
      createdAt: c.createdAt,
      upvoteCount: c.upvoteCount || 0,
      replyCount: (c.replies && (c.replies.totalCount !== undefined ? c.replies.totalCount : c.replies.length)) || c.replyCount || 0,
      replies: parseReplies(c.replies),
      noteHTML: root.innerHTML,
      selector: { exact: exact, prefix: fragment.prefix, suffix: fragment.suffix }
    };
  }

  // giscus' adapter returns replies as a plain array; GitHub GraphQL as {nodes}.
  function parseReplies(replies) {
    var list = Array.isArray(replies) ? replies : (replies && replies.nodes) || [];
    return list.filter(function (r) { return r && !r.deletedAt && !r.isMinimized; }).map(function (r) {
      return { id: r.id, url: r.url, createdAt: r.createdAt, bodyHTML: r.bodyHTML, author: r.author || GHOST };
    });
  }

  function parseTextFragment(href) {
    var out = { prefix: '', suffix: '' };
    var m = /:~:text=([^&]*)/.exec(href);
    if (!m) return out;
    var parts = m[1].split(',');
    var dec = function (s) { try { return decodeURIComponent(s); } catch (e) { return s; } };
    if (parts.length && /-$/.test(parts[0])) out.prefix = dec(parts.shift().slice(0, -1));
    if (parts.length && /^-/.test(parts[parts.length - 1])) out.suffix = dec(parts.pop().slice(1));
    return out;
  }

  // ------------------------------------------------------- serialisation

  // Only escape what URL / Text Fragment / Markdown-link syntax needs; CJK and
  // other non-ASCII stay readable (browsers accept raw UTF-8 in fragments).
  function encodeFragmentPart(s) {
    return s.replace(/[\s%&,\-#()"'<>\[\]\\^`{}|]/g, function (c) {
      return '%' + ('0' + c.charCodeAt(0).toString(16).toUpperCase()).slice(-2);
    });
  }

  // Text Fragments match prefix/suffix on word boundaries, so drop a Latin word
  // that our fixed-width context window cut in half (CJK has no such boundary).
  function trimPartialWordStart(s) { return /^[A-Za-z0-9]/.test(s) && /\s/.test(s) ? s.replace(/^\S*\s+/, '') : s; }
  function trimPartialWordEnd(s) { return /[A-Za-z0-9]$/.test(s) && /\s/.test(s) ? s.replace(/\s+\S*$/, '') : s; }

  function countOccurrences(text, needle) {
    var n = 0, from = 0, at;
    while (needle && (at = text.indexOf(needle, from)) !== -1) { n++; from = at + 1; }
    return n;
  }

  // Split [start, end) of the index into runs that contain no gap (see buildIndex).
  function runsWithoutGaps(start, end) {
    var runs = [], runStart = start;
    for (var i = start + 1; i < end; i++) {
      if (index.gaps[i]) { runs.push([runStart, i]); runStart = i; }
    }
    runs.push([runStart, end]);
    return runs;
  }

  // `offsets` (index range of the quote) is known when the fragment is built
  // from a live selection; it lets us avoid gaps and drop redundant context.
  function buildTextFragment(sel, offsets) {
    var parts = [];
    var exact = sel.exact;
    var unique = index ? countOccurrences(index.text, exact) === 1 : false;
    var prefix = unique ? '' : trimPartialWordStart(sel.prefix || '').trim();
    var suffix = unique ? '' : trimPartialWordEnd(sel.suffix || '').trim();
    var runs = offsets && index ? runsWithoutGaps(offsets.start, offsets.end) : [[0, exact.length]];
    if (prefix) parts.push(encodeFragmentPart(prefix) + '-');
    if (runs.length === 1 && exact.length <= FRAGMENT_SPLIT_AT) {
      parts.push(encodeFragmentPart(exact));
    } else {
      var base = offsets ? offsets.start : 0;
      var first = runs[0], last = runs[runs.length - 1];
      var head = exact.slice(first[0] - base, Math.min(first[1], first[0] + FRAGMENT_EDGE_CHARS) - base).trim();
      var tail = exact.slice(Math.max(last[0], last[1] - FRAGMENT_EDGE_CHARS) - base, last[1] - base).trim();
      parts.push(encodeFragmentPart(head));
      if (tail && tail !== head) parts.push(encodeFragmentPart(tail));
    }
    if (suffix) parts.push('-' + encodeFragmentPart(suffix));
    return ':~:text=' + parts.join(',');
  }

  // Short stable id of a quote, used as the plain fragment (#annot-xxxxxxxx) in
  // front of the Text Fragment so our own script can locate the highlight
  // (browsers hide the `:~:` directive from location.hash). FNV-1a, 32 bit.
  function annotHash(exact) {
    var h = 0x811c9dc5;
    for (var i = 0; i < exact.length; i++) { h ^= exact.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    return ('0000000' + h.toString(16)).slice(-8);
  }

  function permalink(sel, offsets, withId) {
    return cfg.siteUrl + cfg.path + '#' + (withId ? 'annot-' + annotHash(sel.exact) : '') + buildTextFragment(sel, offsets);
  }

  function escapeMarkdown(s) {
    return s.replace(/[\\`*_\[\]<>~|]/g, '\\$&').replace(/^([#>+\-]|\d+\.)/, '\\$1');
  }

  function buildCommentBody(sel, note, offsets) {
    return '> ' + escapeMarkdown(sel.exact) + '\n>\n> <sub>[§ 原文位置](' + permalink(sel, offsets, true) + ')</sub>\n\n' + note.trim() + '\n';
  }

  // ------------------------------------------------------------ selection

  function bindSelection() {
    document.addEventListener('selectionchange', function () {
      clearTimeout(selectionTimer);
      selectionTimer = setTimeout(updateToolbar, 250);
    });
    document.addEventListener('mouseup', function () { setTimeout(updateToolbar, 10); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { hideToolbar(); closeComposer(); } });
    window.addEventListener('scroll', function () { if (toolbar && toolbar.style.display === 'block') positionToolbar(); }, { passive: true });
  }

  function currentRange() {
    var sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
    var range = sel.getRangeAt(0);
    if (!container.contains(range.commonAncestorContainer)) return null;
    if (isExcluded(range.startContainer) || isExcluded(range.endContainer)) return null;
    if (range.toString().trim().length < 2) return null;
    return range;
  }

  function updateToolbar() {
    if (composer && composer.style.display === 'block') return;
    var range = currentRange();
    if (!range) { hideToolbar(); return; }
    ensureToolbar();
    toolbar.style.display = 'block';
    positionToolbar();
  }

  function positionToolbar() {
    var range = currentRange();
    if (!range) return;
    var rect = range.getBoundingClientRect();
    var scrollX = window.pageXOffset, scrollY = window.pageYOffset;
    var w = toolbar.offsetWidth, h = toolbar.offsetHeight;
    var top = rect.top + scrollY - h - 10;
    toolbar.classList.toggle('is-below', rect.top - h - 10 < 60);
    if (rect.top - h - 10 < 60) top = rect.bottom + scrollY + 10;
    var left = rect.left + scrollX + rect.width / 2 - w / 2;
    left = Math.max(scrollX + 8, Math.min(left, scrollX + document.documentElement.clientWidth - w - 8));
    toolbar.style.top = top + 'px';
    toolbar.style.left = left + 'px';
  }

  function hideToolbar() { if (toolbar) toolbar.style.display = 'none'; }

  function ensureToolbar() {
    if (toolbar) return toolbar;
    toolbar = document.createElement('div');
    toolbar.className = 'annotation-toolbar';
    toolbar.innerHTML =
      '<button type="button" class="annotation-tb-comment"><i class="fa fa-comment-o"></i> 评论</button>' +
      '<button type="button" class="annotation-tb-link" title="复制指向这段文字的链接"><i class="fa fa-link"></i></button>' +
      '<span class="annotation-tb-arrow"></span>';
    toolbar.addEventListener('mousedown', function (e) { e.preventDefault(); }); // keep the selection
    toolbar.querySelector('.annotation-tb-comment').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      var offsets = range && rangeToOffsets(range);
      if (!offsets) { hideToolbar(); return; }
      var sel = selectorFromOffsets(offsets);
      hideToolbar();
      openComposer(sel, range.getBoundingClientRect(), '', offsets);
    });
    toolbar.querySelector('.annotation-tb-link').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      var offsets = range && rangeToOffsets(range);
      if (!offsets) return;
      copyText(permalink(selectorFromOffsets(offsets), offsets, false)).then(function () {
        showToast('链接已复制：打开它会直接滚动并高亮到这段文字，适合分享给别人');
      });
      hideToolbar();
    });
    document.body.appendChild(toolbar);
    return toolbar;
  }

  // ------------------------------------------------------------- composer

  function ensureComposer() {
    if (composer) return composer;
    composer = document.createElement('div');
    composer.className = 'annotation-composer';
    composer.setAttribute('role', 'dialog');
    composer.innerHTML =
      '<div class="ac-quote"><i class="fa fa-quote-left"></i><span class="ac-quote-text"></span></div>' +
      '<div class="ac-tabs"><button type="button" class="is-active" data-tab="write">撰写</button><button type="button" data-tab="preview">预览</button>' +
        '<span class="ac-tabs-hint">支持 Markdown</span></div>' +
      '<textarea class="ac-text" rows="4" placeholder="写下你对这段文字的批注…（可引用链接、代码、图片）"></textarea>' +
      '<div class="ac-preview" style="display:none"></div>' +
      '<div class="ac-status"></div>' +
      '<div class="ac-footer">' +
        '<span class="ac-user"></span>' +
        '<span class="ac-buttons">' +
          '<button type="button" class="ac-cancel">取消</button>' +
          '<button type="button" class="ac-login"><i class="fa fa-github"></i> 使用 GitHub 登录</button>' +
          '<button type="button" class="ac-submit"><i class="fa fa-paper-plane"></i> 发表</button>' +
        '</span>' +
      '</div>';
    composer.addEventListener('click', function (e) { e.stopPropagation(); });
    composer.addEventListener('mousedown', function (e) { e.stopPropagation(); });
    composer.querySelector('.ac-cancel').addEventListener('click', function () { clearDraft(); closeComposer(); });
    composer.querySelector('.ac-login').addEventListener('click', login);
    composer.querySelector('.ac-submit').addEventListener('click', submit);
    var ta = composer.querySelector('.ac-text');
    ta.addEventListener('input', saveDraft);
    ta.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
    });
    var tabs = composer.querySelectorAll('.ac-tabs button');
    for (var i = 0; i < tabs.length; i++) tabs[i].addEventListener('click', function () { switchTab(this.getAttribute('data-tab')); });
    document.body.appendChild(composer);
    return composer;
  }

  function openComposer(sel, rect, draftText, offsets) {
    ensureComposer();
    pendingSelector = sel;
    pendingOffsets = offsets || anchor(sel);
    InlinePopover.hide();
    composer.querySelector('.ac-quote-text').textContent = sel.exact;
    composer.querySelector('.ac-quote-text').title = sel.exact;
    composer.querySelector('.ac-text').value = draftText || '';
    setStatus('');
    switchTab('write');
    composer.style.display = 'block';
    positionComposer(rect);
    refreshAuthUI();
    if (window.getSelection) window.getSelection().removeAllRanges();
    composer.querySelector('.ac-text').focus();
    saveDraft();
  }

  function positionComposer(rect) {
    if (window.innerWidth <= 768 || !rect) { composer.style.top = composer.style.left = ''; composer.classList.add('is-sheet'); return; }
    composer.classList.remove('is-sheet');
    var scrollX = window.pageXOffset, scrollY = window.pageYOffset;
    var w = composer.offsetWidth;
    var left = rect.left + scrollX + rect.width / 2 - w / 2;
    left = Math.max(scrollX + 12, Math.min(left, scrollX + document.documentElement.clientWidth - w - 12));
    composer.style.left = left + 'px';
    composer.style.top = (rect.bottom + scrollY + 12) + 'px';
  }

  function closeComposer() {
    if (!composer) return;
    composer.style.display = 'none';
    pendingSelector = null;
    pendingOffsets = null;
  }

  function switchTab(tab) {
    var tabs = composer.querySelectorAll('.ac-tabs button');
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.toggle('is-active', tabs[i].getAttribute('data-tab') === tab);
    var ta = composer.querySelector('.ac-text'), pv = composer.querySelector('.ac-preview');
    ta.style.display = tab === 'write' ? '' : 'none';
    pv.style.display = tab === 'preview' ? 'block' : 'none';
    if (tab === 'preview') renderPreview(ta.value);
  }

  function renderPreview(md) {
    var pv = composer.querySelector('.ac-preview');
    if (!md.trim()) { pv.innerHTML = '<em class="ac-muted">没有内容可预览</em>'; return; }
    pv.innerHTML = '<em class="ac-muted">渲染中…</em>';
    var headers = { 'Content-Type': 'application/json', Accept: 'application/vnd.github+json' };
    if (token) headers.Authorization = 'Bearer ' + token;
    fetch(GITHUB_MARKDOWN, { method: 'POST', headers: headers, body: JSON.stringify({ text: md, mode: 'gfm' }) })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
      .then(function (html) { pv.innerHTML = ''; pv.appendChild(sanitizeHtml(html)); InlinePopover.renderMathIfPresent(pv); })
      .catch(function () { pv.innerHTML = '<em class="ac-muted">预览暂不可用（GitHub API 无法访问）</em>'; });
  }

  function setStatus(msg, kind) {
    var st = composer.querySelector('.ac-status');
    st.textContent = msg || '';
    st.className = 'ac-status' + (kind ? ' is-' + kind : '');
    st.style.display = msg ? 'block' : 'none';
  }

  function setBusy(busy) {
    var btns = composer.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) btns[i].disabled = !!busy;
    composer.classList.toggle('is-busy', !!busy);
  }

  // Draft survives the GitHub login round-trip.
  function draftKey() { return 'annotationDraft:' + cfg.path; }
  function saveDraft() {
    if (!pendingSelector) return;
    try {
      sessionStorage.setItem(draftKey(), JSON.stringify({ selector: pendingSelector, text: composer.querySelector('.ac-text').value }));
    } catch (e) { /* ignore */ }
  }
  function clearDraft() { try { sessionStorage.removeItem(draftKey()); } catch (e) { /* ignore */ } }
  function restoreDraft() {
    var raw = null;
    try { raw = sessionStorage.getItem(draftKey()); } catch (e) { /* ignore */ }
    if (!raw) return;
    var draft;
    try { draft = JSON.parse(raw); } catch (e) { clearDraft(); return; }
    if (!draft || !draft.selector) { clearDraft(); return; }
    var range = anchor(draft.selector);
    var rect = null;
    if (range) {
      var segs = segmentsFor(range.start, range.end);
      if (segs.length) {
        var r = document.createRange();
        r.setStart(segs[0].node, segs[0].from);
        r.setEnd(segs[segs.length - 1].node, segs[segs.length - 1].to);
        rect = r.getBoundingClientRect();
        InlinePopover.scrollToTargetWithOffset(segs[0].node.parentNode);
      }
    }
    openComposer(draft.selector, rect, draft.text);
  }

  // ----------------------------------------------------------------- auth

  function getSession() {
    try {
      var raw = localStorage.getItem(SESSION_KEY);
      return raw ? JSON.parse(raw) : '';
    } catch (e) { return ''; }
  }

  function login() {
    saveDraft();
    var url = new URL(location.href);
    url.hash = '';
    url.searchParams.delete('giscus');
    location.href = GISCUS_ORIGIN + '/api/oauth/authorize?redirect_uri=' + encodeURIComponent(url.toString());
  }

  function ensureToken() {
    if (token) return Promise.resolve(token);
    var session = getSession();
    if (!session) return Promise.reject(new Error('尚未登录 GitHub'));
    return api('/token', { method: 'POST', body: { session: session } }).then(function (data) {
      if (!data.token) throw new Error('no token');
      token = data.token;
      return token;
    });
  }

  function graphql(query, variables) {
    return ensureToken().then(function (tk) {
      return fetch(GITHUB_GRAPHQL, {
        method: 'POST',
        headers: { Authorization: 'Bearer ' + tk, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, variables: variables })
      });
    }).then(function (r) {
      return r.json().then(function (data) {
        if (r.status === 401 || (data.message && /bad credentials/i.test(data.message))) {
          token = null;
          try { localStorage.removeItem(SESSION_KEY); } catch (e) { /* ignore */ }
          throw new Error('登录已过期，请重新登录 GitHub');
        }
        if (data.errors && data.errors.length) throw new Error(data.errors[0].message);
        if (!r.ok) throw new Error(data.message || ('HTTP ' + r.status));
        return data.data;
      });
    });
  }

  function refreshAuthUI() {
    var userEl = composer.querySelector('.ac-user');
    var loginBtn = composer.querySelector('.ac-login');
    var submitBtn = composer.querySelector('.ac-submit');
    if (!getSession()) {
      userEl.innerHTML = '<span class="ac-muted">登录 GitHub 后即可发表</span>';
      loginBtn.style.display = '';
      submitBtn.style.display = 'none';
      return;
    }
    loginBtn.style.display = 'none';
    submitBtn.style.display = '';
    if (viewer) { renderViewer(viewer); return; }
    userEl.innerHTML = '<span class="ac-muted">正在连接 GitHub…</span>';
    graphql('{ viewer { login avatarUrl url } }').then(function (data) {
      viewer = data.viewer;
      renderViewer(viewer);
    }).catch(function (err) {
      userEl.innerHTML = '<span class="ac-muted">' + escapeHtml(err.message) + '</span>';
      if (!getSession()) { loginBtn.style.display = ''; submitBtn.style.display = 'none'; }
    });
  }

  function renderViewer(v) {
    composer.querySelector('.ac-user').innerHTML =
      '<img src="' + escapeAttr(v.avatarUrl) + '" alt="" width="22" height="22"> ' +
      '<a href="' + escapeAttr(v.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(v.login) + '</a>';
  }

  // --------------------------------------------------------------- submit

  var ADD_COMMENT = 'mutation($body: String!, $discussionId: ID!, $replyToId: ID) {' +
    ' addDiscussionComment(input: {body: $body, discussionId: $discussionId, replyToId: $replyToId}) { comment {' +
    ' id url createdAt upvoteCount bodyHTML author { login avatarUrl url } replies { totalCount } } } }';

  function ensureDiscussion() {
    if (discussion && discussion.id) return Promise.resolve(discussion.id);
    return loadAnnotations(true).then(function () {
      if (discussion && discussion.id) return discussion.id;
      var meta = document.querySelector("meta[property='og:description'], meta[name='description']");
      var backLink = location.href.replace(/#.*$/, '');
      var body = '# ' + cfg.path + '\n\n' + (meta ? meta.content : '') + '\n\n' + backLink;
      return ensureToken().then(function (tk) {
        return api('/discussions', {
          method: 'POST',
          headers: { Authorization: 'Bearer ' + tk },
          body: { input: { repositoryId: cfg.repoId, categoryId: cfg.categoryId, title: cfg.path, body: body } }
        });
      }).then(function (data) {
        if (!data.id) throw new Error('无法创建讨论串');
        discussion = { id: data.id, url: '', totalCommentCount: 0 };
        return data.id;
      });
    });
  }

  function submit() {
    if (!pendingSelector) return;
    var note = composer.querySelector('.ac-text').value.trim();
    if (!note) { setStatus('批注内容不能为空', 'error'); return; }
    var sel = pendingSelector;
    var body = buildCommentBody(sel, note, pendingOffsets);
    setBusy(true);
    setStatus('正在发表…');
    ensureDiscussion().then(function (id) {
      return graphql(ADD_COMMENT, { body: body, discussionId: id });
    }).then(function (data) {
      var c = data.addDiscussionComment.comment;
      var a = parseComment(c) || {
        id: c.id, url: c.url, author: c.author, createdAt: c.createdAt, upvoteCount: 0, replyCount: 0,
        noteHTML: c.bodyHTML, selector: sel
      };
      annotations.push(a);
      clearDraft();
      closeComposer();
      setBusy(false);
      applyHighlights();
      if (a.range) flashMarks(a.marks);
      showToast('批注已发表');
      refreshGiscus();
    }).catch(function (err) {
      setBusy(false);
      var msg = err.message || String(err);
      setStatus(msg + ' — 你也可以复制引用后粘贴到下方评论框发表。', 'error');
      var fallback = document.createElement('button');
      fallback.type = 'button';
      fallback.className = 'ac-fallback';
      fallback.innerHTML = '<i class="fa fa-clipboard"></i> 复制引用';
      fallback.addEventListener('click', function () {
        copyText(body).then(function () {
          showToast('已复制，请粘贴到评论框（⌘/Ctrl+V）');
          clearDraft(); closeComposer();
          var target = document.getElementById('comments');
          if (target) InlinePopover.scrollToTargetWithOffset(target);
        });
      });
      composer.querySelector('.ac-status').appendChild(document.createTextNode(' '));
      composer.querySelector('.ac-status').appendChild(fallback);
      if (!getSession()) refreshAuthUI();
    });
  }

  // Refresh the giscus list without blanking it: load a second, hidden iframe
  // with the same src and swap it in once it has reported its height.
  var pendingSwap = null;
  function refreshGiscus() {
    var old = document.querySelector('.giscus iframe.giscus-frame');
    if (!old || pendingSwap) return;
    var fresh = document.createElement('iframe');
    ['src', 'title', 'scrolling', 'allow'].forEach(function (k) { if (old.getAttribute(k)) fresh.setAttribute(k, old.getAttribute(k)); });
    fresh.className = 'giscus-frame';
    fresh.style.cssText = 'position:absolute;top:0;left:0;width:100%;visibility:hidden;height:' + old.offsetHeight + 'px';
    old.parentNode.style.position = 'relative';
    old.parentNode.appendChild(fresh);
    pendingSwap = { old: old, fresh: fresh, timer: setTimeout(function () { finishSwap(); }, 15000) };
  }

  function finishSwap() {
    if (!pendingSwap) return;
    var sw = pendingSwap;
    pendingSwap = null;
    clearTimeout(sw.timer);
    if (!sw.fresh.parentNode) return;
    sw.fresh.style.position = ''; sw.fresh.style.visibility = ''; sw.fresh.style.top = ''; sw.fresh.style.left = '';
    if (sw.old.parentNode) sw.old.parentNode.removeChild(sw.old);
  }

  // giscus (data-emit-metadata="1") posts its height and discussion metadata.
  // Heights are applied here for swapped-in iframes (giscus' client.js only
  // knows the iframe it created); a changed comment count triggers a refetch,
  // e.g. when a reader pasted a formatted quote straight into the giscus box.
  function bindGiscusMessages() {
    window.addEventListener('message', function (event) {
      if (event.origin !== GISCUS_ORIGIN) return;
      var g = event.data && event.data.giscus;
      if (!g) return;
      var frames = document.querySelectorAll('.giscus iframe.giscus-frame');
      for (var i = 0; i < frames.length; i++) {
        if (frames[i].contentWindow !== event.source) continue;
        if (g.resizeHeight) frames[i].style.height = g.resizeHeight + 'px';
        // the widget has rendered once it reports either its height or its metadata
        if (pendingSwap && frames[i] === pendingSwap.fresh && (g.resizeHeight || g.discussion)) finishSwap();
      }
      var d = g.discussion;
      if (!d || typeof d.totalCommentCount !== 'number') return;
      if (discussion && discussion.totalCommentCount === d.totalCommentCount) return;
      if (!discussion && d.totalCommentCount === 0) return;
      loadAnnotations(true);
    });
  }

  // ---------------------------------------------------------------- utils

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(function () { return legacyCopy(text); });
    }
    return Promise.resolve(legacyCopy(text));
  }

  function legacyCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  function showToast(msg) {
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'annotation-toast';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add('is-visible');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(function () { toast.classList.remove('is-visible'); }, 3000);
  }

  function relativeTime(iso) {
    var diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前';
    if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前';
    if (diff < 86400 * 30) return Math.floor(diff / 86400) + ' 天前';
    var d = new Date(iso);
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
  }

  function pad(n) { return (n < 10 ? '0' : '') + n; }

  function escapeHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function escapeAttr(s) { return escapeHtml(s); }

  // Exposed for debugging / tests in the browser console.
  window.BlogAnnotations = {
    reload: function () { return loadAnnotations(true); },
    anchor: anchor,
    buildIndex: buildIndex,
    buildTextFragment: buildTextFragment,
    annotHash: annotHash,
    permalink: permalink,
    refreshGiscus: refreshGiscus,
    buildCommentBody: buildCommentBody,
    parseComment: parseComment,
    list: function () { return annotations; }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
