/**
 * annotations.js — highlight annotations ("划线批注") on blog posts.
 *
 * Interaction (code-review / WeChat-reading style, no hover popups):
 *   - Select text in the article -> floating toolbar: 「评论」 / 「复制链接」.
 *   - 「评论」 opens a large editor panel *in the flow*, right below the
 *     paragraph, with 取消 / 提交评论 bottom-right. If the selection lies inside
 *     an already-annotated passage the note joins that thread instead.
 *   - Every annotated passage ends with a small comment-count marker; clicking
 *     it (or the highlight) expands a thread panel below the paragraph: all
 *     notes on that passage, their replies, and a box to add yours.
 *
 * Storage: a note is a normal comment of the post's giscus / GitHub Discussions
 * thread —
 *
 *     > quoted passage
 *     >
 *     > <sub>[§ 原文位置](https://arganzheng.life/<slug>.html#annot-<hash>) · [⚑ Issue #N](…)</sub>
 *
 *     the note
 *
 * — so it reads naturally on GitHub. The optional `⚑ Issue` link is added when
 * the reader ticks 「同时提交 Issue」 (code-review style "this needs fixing"): the
 * relay files a GitHub Issue first, then the comment links to it and is shown
 * with a red flag badge. On load the thread is fetched through the
 * secret-free relay in tools/annotations-worker/, comments of that shape are
 * parsed into a W3C TextQuoteSelector { exact, prefix, suffix } and anchored
 * exactly first, then fuzzily (approx-string-match, the algorithm Hypothesis
 * uses) so highlights survive small edits.
 *
 * Links: `#annot-<hash>` (FNV-1a of the quote) locates a thread, `#hl=<text>`
 * (from 「复制链接」) flashes a passage. Both are handled here on load and on
 * hashchange — no Text Fragment directive, so no sticky purple browser
 * highlight and no dependency on the browser's matcher (footnote markers
 * broke it).
 *
 * Posting reuses the reader's giscus session (localStorage["giscus-session"],
 * written by giscus' client.js): the relay exchanges it for a GitHub token and
 * the browser calls GitHub GraphQL directly, as the giscus iframe does. Any
 * failure degrades to copying the quote for pasting into giscus.
 */

(function () {
  'use strict';

  var GISCUS_ORIGIN = 'https://giscus.app';
  var GITHUB_GRAPHQL = 'https://api.github.com/graphql';
  var GITHUB_MARKDOWN = 'https://api.github.com/markdown';
  var SESSION_KEY = 'giscus-session';
  var CONTEXT_CHARS = 32;
  var EXCLUDE_SELECTOR = '.comment, .pager, .related-posts, .share, .footnotes, .reversefootnote, sup[id^="fnref"], a.footnote, ' +
    'script, style, noscript, svg, .katex, .mermaid, button, .anchorjs-link, .annotation-toolbar, .annotation-panel, .annotation-marker';
  var BLOCK_SELECTOR = 'p, li, pre, blockquote, h1, h2, h3, h4, h5, h6, dd, dt, figure, .highlight, table';
  var GHOST = { login: 'ghost', url: 'https://github.com/ghost', avatarUrl: 'https://avatars.githubusercontent.com/u/10137?s=64&v=4' };

  var cfg = null;
  var container = null;
  var index = null;          // { text, nodes:[{node, charIdx:[]}] }
  var annotations = [];      // parsed + anchored annotations
  var discussion = null;     // { id, url, totalCommentCount }
  var token = null;
  var viewer = null;
  var toolbar = null;
  var panel = null;          // the single in-flow panel (thread or editor)
  var panelState = null;     // { kind:'thread'|'editor', ids, selector, offsets, join }
  var toast = null;
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
      issues: section.getAttribute('data-issues') === '1',
      section: section
    };

    bindSelection();
    bindGiscusMessages();
    window.addEventListener('hashchange', focusFromHash);
    whenRichContentSettled(function () {
      loadAnnotations(false).then(function () { if (!restoreDraft()) focusFromHash(); });
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

  function buildIndex() {
    var text = '';
    var nodes = [];
    var lastWasSpace = true;
    var walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) { return isExcluded(n) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT; }
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
        charIdx[i] = text.length;
        text += ch;
      }
      nodes.push({ node: node, charIdx: charIdx });
    }
    index = { text: text, nodes: nodes };
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
          if (range.comparePoint(node, k) >= 0) { start = firstIndexAtOrAfter(entry, k); break; }
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
    while (start < end && index.text.charAt(start) === ' ') start++;
    while (end > start && index.text.charAt(end - 1) === ' ') end--;
    return end > start ? { start: start, end: end } : null;
  }

  function firstIndexAtOrAfter(entry, k) {
    for (var i = k; i < entry.charIdx.length; i++) if (entry.charIdx[i] !== -1) return entry.charIdx[i];
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

  // Cut text nodes at every boundary and wrap each piece in ONE <mark> carrying
  // every id that covers it (overlaps never nest). Pieces are processed
  // back-to-front so splitText() never invalidates pending offsets.
  function wrapPieces(items, className) {
    var perNode = [], byNode = new Map(), marks = [];
    items.forEach(function (it) {
      segmentsFor(it.start, it.end).forEach(function (sg) {
        var bucket = byNode.get(sg.node);
        if (!bucket) { bucket = { node: sg.node, segs: [] }; byNode.set(sg.node, bucket); perNode.push(bucket); }
        bucket.segs.push({ from: sg.from, to: sg.to, id: it.id });
      });
    });
    perNode.forEach(function (bucket) {
      var node = bucket.node, bounds = [];
      bucket.segs.forEach(function (sg) { if (bounds.indexOf(sg.from) === -1) bounds.push(sg.from); if (bounds.indexOf(sg.to) === -1) bounds.push(sg.to); });
      bounds.sort(function (x, y) { return x - y; });
      for (var b = bounds.length - 2; b >= 0; b--) {
        var from = bounds[b], to = bounds[b + 1];
        var ids = bucket.segs.filter(function (sg) { return sg.from <= from && sg.to >= to; }).map(function (sg) { return sg.id; });
        if (!ids.length) continue;
        var piece = node;
        if (to < piece.nodeValue.length) piece.splitText(to);
        if (from > 0) piece = piece.splitText(from);
        if (!piece.nodeValue.trim()) continue;
        var mark = document.createElement('mark');
        mark.className = className + (ids.length > 1 ? ' is-multi' : '');
        mark.setAttribute('data-annotation-ids', ids.join(' '));
        piece.parentNode.insertBefore(mark, piece);
        mark.appendChild(piece);
        marks.push(mark);
      }
    });
    marks.sort(function (x, y) { return x.compareDocumentPosition(y) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1; });
    return marks;
  }

  function unwrap(selector) {
    var marks = container.querySelectorAll(selector);
    for (var i = 0; i < marks.length; i++) {
      var m = marks[i], parent = m.parentNode;
      while (m.firstChild) parent.insertBefore(m.firstChild, m);
      parent.removeChild(m);
    }
    container.normalize();
  }

  function applyHighlights() {
    var markers = container.querySelectorAll('.annotation-marker');
    for (var i = 0; i < markers.length; i++) markers[i].parentNode.removeChild(markers[i]);
    unwrap('mark.annotation-hl');
    buildIndex();

    var orphans = [], items = [];
    annotations.forEach(function (a) {
      a.range = anchor(a.selector);
      a.marks = [];
      if (a.range) items.push({ start: a.range.start, end: a.range.end, id: a.id }); else orphans.push(a);
    });
    wrapPieces(items, 'annotation-hl').forEach(function (mark) {
      bindMark(mark);
      mark.getAttribute('data-annotation-ids').split(' ').forEach(function (id) { var an = findAnnotation(id); if (an) an.marks.push(mark); });
    });
    insertMarkers();
    buildIndex();
    renderOrphans(orphans);
    if (panelState && panelState.kind === 'thread') refreshThreadPanel();
  }

  // One marker per distinct passage (annotations with an identical range share it).
  function insertMarkers() {
    var groups = {};
    annotations.forEach(function (a) {
      if (!a.range || !a.marks.length) return;
      var key = a.range.start + '-' + a.range.end;
      (groups[key] = groups[key] || []).push(a);
    });
    Object.keys(groups).forEach(function (key) {
      var group = groups[key];
      var count = group.reduce(function (n, a) { return n + 1 + (a.replies || []).length; }, 0);
      var last = group[0].marks[group[0].marks.length - 1];
      var marker = document.createElement('span');
      marker.className = 'annotation-marker';
      marker.setAttribute('data-annotation-ids', group.map(function (a) { return a.id; }).join(' '));
      marker.setAttribute('title', count + ' 条批注，点击查看');
      marker.innerHTML = '<i class="fa fa-comment"></i><span class="annotation-marker-count">' + count + '</span>';
      marker.addEventListener('click', function (e) {
        e.preventDefault(); e.stopPropagation();
        toggleThread(group.map(function (a) { return a.id; }), last);
      });
      last.insertAdjacentElement('afterend', marker);
    });
  }

  function reanchorAll() {
    if (annotations.length) applyHighlights(); else buildIndex();
  }

  function bindMark(mark) {
    if (mark.closest('a[href]:not(.inline-tip)')) return; // links keep navigating; use the marker
    mark.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      toggleThread(idsFor(mark), mark);
    });
  }

  function idsFor(mark) {
    return (mark.getAttribute('data-annotation-ids') || '').split(' ').filter(Boolean);
  }

  function findAnnotation(id) {
    for (var i = 0; i < annotations.length; i++) if (annotations[i].id === id) return annotations[i];
    return null;
  }

  function flashMarks(marks) {
    marks.forEach(function (m) {
      m.classList.add('is-new');
      setTimeout(function () { m.classList.remove('is-new'); }, 2500);
    });
  }

  // Temporary highlight for `#hl=` links: wrap, flash, unwrap.
  function flashRange(range) {
    var marks = wrapPieces([{ start: range.start, end: range.end, id: 'flash' }], 'annotation-flash');
    if (!marks.length) return;
    scrollIntoViewInstant(marks[0]);
    setTimeout(function () { unwrap('mark.annotation-flash'); buildIndex(); }, 2600);
  }

  function scrollIntoViewInstant(el) {
    window.scrollTo({ top: Math.max(0, el.getBoundingClientRect().top + window.pageYOffset - 160), behavior: 'instant' });
  }

  // #annot-<hash> -> open that thread; #hl=<text> -> flash that passage.
  function focusFromHash() {
    var h = location.hash || '';
    var m = /^#annot-([0-9a-f]{8})/.exec(h);
    if (m) {
      for (var i = 0; i < annotations.length; i++) {
        var a = annotations[i];
        if (annotHash(a.selector.exact) !== m[1] || !a.marks.length) continue;
        scrollIntoViewInstant(a.marks[0]);
        flashMarks(a.marks);
        openThread(groupIdsFor(a), a.marks[a.marks.length - 1]);
        return;
      }
      showToast('这条批注对应的原文找不到了（可能已被修改）');
      return;
    }
    var hl = /^#hl=(.*)$/.exec(h);
    if (hl) {
      var exact;
      try { exact = decodeURIComponent(hl[1]); } catch (e) { exact = hl[1]; }
      var range = anchor({ exact: exact.replace(/\+/g, ' ') });
      if (range) flashRange(range); else showToast('链接指向的文字在本页找不到了');
    }
  }

  // ids of every annotation sharing `a`'s exact passage
  function groupIdsFor(a) {
    return annotations.filter(function (b) {
      return b.range && a.range && b.range.start === a.range.start && b.range.end === a.range.end;
    }).map(function (b) { return b.id; });
  }

  // ---------------------------------------------------------------- panel

  // Block element after which an in-flow panel for `node` is inserted.
  function blockFor(node) {
    var el = node.nodeType === 1 ? node : node.parentNode;
    var cell = el.closest('td, th');
    var block = cell ? cell.closest('table') : el.closest(BLOCK_SELECTOR);
    if (block && block.tagName === 'PRE' && block.parentNode.classList.contains('highlight')) block = block.parentNode;
    if (block && block.parentNode && block.parentNode.classList.contains('highlighter-rouge')) block = block.parentNode;
    return (block && container.contains(block) && block !== container) ? block : el;
  }

  function ensurePanel() {
    if (panel) return panel;
    panel = document.createElement('div');
    panel.className = 'annotation-panel';
    panel.addEventListener('click', function (e) { e.stopPropagation(); });
    panel.addEventListener('mousedown', function (e) { e.stopPropagation(); });
    return panel;
  }

  function mountPanel(afterNode) {
    ensurePanel();
    var block = blockFor(afterNode);
    block.parentNode.insertBefore(panel, block.nextSibling);
    hideToolbar();
  }

  function closePanel() {
    if (panel && panel.parentNode) panel.parentNode.removeChild(panel);
    panelState = null;
  }

  function toggleThread(ids, anchorMark) {
    if (panelState && panelState.kind === 'thread' && panelState.ids.join() === ids.join()) { closePanel(); return; }
    openThread(ids, anchorMark);
  }

  function openThread(ids, anchorMark) {
    var list = ids.map(findAnnotation).filter(Boolean);
    if (!list.length) return;
    panelState = { kind: 'thread', ids: ids };
    ensurePanel();
    panel.className = 'annotation-panel is-thread';
    renderThread(list);
    mountPanel(anchorMark || list[0].marks[list[0].marks.length - 1]);
    if (window.getSelection) window.getSelection().removeAllRanges();
  }

  function refreshThreadPanel() {
    var list = panelState.ids.map(findAnnotation).filter(Boolean);
    if (!list.length || !list[0].marks.length) { closePanel(); return; }
    renderThread(list);
    if (!panel.parentNode) mountPanel(list[0].marks[list[0].marks.length - 1]);
  }

  function renderThread(list) {
    var primary = list[0];
    var total = list.reduce(function (n, a) { return n + 1 + (a.replies || []).length; }, 0);
    panel.innerHTML =
      '<div class="ap-head">' +
        '<i class="fa fa-quote-left"></i><span class="ap-quote" title="' + escapeAttr(primary.selector.exact) + '">' + escapeHtml(primary.selector.exact) + '</span>' +
        '<span class="ap-count">' + total + ' 条批注</span>' +
        '<button type="button" class="ap-close" title="收起">×</button>' +
      '</div>' +
      '<div class="ap-thread"></div>' +
      '<div class="ap-editor"></div>';
    panel.querySelector('.ap-close').addEventListener('click', closePanel);
    var thread = panel.querySelector('.ap-thread');
    var editorHost = panel.querySelector('.ap-editor');
    var replyTo = null; // { annotation, mention } or null = comment on the passage

    function setReplyTarget(a, mentionLogin) {
      replyTo = a ? { annotation: a, mention: mentionLogin } : null;
      var chip = editorHost.querySelector('.ap-reply-chip');
      chip.style.display = replyTo ? 'flex' : 'none';
      if (replyTo) chip.querySelector('span').textContent = '回复 @' + mentionLogin;
      var ta = editorHost.querySelector('.ap-text');
      ta.placeholder = replyTo ? '回复 @' + mentionLogin + '…' : '对这段文字发表评论…';
      editorHost.querySelector('.ap-submit-label').textContent = replyTo ? '回复' : '发表评论';
      var issueOpt = editorHost.querySelector('.ap-issue'); // replies are never issues
      if (issueOpt) { issueOpt.style.display = replyTo ? 'none' : ''; if (replyTo) issueOpt.querySelector('input').checked = false; }
      if (replyTo && mentionLogin !== a.author.login && !ta.value) ta.value = '@' + mentionLogin + ' ';
      ta.focus();
      ta.dispatchEvent(new Event('input'));
    }

    list.forEach(function (a) {
      thread.appendChild(commentEl(a, a.noteHTML, false, function () { setReplyTarget(a, a.author.login); }));
      (a.replies || []).forEach(function (r) {
        thread.appendChild(commentEl(r, r.bodyHTML, true, function () { setReplyTarget(a, r.author.login); }, a));
      });
    });
    renderEditor(editorHost, {
      placeholder: '对这段文字发表评论…',
      submitLabel: '发表评论',
      compact: true,
      replyChip: true,
      issueOption: true,
      onCancel: closePanel,
      onSubmit: function (text, extra) {
        return replyTo ? postReply(replyTo.annotation, text) : postAnnotation(primary.selector, text, extra.issue);
      },
      fallbackText: function (text) { return replyTo ? text : buildCommentBody(primary.selector, text); }
    });
    editorHost.querySelector('.ap-reply-chip button').addEventListener('click', function () {
      var ta = editorHost.querySelector('.ap-text');
      if (replyTo && ta.value.trim() === '@' + replyTo.mention) ta.value = '';
      setReplyTarget(null);
    });
  }

  // `parent` is the top-level annotation a reply belongs to (undefined for top-level).
  function commentEl(c, html, isReply, onReply, parent) {
    var el = document.createElement('div');
    var mine = viewer && c.author && viewer.login === c.author.login;
    el.className = 'ap-comment' + (isReply ? ' is-reply' : '') + (c.issue ? ' has-issue' : '');
    el.innerHTML =
      '<a class="ap-avatar" href="' + escapeAttr(c.author.url) + '" target="_blank" rel="noopener noreferrer"><img src="' + escapeAttr(c.author.avatarUrl) + '" alt=""></a>' +
      '<div class="ap-comment-main">' +
        '<div class="ap-comment-meta"><a href="' + escapeAttr(c.author.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(c.author.login) + '</a>' +
          '<time datetime="' + escapeAttr(c.createdAt) + '" title="' + escapeAttr(new Date(c.createdAt).toLocaleString()) + '">' + relativeTime(c.createdAt) + '</time>' +
          (c.upvoteCount ? '<span class="ap-upvotes"><i class="fa fa-caret-up"></i> ' + c.upvoteCount + '</span>' : '') +
          (c.issue ? '<a class="ap-issue-badge" href="' + escapeAttr(c.issue.url) + '" target="_blank" rel="noopener noreferrer" title="已同时提交为 GitHub Issue"><i class="fa fa-flag"></i> Issue' + (c.issue.number ? ' #' + c.issue.number : '') + '</a>' : '') +
          '<span class="ap-meta-actions">' +
            (onReply ? '<button type="button" class="ap-reply-btn"><i class="fa fa-reply"></i> 回复</button>' : '') +
            (mine ? '<button type="button" class="ap-edit-btn" title="编辑"><i class="fa fa-pencil"></i> 编辑</button>' +
                    '<button type="button" class="ap-delete-btn" title="删除"><i class="fa fa-trash-o"></i> 删除</button>' : '') +
            '<a class="ap-github" href="' + escapeAttr(c.url) + '" target="_blank" rel="noopener noreferrer" title="在 GitHub 上查看 / 编辑"><i class="fa fa-github"></i></a>' +
          '</span>' +
        '</div>' +
        '<div class="ap-comment-body"></div>' +
      '</div>';
    var body = el.querySelector('.ap-comment-body');
    body.appendChild(sanitizeHtml(html));
    InlinePopover.renderMathIfPresent(body);
    if (onReply) el.querySelector('.ap-reply-btn').addEventListener('click', onReply);
    if (mine) {
      el.querySelector('.ap-edit-btn').addEventListener('click', function () { startEdit(el, c, isReply, parent); });
      el.querySelector('.ap-delete-btn').addEventListener('click', function () { deleteComment(el, c, isReply, parent); });
    }
    return el;
  }

  // ------------------------------------------------- edit / delete (own comments)

  var NODE_BODY = 'query($id: ID!) { node(id: $id) { ... on DiscussionComment { body } } }';
  var UPDATE_COMMENT = 'mutation($id: ID!, $body: String!) { updateDiscussionComment(input: {commentId: $id, body: $body}) { comment { id bodyHTML } } }';
  var DELETE_COMMENT = 'mutation($id: ID!) { deleteDiscussionComment(input: {id: $id}) { comment { id } } }';

  // A top-level note's raw body starts with the quote header; only the note is editable.
  function stripQuoteHeader(md) {
    var lines = md.replace(/\r/g, '').split('\n'), i = 0;
    if (!/^>/.test(lines[0])) return md;
    while (i < lines.length && /^>/.test(lines[i])) i++;
    while (i < lines.length && !lines[i].trim()) i++;
    return lines.slice(i).join('\n');
  }

  function startEdit(el, c, isReply, parent) {
    if (el.querySelector('.ap-inline-editor')) return;
    var bodyEl = el.querySelector('.ap-comment-body');
    var host = document.createElement('div');
    host.className = 'ap-editor ap-inline-editor';
    host.innerHTML = '<span class="ap-muted">正在读取原文…</span>';
    bodyEl.style.display = 'none';
    bodyEl.parentNode.insertBefore(host, bodyEl.nextSibling);
    function cancel() { host.parentNode.removeChild(host); bodyEl.style.display = ''; }
    graphql(NODE_BODY, { id: c.id }).then(function (data) {
      var raw = (data.node && data.node.body) || '';
      var ta = renderEditor(host, {
        placeholder: '编辑评论…',
        submitLabel: '保存',
        compact: true,
        inline: true,
        initialText: isReply ? raw : stripQuoteHeader(raw),
        onCancel: cancel,
        onSubmit: function (text) {
          var body = isReply ? text : buildCommentBody(c.selector, text, c.issue);
          return graphql(UPDATE_COMMENT, { id: c.id, body: body }).then(function (res) {
            var html = res.updateDiscussionComment.comment.bodyHTML;
            if (isReply) c.bodyHTML = html;
            else { var p = parseComment({ id: c.id, bodyHTML: html, author: c.author, createdAt: c.createdAt }); c.noteHTML = p ? p.noteHTML : html; }
            refreshThreadPanel();
            showToast('已保存');
            refreshGiscus();
          });
        }
      });
      ta.focus();
    }).catch(function (err) { host.innerHTML = '<span class="ap-status is-error">' + escapeHtml(err.message) + '</span>'; setTimeout(cancel, 2500); });
  }

  function deleteComment(el, c, isReply, parent) {
    var n = !isReply && c.replies && c.replies.length;
    if (!window.confirm(n ? '删除这条评论？它下面的 ' + n + ' 条回复也会一起删除。' : '删除这条评论？')) return;
    el.classList.add('is-deleting');
    graphql(DELETE_COMMENT, { id: c.id }).then(function () {
      if (isReply) {
        parent.replies = parent.replies.filter(function (r) { return r.id !== c.id; });
        parent.replyCount = parent.replies.length;
      } else {
        annotations = annotations.filter(function (a) { return a.id !== c.id; });
        if (panelState) panelState.ids = panelState.ids.filter(function (id) { return id !== c.id; });
      }
      applyHighlights(); // re-renders (or closes) the thread panel
      showToast('已删除');
      refreshGiscus();
    }).catch(function (err) { el.classList.remove('is-deleting'); showToast('删除失败：' + err.message); });
  }

  // ---------------------------------------------------------------- editor

  // Shared editor block: textarea + preview + auth + buttons. `opts.onSubmit(text)`
  // returns a promise; the editor shows errors and offers the clipboard fallback.
  var FORMAT_BUTTONS = [
    { key: 'bold', icon: 'fa-bold', title: '加粗 (**文字**)' },
    { key: 'italic', icon: 'fa-italic', title: '斜体 (*文字*)' },
    { key: 'heading', icon: 'fa-header', title: '标题 (### )' },
    { key: 'quote', icon: 'fa-quote-right', title: '引用 (> )' },
    { key: 'code', icon: 'fa-code', title: '行内代码 (`code`)' },
    { key: 'codeblock', icon: 'fa-file-code-o', title: '代码块 (```)' },
    { key: 'link', icon: 'fa-link', title: '链接 [文字](url)' },
    { key: 'image', icon: 'fa-picture-o', title: '图片 ![说明](url)' },
    { key: 'ul', icon: 'fa-list-ul', title: '无序列表 (- )' },
    { key: 'ol', icon: 'fa-list-ol', title: '有序列表 (1. )' }
  ];

  function renderEditor(host, opts) {
    host.classList.toggle('is-inline', !!opts.inline);
    host.innerHTML =
      '<div class="ap-tabs"><button type="button" class="is-active" data-tab="write">撰写</button><button type="button" data-tab="preview">预览</button>' +
        '<span class="ap-format">' + FORMAT_BUTTONS.map(function (b) {
          return '<button type="button" data-format="' + b.key + '" title="' + b.title + '"><i class="fa ' + b.icon + '"></i></button>';
        }).join('') + '</span>' +
      '</div>' +
      (opts.replyChip ? '<div class="ap-reply-chip" style="display:none"><i class="fa fa-reply"></i><span></span><button type="button" title="改为对这段文字评论">×</button></div>' : '') +
      '<textarea class="ap-text" rows="' + (opts.compact ? 4 : 6) + '" placeholder="' + escapeAttr(opts.placeholder) + '"></textarea>' +
      '<div class="ap-preview" style="display:none"></div>' +
      '<div class="ap-status" style="display:none"></div>' +
      '<div class="ap-footer">' +
        '<span class="ap-user"></span>' +
        '<span class="ap-buttons">' +
          '<span class="ap-hint">Markdown · ⌘/Ctrl+Enter 提交</span>' +
          (opts.issueOption && cfg.issues ? '<label class="ap-issue" title="除评论外，再以你的名义在 GitHub 仓库创建一个 Issue，提醒作者这里可能有问题"><input type="checkbox"><i class="fa fa-flag"></i> 同时提交 Issue</label>' : '') +
          (opts.onCancel ? '<button type="button" class="ap-cancel">取消</button>' : '') +
          '<button type="button" class="ap-login"><i class="fa fa-github"></i> 使用 GitHub 登录</button>' +
          '<button type="button" class="ap-submit" disabled><i class="fa fa-paper-plane"></i> <span class="ap-submit-label">' + escapeHtml(opts.submitLabel) + '</span></button>' +
        '</span>' +
      '</div>';
    var ta = host.querySelector('.ap-text');
    var submitBtn = host.querySelector('.ap-submit');
    if (opts.initialText) ta.value = opts.initialText;
    function updateSubmitState() { submitBtn.disabled = host.classList.contains('is-busy') || !ta.value.trim(); }
    ta.addEventListener('input', function () { updateSubmitState(); if (opts.onChange) opts.onChange(ta.value); });
    updateSubmitState();
    if (opts.onCancel) host.querySelector('.ap-cancel').addEventListener('click', opts.onCancel);
    host.querySelector('.ap-login').addEventListener('click', function () { if (opts.beforeLogin) opts.beforeLogin(ta.value); login(); });
    var tabs = host.querySelectorAll('.ap-tabs button[data-tab]');
    for (var i = 0; i < tabs.length; i++) tabs[i].addEventListener('click', function () { switchTab(host, this.getAttribute('data-tab')); });
    var fmts = host.querySelectorAll('.ap-format button');
    for (var f = 0; f < fmts.length; f++) fmts[f].addEventListener('click', function () { switchTab(host, 'write'); applyFormat(ta, this.getAttribute('data-format')); });

    function submit() {
      var text = ta.value.trim();
      if (!text) return;
      var issueBox = host.querySelector('.ap-issue input');
      var wantIssue = !!(issueBox && issueBox.checked);
      setBusy(host, true);
      setStatus(host, wantIssue ? '正在创建 Issue…' : '正在发表…');
      opts.onSubmit(text, { issue: wantIssue }).then(function () {
        setBusy(host, false);
      }).catch(function (err) {
        setBusy(host, false);
        updateSubmitState();
        var msg = err.message || String(err);
        if (err.issuesDisabled) { cfg.issues = false; var lbl = host.querySelector('.ap-issue'); if (lbl) lbl.parentNode.removeChild(lbl); }
        setStatus(host, msg + ' — 也可以复制内容后粘贴到文末评论框发表。', 'error');
        var fallback = document.createElement('button');
        fallback.type = 'button';
        fallback.className = 'ap-fallback';
        fallback.innerHTML = '<i class="fa fa-clipboard"></i> 复制内容';
        fallback.addEventListener('click', function () {
          copyText(opts.fallbackText ? opts.fallbackText(text) : text).then(function () {
            showToast('已复制，请粘贴到评论框（⌘/Ctrl+V）');
            var target = document.getElementById('comments');
            if (target) InlinePopover.scrollToTargetWithOffset(target);
          });
        });
        host.querySelector('.ap-status').appendChild(document.createTextNode(' '));
        host.querySelector('.ap-status').appendChild(fallback);
        refreshAuthUI(host);
      });
    }
    submitBtn.addEventListener('click', submit);
    ta.addEventListener('keydown', function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); submit(); return; }
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && (e.key === 'b' || e.key === 'i' || e.key === 'k')) {
        e.preventDefault();
        applyFormat(ta, e.key === 'b' ? 'bold' : e.key === 'i' ? 'italic' : 'link');
      }
    });
    refreshAuthUI(host);
    return ta;
  }

  // Minimal Markdown helpers for the toolbar: wrap the selection or prefix the
  // selected lines; setRangeText keeps the browser's undo stack intact.
  function applyFormat(ta, kind) {
    var v = ta.value, s = ta.selectionStart, e = ta.selectionEnd, sel = v.slice(s, e);
    function wrap(left, right, placeholder) {
      var inner = sel || placeholder;
      ta.setRangeText(left + inner + right, s, e, 'select');
      if (!sel) ta.setSelectionRange(s + left.length, s + left.length + inner.length);
      else ta.setSelectionRange(s + left.length, s + left.length + inner.length);
    }
    function prefixLines(prefixFn) {
      var ls = v.lastIndexOf('\n', s - 1) + 1;
      var le = v.indexOf('\n', e); if (le === -1) le = v.length;
      var lines = v.slice(ls, le).split('\n');
      var out = lines.map(function (l, i) { return prefixFn(i) + l; }).join('\n');
      ta.setRangeText(out, ls, le, 'select');
      ta.setSelectionRange(ls, ls + out.length);
    }
    function block(open, close, placeholder) {
      var inner = sel || placeholder;
      var before = (s > 0 && v.charAt(s - 1) !== '\n') ? '\n' : '';
      var after = (e < v.length && v.charAt(e) !== '\n') ? '\n' : '';
      var text = before + open + '\n' + inner + '\n' + close + after;
      ta.setRangeText(text, s, e, 'select');
      var innerStart = s + before.length + open.length + 1;
      ta.setSelectionRange(innerStart, innerStart + inner.length);
    }
    switch (kind) {
      case 'bold': wrap('**', '**', '加粗文字'); break;
      case 'italic': wrap('*', '*', '斜体文字'); break;
      case 'code': wrap('`', '`', 'code'); break;
      case 'heading': prefixLines(function () { return '### '; }); break;
      case 'quote': prefixLines(function () { return '> '; }); break;
      case 'ul': prefixLines(function () { return '- '; }); break;
      case 'ol': prefixLines(function (i) { return (i + 1) + '. '; }); break;
      case 'codeblock': block('```', '```', '代码'); break;
      case 'link':
        if (/^https?:\/\//.test(sel)) { ta.setRangeText('[链接文字](' + sel + ')', s, e, 'select'); ta.setSelectionRange(s + 1, s + 5); }
        else { var t = sel || '链接文字'; ta.setRangeText('[' + t + '](url)', s, e, 'select'); ta.setSelectionRange(s + t.length + 3, s + t.length + 6); }
        break;
      case 'image': ta.setRangeText('![' + (sel || '图片说明') + '](url)', s, e, 'select'); ta.setSelectionRange(s + (sel || '图片说明').length + 4, s + (sel || '图片说明').length + 7); break;
    }
    ta.focus();
    ta.dispatchEvent(new Event('input'));
  }

  function switchTab(host, tab) {
    var tabs = host.querySelectorAll('.ap-tabs button');
    for (var i = 0; i < tabs.length; i++) tabs[i].classList.toggle('is-active', tabs[i].getAttribute('data-tab') === tab);
    var ta = host.querySelector('.ap-text'), pv = host.querySelector('.ap-preview');
    ta.style.display = tab === 'write' ? '' : 'none';
    pv.style.display = tab === 'preview' ? 'block' : 'none';
    if (tab === 'preview') renderPreview(pv, ta.value);
  }

  function renderPreview(pv, md) {
    if (!md.trim()) { pv.innerHTML = '<em class="ap-muted">没有内容可预览</em>'; return; }
    pv.innerHTML = '<em class="ap-muted">渲染中…</em>';
    var headers = { 'Content-Type': 'application/json', Accept: 'application/vnd.github+json' };
    if (token) headers.Authorization = 'Bearer ' + token;
    fetch(GITHUB_MARKDOWN, { method: 'POST', headers: headers, body: JSON.stringify({ text: md, mode: 'gfm' }) })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
      .then(function (html) { pv.innerHTML = ''; pv.appendChild(sanitizeHtml(html)); InlinePopover.renderMathIfPresent(pv); })
      .catch(function () { pv.innerHTML = '<em class="ap-muted">预览暂不可用（GitHub API 无法访问）</em>'; });
  }

  function setStatus(host, msg, kind) {
    var st = host.querySelector('.ap-status');
    st.textContent = msg || '';
    st.className = 'ap-status' + (kind ? ' is-' + kind : '');
    st.style.display = msg ? 'block' : 'none';
  }

  function setBusy(host, busy) {
    var btns = host.querySelectorAll('button, textarea, input');
    for (var i = 0; i < btns.length; i++) btns[i].disabled = !!busy;
    host.classList.toggle('is-busy', !!busy);
    var ta = host.querySelector('.ap-text'), submit = host.querySelector('.ap-submit');
    if (!busy && ta && submit) submit.disabled = !ta.value.trim();
  }

  function refreshAuthUI(host) {
    var userEl = host.querySelector('.ap-user');
    var loginBtn = host.querySelector('.ap-login');
    var submitBtn = host.querySelector('.ap-submit');
    if (!getSession()) {
      userEl.innerHTML = '<span class="ap-muted">登录 GitHub 后即可发表（与文末评论区同一账号）</span>';
      loginBtn.style.display = '';
      submitBtn.style.display = 'none';
      return;
    }
    loginBtn.style.display = 'none';
    submitBtn.style.display = '';
    if (viewer) { renderViewer(userEl, viewer); return; }
    userEl.innerHTML = '<span class="ap-muted">正在连接 GitHub…</span>';
    graphql('{ viewer { login avatarUrl url } }').then(function (data) {
      viewer = data.viewer;
      renderViewer(userEl, viewer);
      // now we know which comments are the reader's own -> show 编辑 / 删除
      if (panelState && panelState.kind === 'thread' && !panel.querySelector('.ap-text').value) refreshThreadPanel();
    }).catch(function (err) {
      userEl.innerHTML = '<span class="ap-muted">' + escapeHtml(err.message) + '</span>';
      if (!getSession()) { loginBtn.style.display = ''; submitBtn.style.display = 'none'; }
    });
  }

  function renderViewer(userEl, v) {
    userEl.innerHTML =
      '<img src="' + escapeAttr(v.avatarUrl) + '" alt="" width="22" height="22"> ' +
      '<a href="' + escapeAttr(v.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(v.login) + '</a>';
  }

  // --------------------------------------------------------- new annotation

  // An existing annotation whose passage contains the selection -> join it.
  function joinTargetFor(offsets) {
    var best = null;
    annotations.forEach(function (a) {
      if (!a.range) return;
      if (a.range.start <= offsets.start && a.range.end >= offsets.end) {
        if (!best || (a.range.end - a.range.start) < (best.range.end - best.range.start)) best = a;
      }
    });
    return best;
  }

  function openComposer(sel, offsets, anchorNode, draftText) {
    var join = joinTargetFor(offsets);
    if (join) {
      // same passage -> one thread; the editor there defaults to a comment on the passage
      openThread(groupIdsFor(join), join.marks[join.marks.length - 1]);
      var joinTa = panel.querySelector('.ap-text');
      if (joinTa) { if (draftText) joinTa.value = draftText; joinTa.focus(); joinTa.dispatchEvent(new Event('input')); }
      return;
    }
    panelState = { kind: 'editor', selector: sel, offsets: offsets };
    ensurePanel();
    panel.className = 'annotation-panel is-editor';
    panel.innerHTML =
      '<div class="ap-head">' +
        '<i class="fa fa-quote-left"></i><span class="ap-quote" title="' + escapeAttr(sel.exact) + '">' + escapeHtml(sel.exact) + '</span>' +
        '<span class="ap-count">新批注</span>' +
        '<button type="button" class="ap-close" title="取消">×</button>' +
      '</div>' +
      '<div class="ap-editor"></div>';
    panel.querySelector('.ap-close').addEventListener('click', cancelComposer);
    var ta = renderEditor(panel.querySelector('.ap-editor'), {
      placeholder: '写下你对这段文字的批注…',
      submitLabel: '提交评论',
      initialText: draftText || '',
      issueOption: true,
      onCancel: cancelComposer,
      onChange: saveDraft,
      beforeLogin: saveDraft,
      fallbackText: function (text) { return buildCommentBody(sel, text); },
      onSubmit: function (text, extra) { return postAnnotation(sel, text, extra.issue); }
    });
    mountPanel(anchorNode);
    if (window.getSelection) window.getSelection().removeAllRanges();
    ta.focus();
    saveDraft(ta.value);
  }

  function cancelComposer() { clearDraft(); closePanel(); }

  // With `withIssue` the GitHub Issue is filed first so the comment can link to
  // it (the comment is the record readers see; the issue is the author's todo).
  function postAnnotation(sel, text, withIssue) {
    var issue = null;
    return ensureDiscussion().then(function (id) {
      return (withIssue ? createIssue(sel, text) : Promise.resolve(null)).then(function (is) {
        issue = is;
        return graphql(ADD_COMMENT, { body: buildCommentBody(sel, text, issue), discussionId: id });
      });
    }).then(function (data) {
      var c = data.addDiscussionComment.comment;
      var a = parseComment(c) || {
        id: c.id, url: c.url, author: c.author || GHOST, createdAt: c.createdAt, upvoteCount: 0, replyCount: 0, replies: [],
        noteHTML: c.bodyHTML, selector: sel, issue: issue
      };
      annotations.push(a);
      clearDraft();
      closePanel();
      applyHighlights();
      if (a.marks.length) {
        flashMarks(a.marks);
        openThread(groupIdsFor(a), a.marks[a.marks.length - 1]);
      }
      showToast(issue ? '批注已发表，Issue #' + issue.number + ' 已创建' : '批注已发表');
      refreshGiscus();
    });
  }

  function createIssue(sel, text) {
    var titleEl = document.querySelector('.page-header .title, .post-heading h1, h1');
    var postTitle = (titleEl && titleEl.textContent) || document.title.split(/\s[-|]\s/)[0];
    var snippet = sel.exact.length > 40 ? sel.exact.slice(0, 40) + '…' : sel.exact;
    var body = '> ' + escapeMarkdown(sel.exact) + '\n>\n> [§ 原文位置](' + threadLink(sel) + ')\n\n' + text.trim() + '\n';
    return ensureToken().then(function (tk) {
      return api('/issues', {
        method: 'POST',
        headers: { Authorization: 'Bearer ' + tk },
        body: { title: postTitle.trim() + '：「' + snippet + '」', body: body }
      });
    }).catch(function (err) {
      if (/501|未启用/.test(err.message)) err.issuesDisabled = true;
      err.message = '创建 Issue 失败：' + err.message;
      throw err;
    });
  }

  function postReply(a, text) {
    return graphql(ADD_COMMENT, { body: text, discussionId: discussion.id, replyToId: a.id }).then(function (data) {
      var c = data.addDiscussionComment.comment;
      a.replies = (a.replies || []).concat([{ id: c.id, url: c.url, author: c.author || GHOST, createdAt: c.createdAt, bodyHTML: c.bodyHTML }]);
      a.replyCount = a.replies.length;
      applyHighlights(); // marker counts
      openThread(groupIdsFor(a), a.marks[a.marks.length - 1]);
      var items = panel.querySelectorAll('.ap-comment');
      if (items.length) items[items.length - 1].classList.add('is-new');
      showToast('评论已发表');
      refreshGiscus();
    });
  }

  // Draft survives the GitHub login round-trip.
  function draftKey() { return 'annotationDraft:' + cfg.path; }
  function saveDraft(text) {
    if (!panelState || panelState.kind !== 'editor') return;
    if (typeof text !== 'string') { var ta = panel && panel.querySelector('.ap-text'); text = ta ? ta.value : ''; }
    try { sessionStorage.setItem(draftKey(), JSON.stringify({ selector: panelState.selector, text: text })); } catch (e) { /* ignore */ }
  }
  function clearDraft() { try { sessionStorage.removeItem(draftKey()); } catch (e) { /* ignore */ } }
  function restoreDraft() {
    var raw = null;
    try { raw = sessionStorage.getItem(draftKey()); } catch (e) { /* ignore */ }
    if (!raw) return false;
    var draft;
    try { draft = JSON.parse(raw); } catch (e) { clearDraft(); return false; }
    if (!draft || !draft.selector) { clearDraft(); return false; }
    var range = anchor(draft.selector);
    if (!range) { clearDraft(); return false; }
    var segs = segmentsFor(range.start, range.end);
    if (!segs.length) { clearDraft(); return false; }
    var last = segs[segs.length - 1].node;
    scrollIntoViewInstant(last.parentNode);
    openComposer(draft.selector, range, last, draft.text);
    return true;
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
    var links = quote.querySelectorAll('a[href*="#annot-"], a[href*=":~:text="]');
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
    var issueLink = linkBlock.querySelector('a[href*="/issues/"]');
    var issueNo = issueLink && /\/issues\/(\d+)/.exec(issueLink.getAttribute('href'));
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
      selector: { exact: exact, prefix: fragment.prefix, suffix: fragment.suffix },
      issue: issueLink ? { url: issueLink.getAttribute('href'), number: issueNo ? +issueNo[1] : 0 } : null
    };
  }

  // giscus' adapter returns replies as a plain array; GitHub GraphQL as {nodes}.
  function parseReplies(replies) {
    var list = Array.isArray(replies) ? replies : (replies && replies.nodes) || [];
    return list.filter(function (r) { return r && !r.deletedAt && !r.isMinimized; }).map(function (r) {
      return { id: r.id, url: r.url, createdAt: r.createdAt, bodyHTML: r.bodyHTML, author: r.author || GHOST };
    });
  }

  // Older links carried a Text Fragment with prefix-/-suffix context; keep reading it.
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

  // Escape only what URL / Markdown-link syntax needs; CJK stays readable.
  function encodeFragmentPart(s) {
    return s.replace(/[\s%&#()"'<>\[\]\\^`{}|]/g, function (c) {
      return '%' + ('0' + c.charCodeAt(0).toString(16).toUpperCase()).slice(-2);
    });
  }

  // Short stable id of a quote (FNV-1a, 32 bit) used in #annot-<hash> links.
  function annotHash(exact) {
    var h = 0x811c9dc5;
    for (var i = 0; i < exact.length; i++) { h ^= exact.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    return ('0000000' + h.toString(16)).slice(-8);
  }

  function threadLink(sel) { return cfg.siteUrl + cfg.path + '#annot-' + annotHash(sel.exact); }
  function shareLink(sel) { return cfg.siteUrl + cfg.path + '#hl=' + encodeFragmentPart(sel.exact); }

  function escapeMarkdown(s) {
    return s.replace(/[\\`*_\[\]<>~|]/g, '\\$&').replace(/^([#>+\-]|\d+\.)/, '\\$1');
  }

  function buildCommentBody(sel, note, issue) {
    var links = '[§ 原文位置](' + threadLink(sel) + ')' + (issue ? ' · [⚑ Issue #' + issue.number + '](' + issue.url + ')' : '');
    return '> ' + escapeMarkdown(sel.exact) + '\n>\n> <sub>' + links + '</sub>\n\n' + note.trim() + '\n';
  }

  // ------------------------------------------------------------ selection

  function bindSelection() {
    document.addEventListener('selectionchange', function () {
      clearTimeout(selectionTimer);
      selectionTimer = setTimeout(updateToolbar, 250);
    });
    document.addEventListener('mouseup', function () { setTimeout(updateToolbar, 10); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { hideToolbar(); if (panelState && panelState.kind === 'thread') closePanel(); } });
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
    if (panelState && panelState.kind === 'editor') return;
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
    var below = rect.top - h - 10 < 60;
    toolbar.classList.toggle('is-below', below);
    var top = below ? rect.bottom + scrollY + 10 : rect.top + scrollY - h - 10;
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
      '<button type="button" class="annotation-tb-link" title="复制分享链接：打开后自动定位并高亮这段文字"><i class="fa fa-link"></i></button>' +
      '<span class="annotation-tb-arrow"></span>';
    toolbar.addEventListener('mousedown', function (e) { e.preventDefault(); }); // keep the selection
    toolbar.querySelector('.annotation-tb-comment').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      var offsets = range && rangeToOffsets(range);
      if (!offsets) { hideToolbar(); return; }
      var sel = selectorFromOffsets(offsets);
      var endNode = range.endContainer;
      hideToolbar();
      openComposer(sel, offsets, endNode);
    });
    toolbar.querySelector('.annotation-tb-link').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      var offsets = range && rangeToOffsets(range);
      if (!offsets) return;
      copyText(shareLink(selectorFromOffsets(offsets))).then(function () {
        showToast('分享链接已复制：打开后会自动定位并高亮这段文字');
      });
      hideToolbar();
    });
    document.body.appendChild(toolbar);
    return toolbar;
  }

  // ----------------------------------------------------------------- auth

  function getSession() {
    try {
      var raw = localStorage.getItem(SESSION_KEY);
      return raw ? JSON.parse(raw) : '';
    } catch (e) { return ''; }
  }

  function login() {
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

  // --------------------------------------------------------------- GitHub

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

  // Refresh the giscus list without blanking it: load a second, hidden iframe
  // with the same src and swap it in once it has reported.
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
  // knows the iframe it created); a changed comment count triggers a refetch.
  function bindGiscusMessages() {
    window.addEventListener('message', function (event) {
      if (event.origin !== GISCUS_ORIGIN) return;
      var g = event.data && event.data.giscus;
      if (!g) return;
      var frames = document.querySelectorAll('.giscus iframe.giscus-frame');
      for (var i = 0; i < frames.length; i++) {
        if (frames[i].contentWindow !== event.source) continue;
        if (g.resizeHeight) frames[i].style.height = g.resizeHeight + 'px';
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
    toast._timer = setTimeout(function () { toast.classList.remove('is-visible'); }, 3500);
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
    annotHash: annotHash,
    threadLink: threadLink,
    shareLink: shareLink,
    buildCommentBody: buildCommentBody,
    parseComment: parseComment,
    openThread: openThread,
    closePanel: closePanel,
    refreshGiscus: refreshGiscus,
    list: function () { return annotations; }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
