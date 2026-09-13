/**
 * annotations.js — highlight comments ("划线评论") on blog posts.
 *
 * Interaction (code-review / WeChat-reading style, no hover popups):
 *   - Select text in the article -> floating toolbar: 「赞」 / 「存疑」 / 「评论」 /
 *     「复制」 / 「搜一搜」 / 「分享」.
 *   - 「赞」 / 「存疑」 are anonymous per-passage counters (worker /reactions,
 *     D1; no login, one per browser in localStorage) — "raising a hand".
 *   - 「分享」 opens the article's share popover (js/share.js, window.BlogShare)
 *     for the passage link; completed shares are counted per passage (`share`
 *     in /reactions) and on the article.
 *     A passage with reactions but no comment is underlined too (its quote is
 *     stored server-side and re-anchored here); 存疑 shows as a red dotted line.
 *   - 「评论」 opens a large editor panel *in the flow*, right below the
 *     paragraph, with 取消 / 提交评论 bottom-right. If the selection lies inside
 *     an already-underlined passage the note joins that passage instead.
 *   - Every passage ends with a small marker (💬 comments · 👍 · ❓, non-zero
 *     ones); clicking it (or the underline) expands a thread panel below the
 *     paragraph: a 赞 / 存疑 row, all notes on that passage, their replies, and
 *     a box to add yours.
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
 * (from 「分享」 of a passage nobody commented on) flashes a passage. Both are handled here on load and on
 * hashchange — no Text Fragment directive, so no sticky purple browser
 * highlight and no dependency on the browser's matcher (footnote markers
 * broke it).
 *
 * The comment section at the bottom of the post is rendered here too (no giscus
 * iframe any more): the same discussion, the same commentEl / renderEditor, so
 * plain comments and passage notes share reply / edit / delete / 「同时提交
 * Issue」. giscus is kept only as the OAuth broker: login redirects to
 * giscus.app/api/oauth/authorize, which comes back with ?giscus=<session>; we
 * store it under localStorage["giscus-session"], the relay exchanges it for a
 * GitHub token and the browser calls GitHub GraphQL directly. Any failure
 * degrades to copying the Markdown for posting on GitHub.
 */

(function () {
  'use strict';

  var GISCUS_ORIGIN = 'https://giscus.app';
  var GITHUB_GRAPHQL = 'https://api.github.com/graphql';
  var GITHUB_MARKDOWN = 'https://api.github.com/markdown';
  var SESSION_KEY = 'giscus-session';
  var CONTEXT_CHARS = 32;
  var EXCLUDE_SELECTOR = '.comment, .pager, .related-posts, .footnotes, .reversefootnote, sup[id^="fnref"], a.footnote, ' +
    'script, style, noscript, svg, .katex, .mermaid, button, .anchorjs-link, .annotation-toolbar, .annotation-panel, .annotation-marker';
  var BLOCK_SELECTOR = 'p, li, pre, blockquote, h1, h2, h3, h4, h5, h6, dd, dt, figure, .highlight, table';
  var GHOST = { login: 'ghost', url: 'https://github.com/ghost', avatarUrl: 'https://avatars.githubusercontent.com/u/10137?s=64&v=4' };

  var cfg = null;
  var container = null;
  var index = null;          // { text, nodes:[{node, charIdx:[]}] }
  var comments = [];         // every top-level comment of the post's discussion (see parseComment)
  var annotations = [];      // the subset with a selector, anchored in the article
  var reactions = {};        // hash -> { hash, quote, up, doubt, range, marks }: anonymous passage 赞 / 存疑 (worker /reactions)
  var discussion = null;     // { id, url, totalCommentCount, likes: { up, mine } }
  var pageViews = null;      // number once GET/POST /views answered; stays null when the worker has no counter
  var loaded = false;        // loadDiscussion() has answered (either way)
  var loadError = null;
  var commentsHost = null;   // .annotation-comments in section.comment (the bottom comment section)
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
    // List pages have no comment section but do have action bars (js/share.js)
    // that vote through our auth helpers: pick up a returning OAuth session and
    // the API base from any element that carries it, then stop.
    var carrier = section || document.querySelector('[data-annotations-api]');
    var api = (localStorage.getItem('annotationsApi') || (carrier && carrier.getAttribute('data-annotations-api')) || '').replace(/\/$/, '');
    takeSessionFromUrl();
    cfg = { api: api, path: location.pathname };
    if (!section || !container || !window.InlinePopover) return;
    if (!api) return;

    cfg = {
      api: api,
      path: section.getAttribute('data-page-path') || location.pathname,
      siteUrl: (section.getAttribute('data-site-url') || location.origin).replace(/^http:/, 'https:').replace(/\/$/, ''),
      repoId: section.getAttribute('data-repo-id') || '',
      categoryId: section.getAttribute('data-category-id') || '',
      issues: section.getAttribute('data-issues') === '1',
      repo: section.getAttribute('data-repo') || '',
      section: section
    };

    bindSelection();
    initCommentSection();
    loadViews();
    window.addEventListener('hashchange', focusFromHash);
    whenRichContentSettled(function () {
      Promise.all([loadDiscussion(false), loadReactions()]).then(function () { if (!restoreDraft()) focusFromHash(); });
    });
  }

  // giscus' OAuth flow redirects back to `redirect_uri?giscus=<session>`; its
  // client.js used to store that — now we do (same key, same JSON encoding).
  function takeSessionFromUrl() {
    var url = new URL(location.href);
    var session = url.searchParams.get('giscus');
    if (!session) return;
    try { localStorage.setItem(SESSION_KEY, JSON.stringify(session)); } catch (e) { /* ignore */ }
    url.searchParams.delete('giscus');
    history.replaceState(history.state, '', url.toString());
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

    var orphans = [], items = [], anchoredHash = {};
    annotations.forEach(function (a) {
      a.range = anchor(a.selector);
      a.marks = [];
      if (a.range) { items.push({ start: a.range.start, end: a.range.end, id: a.id }); anchoredHash[annotHash(a.selector.exact)] = true; }
      else orphans.push(a);
    });
    // Passages with 赞 / 存疑 but no comment: anchor their stored quote so they
    // get the underline too (id 'r:<hash>'). A commented passage's reactions ride
    // on the comment marks.
    Object.keys(reactions).forEach(function (h) {
      var r = reactions[h];
      r.range = null; r.marks = [];
      if (!(r.up > 0 || r.doubt > 0) || anchoredHash[h]) return;
      r.range = anchor({ exact: r.quote });
      if (r.range) items.push({ start: r.range.start, end: r.range.end, id: 'r:' + h });
    });
    wrapPieces(items, 'annotation-hl').forEach(function (mark) {
      bindMark(mark);
      var doubt = false;
      mark.getAttribute('data-annotation-ids').split(' ').forEach(function (id) {
        var an = findAnnotation(id), r = null;
        if (an) { an.marks.push(mark); r = reactions[annotHash(an.selector.exact)]; }
        else if (id.indexOf('r:') === 0 && reactions[id.slice(2)]) { r = reactions[id.slice(2)]; r.marks.push(mark); }
        if (r && r.doubt > 0) doubt = true;
      });
      if (doubt) mark.classList.add('has-doubt');
    });
    insertMarkers();
    buildIndex();
    renderOrphans(orphans);
    if (panelState && panelState.kind === 'thread') refreshThreadPanel();
  }

  // Every underlined passage: `{ key, ids, list, hash, exact, reaction, marks }` —
  // `list` = the annotations sharing that exact range (may be empty for a
  // reaction-only passage, whose ids are ['r:<hash>']).
  function passages() {
    var groups = {}, out = [];
    annotations.forEach(function (a) {
      if (!a.range || !a.marks.length) return;
      var key = a.range.start + '-' + a.range.end;
      if (!groups[key]) { groups[key] = { key: key, list: [], marks: a.marks }; out.push(groups[key]); }
      groups[key].list.push(a);
    });
    out.forEach(function (g) {
      g.exact = g.list[0].selector.exact; g.hash = annotHash(g.exact);
      g.ids = g.list.map(function (a) { return a.id; }); g.reaction = reactions[g.hash] || null;
    });
    Object.keys(reactions).forEach(function (h) {
      var r = reactions[h];
      if (!r.marks.length) return;
      out.push({ key: r.range.start + '-' + r.range.end, ids: ['r:' + h], list: [], hash: h, exact: r.quote, reaction: r, marks: r.marks });
    });
    return out;
  }
  // The passage behind a set of ids (annotation ids or 'r:<hash>'); null when it
  // is not on the page. A reaction-only passage that just got its first comment
  // is found again through the hash.
  function passageFor(ids) {
    var hash = null, a = null;
    for (var i = 0; i < ids.length && !hash; i++) {
      if (ids[i].indexOf('r:') === 0) hash = ids[i].slice(2);
      else if ((a = findAnnotation(ids[i]))) hash = annotHash(a.selector.exact);
    }
    if (!hash) return null;
    var all = passages();
    for (var j = 0; j < all.length; j++) if (all[j].hash === hash) return all[j];
    return null;
  }
  function commentCount(p) { return p.list.reduce(function (n, a) { return n + (a.deleted ? 0 : 1) + (a.replies || []).length; }, 0); }

  // One marker per passage: 💬 comments · 👍 up · ❓ doubt (only the non-zero ones).
  function markerHtml(p) {
    var r = p.reaction, n = commentCount(p);
    return (n ? '<i class="fa fa-comment"></i><span class="annotation-marker-count">' + n + '</span>' : '') +
      (r && r.up ? '<i class="fa fa-thumbs-up"></i><span class="annotation-marker-count">' + r.up + '</span>' : '') +
      (r && r.doubt ? '<i class="fa fa-question-circle"></i><span class="annotation-marker-count">' + r.doubt + '</span>' : '');
  }
  function markerTitle(p) {
    var r = p.reaction, parts = [], n = commentCount(p);
    if (n) parts.push(n + ' 条评论');
    if (r && r.up) parts.push(r.up + ' 人赞');
    if (r && r.doubt) parts.push(r.doubt + ' 人存疑');
    if (r && r.share) parts.push(r.share + ' 次分享');
    return parts.join(' · ') + '，点击查看';
  }
  function insertMarkers() {
    passages().forEach(function (p) {
      var last = p.marks[p.marks.length - 1];
      var marker = document.createElement('span');
      marker.className = 'annotation-marker' + (p.reaction && p.reaction.doubt ? ' has-doubt' : '');
      marker.setAttribute('data-annotation-ids', p.ids.join(' '));
      marker.setAttribute('data-hash', p.hash);
      marker.setAttribute('title', markerTitle(p));
      marker.innerHTML = markerHtml(p);
      marker.addEventListener('click', function (e) {
        e.preventDefault(); e.stopPropagation();
        toggleThread(p.ids, last);
      });
      last.insertAdjacentElement('afterend', marker);
    });
  }

  function reanchorAll() {
    if (annotations.length || Object.keys(reactions).length) applyHighlights(); else buildIndex();
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
      var p = passageFor(['r:' + m[1]]);
      if (p) {
        scrollIntoViewInstant(p.marks[0]);
        flashMarks(p.marks);
        openThread(p.ids, p.marks[p.marks.length - 1]);
        return;
      }
      showToast('这条评论对应的原文找不到了（可能已被修改）');
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
    var p = passageFor(ids);
    if (!p) return;
    panelState = { kind: 'thread', ids: p.ids };
    ensurePanel();
    panel.className = 'annotation-panel is-thread';
    renderThread(p);
    mountPanel(anchorMark || p.marks[p.marks.length - 1]);
    if (window.getSelection) window.getSelection().removeAllRanges();
  }

  function refreshThreadPanel() {
    var p = passageFor(panelState.ids);
    if (!p || !p.marks.length) { closePanel(); return; }
    panelState.ids = p.ids;
    renderThread(p);
    if (!panel.parentNode) mountPanel(p.marks[p.marks.length - 1]);
  }

  // 赞 / 存疑 row of the thread panel (also re-rendered alone after a click).
  function reactBarHtml(p) {
    var r = p.reaction || { up: 0, doubt: 0, share: 0 }, up = myReaction(p.hash, 'up'), doubt = myReaction(p.hash, 'doubt');
    return '<button type="button" class="ap-react-btn ap-react-up' + (up ? ' is-on' : '') + '" title="' + (up ? '取消赞' : '赞这段话（不用登录）') + '"><i class="fa ' + (up ? 'fa-thumbs-up' : 'fa-thumbs-o-up') + '"></i> 赞' + (r.up ? ' <b>' + r.up + '</b>' : '') + '</button>' +
      '<button type="button" class="ap-react-btn ap-react-doubt' + (doubt ? ' is-on' : '') + '" title="' + (doubt ? '取消存疑' : '觉得这段话有问题？（不用登录）') + '"><i class="fa ' + (doubt ? 'fa-question-circle' : 'fa-question-circle-o') + '"></i> 存疑' + (r.doubt ? ' <b>' + r.doubt + '</b>' : '') + '</button>' +
      '<button type="button" class="ap-react-btn ap-react-share" title="分享这段话（微博 / X / 微信 / 复制链接）" aria-haspopup="true" aria-expanded="false"><i class="fa fa-share-alt"></i> 分享' + (r.share ? ' <b>' + r.share + '</b>' : '') + '</button>' +
      (doubt ? '<a href="#" class="ap-react-say">说说哪里不对 →</a>' : '');
  }
  function bindReactBar(host, p) {
    host.querySelector('.ap-react-up').addEventListener('click', function () { react(p.exact, 'up'); });
    host.querySelector('.ap-react-doubt').addEventListener('click', function () { react(p.exact, 'doubt'); });
    host.querySelector('.ap-react-share').addEventListener('click', function (e) { e.stopPropagation(); sharePassage(e.currentTarget, p.exact, p.list.length > 0); });
    var say = host.querySelector('.ap-react-say');
    if (say) say.addEventListener('click', function (e) {
      e.preventDefault();
      var ta = panel.querySelector('.ap-text');
      if (ta) { ta.focus(); ta.scrollIntoView({ block: 'nearest' }); }
    });
  }

  function renderThread(p) {
    var list = p.list, primary = list[0] || null;
    var selector = primary ? primary.selector : { exact: p.exact };
    var total = commentCount(p);
    panel.innerHTML =
      '<div class="ap-head">' +
        '<i class="fa fa-quote-left"></i><span class="ap-quote" title="' + escapeAttr(p.exact) + '">' + escapeHtml(p.exact) + '</span>' +
        '<span class="ap-count">' + (total ? total + ' 条评论' : '还没有评论') + '</span>' +
        '<button type="button" class="ap-close" title="收起">×</button>' +
      '</div>' +
      '<div class="ap-react"></div>' +
      '<div class="ap-thread"></div>' +
      '<div class="ap-editor"></div>';
    panel.querySelector('.ap-close').addEventListener('click', closePanel);
    var reactHost = panel.querySelector('.ap-react');
    reactHost.innerHTML = reactBarHtml(p);
    bindReactBar(reactHost, p);
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
        return replyTo ? postReply(replyTo.annotation, text) : postAnnotation(selector, text, extra.issue);
      },
      fallbackText: function (text) { return replyTo ? text : buildCommentBody(selector, text); }
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
    var mine = !c.deleted && viewer && c.author && viewer.login === c.author.login;
    if (c.deleted) onReply = null;
    el.className = 'ap-comment' + (isReply ? ' is-reply' : '') + (c.issue ? ' has-issue' : '') + (c.deleted ? ' is-deleted' : '');
    el.setAttribute('data-comment-id', c.id);
    el.innerHTML =
      '<a class="ap-avatar" href="' + escapeAttr(c.author.url) + '" target="_blank" rel="noopener noreferrer"><img src="' + escapeAttr(c.author.avatarUrl) + '" alt=""></a>' +
      '<div class="ap-comment-main">' +
        '<div class="ap-comment-meta"><a href="' + escapeAttr(c.author.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(c.author.login) + '</a>' +
          (c.owner && !c.deleted ? '<span class="ap-owner" title="博客作者">作者</span>' : '') +
          '<time datetime="' + escapeAttr(c.createdAt) + '" title="' + escapeAttr(new Date(c.createdAt).toLocaleString()) + '">' + relativeTime(c.createdAt) + '</time>' +
          (c.lastEditedAt ? '<span class="ap-edited" title="' + escapeAttr(new Date(c.lastEditedAt).toLocaleString()) + '">已编辑</span>' : '') +
          (c.issue ? '<a class="ap-issue-badge" href="' + escapeAttr(c.issue.url) + '" target="_blank" rel="noopener noreferrer" title="已同时提交为 GitHub Issue"><i class="fa fa-flag"></i> Issue' + (c.issue.number ? ' #' + c.issue.number : '') + '</a>' : '') +
          '<span class="ap-meta-actions">' +
            (c.deleted ? '' : voteHtml(c)) +
            (onReply ? '<button type="button" class="ap-reply-btn"><i class="fa fa-reply"></i> 回复</button>' : '') +
            (mine ? '<button type="button" class="ap-edit-btn" title="编辑"><i class="fa fa-pencil"></i> 编辑</button>' +
                    '<button type="button" class="ap-delete-btn" title="删除"><i class="fa fa-trash-o"></i> 删除</button>' : '') +
            '<a class="ap-github" href="' + escapeAttr(c.url) + '" target="_blank" rel="noopener noreferrer" title="在 GitHub 上查看 / 编辑"><i class="fa fa-github"></i></a>' +
          '</span>' +
        '</div>' +
        '<div class="ap-comment-body"></div>' +
      '</div>';
    var body = el.querySelector('.ap-comment-body');
    if (c.deleted) body.innerHTML = '<em class="ap-muted">此评论已删除</em>';
    else {
      body.appendChild(sanitizeHtml(html));
      InlinePopover.renderMathIfPresent(body);
      // same hash twice fires no hashchange — re-run the focus by hand
      body.addEventListener('click', function (e) {
        var a = e.target.closest && e.target.closest('a[href^="#"]');
        if (a && location.hash === a.getAttribute('href')) { e.preventDefault(); focusFromHash(); }
      });
    }
    if (onReply) el.querySelector('.ap-reply-btn').addEventListener('click', onReply);
    if (!c.deleted) bindVote(el.querySelector('.ap-vote'), c);
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
        initialText: c.selector ? stripQuoteHeader(raw) : raw,
        onCancel: cancel,
        onSubmit: function (text) {
          var body = c.selector ? buildCommentBody(c.selector, text, c.issue) : text;
          return graphql(UPDATE_COMMENT, { id: c.id, body: body }).then(function (res) {
            c.bodyHTML = res.updateDiscussionComment.comment.bodyHTML;
            c.lastEditedAt = new Date().toISOString();
            if (!isReply) { var keep = c.selector; c.selector = null; c.noteHTML = null; c.issue = null; parseBodyHeader(c); if (!c.selector && keep) { c.selector = keep; c.noteHTML = c.bodyHTML; } }
            syncViews();
            flashComment(c.id);
            showToast('已保存');
          });
        }
      });
      ta.focus();
    }).catch(function (err) { host.innerHTML = '<span class="ap-status is-error">' + escapeHtml(err.message) + '</span>'; setTimeout(cancel, 2500); });
  }

  function deleteComment(el, c, isReply, parent) {
    var n = !isReply && c.replies && c.replies.length;
    // GitHub soft-deletes a comment that has replies: the replies stay and the
    // comment shows as 「此评论已删除」 (same as on GitHub). A highlighted passage
    // loses its anchor with the quote header, so it disappears from the article.
    var msg = n
      ? (c.selector ? '删除这条评论？它的 ' + n + ' 条回复会保留（显示为「此评论已删除」），但这段划线会从文中消失。' : '删除这条评论？它的 ' + n + ' 条回复会保留，原位显示为「此评论已删除」。')
      : '删除这条评论？';
    if (!window.confirm(msg)) return;
    el.classList.add('is-deleting');
    graphql(DELETE_COMMENT, { id: c.id }).then(function () {
      if (isReply) {
        parent.replies = parent.replies.filter(function (r) { return r.id !== c.id; });
        parent.replyCount = parent.replies.length;
      } else if (c.replies.length) {
        c.deleted = true; c.bodyHTML = ''; c.selector = null; c.noteHTML = null; c.issue = null;
        if (panelState) panelState.ids = panelState.ids.filter(function (id) { return id !== c.id; });
      } else {
        comments = comments.filter(function (a) { return a.id !== c.id; });
        if (panelState) panelState.ids = panelState.ids.filter(function (id) { return id !== c.id; });
      }
      syncViews(); // re-renders (or closes) the thread panel and the comment section
      showToast('已删除');
    }).catch(function (err) { el.classList.remove('is-deleting'); showToast('删除失败：' + err.message); });
  }

  // ------------------------------------------------- bottom comment section

  // The classic comment list under the post: the same discussion, the same
  // commentEl / renderEditor as the in-article panel, so every comment (with or
  // without a passage) gets the same reply / edit / delete / issue controls.
  function initCommentSection() {
    commentsHost = cfg.section.querySelector('.annotation-comments');
    if (!commentsHost) return;
    commentsHost.innerHTML =
      '<div class="ac-head"><span class="ac-count">正在加载评论…</span></div>' +
      '<div class="ac-hot"></div>' +
      '<div class="ac-list"></div>' +
      '<div class="ap-editor ac-editor"></div>';
    renderEditor(commentsHost.querySelector('.ac-editor'), {
      placeholder: '写下你的评论…（想针对某句话说？选中正文里的文字，点「评论」）',
      submitLabel: '发表评论',
      issueOption: true,
      clearOnSubmit: true,
      initialText: readCommentDraft(),
      onChange: saveCommentDraft,
      beforeLogin: saveCommentDraft,
      onSubmit: function (text, extra) { return postComment(text, extra.issue).then(function () { saveCommentDraft(''); }); }
    });
  }

  function renderCommentSection() {
    if (!commentsHost) return;
    renderLikeBar();
    renderHotPassages();
    var list = commentsHost.querySelector('.ac-list');
    list.innerHTML = '';
    if (!comments.length) {
      if (!loadError) list.innerHTML = '<p class="ac-empty">还没有评论。可以在下方留言，也可以选中正文任意文字，对那句话发表评论。</p>';
      return;
    }
    comments.forEach(function (c) {
      var group = document.createElement('div');
      group.className = 'ac-group';
      group.appendChild(commentEl(c, c.bodyHTML, false, function () { openInlineReply(c, c.author.login, group); }, null));
      c.replies.forEach(function (r) {
        group.appendChild(commentEl(r, r.bodyHTML, true, function () { openInlineReply(c, r.author.login, group); }, c));
      });
      list.appendChild(group);
    });
  }

  // 「最受关注的段落」: the anchored passages ranked by votes + activity, Medium's
  // "top highlight" — only when there is something to rank (2+ passages), max 3.
  var HOT_MAX = 3;
  function renderHotPassages() {
    var host = commentsHost.querySelector('.ac-hot');
    // score = passage 赞 + 2 × 存疑 + 2 × net comment votes + comments
    var ranked = passages().map(function (p) {
      var votes = 0, n = commentCount(p), r = p.reaction || { up: 0, doubt: 0 };
      p.list.forEach(function (a) {
        votes += a.votes.up - a.votes.down;
        a.replies.forEach(function (x) { votes += x.votes.up - x.votes.down; });
      });
      return { p: p, votes: votes, comments: n, up: r.up, doubt: r.doubt, score: r.up + 2 * r.doubt + votes * 2 + n };
    }).filter(function (x) { return x.score > 0; }).sort(function (x, y) { return y.score - x.score; });
    if (ranked.length < 2) { host.innerHTML = ''; return; }
    host.innerHTML = '<div class="ac-hot-title"><i class="fa fa-fire"></i> 最受关注的段落</div>';
    ranked.slice(0, HOT_MAX).forEach(function (x) {
      var p = x.p, meta = [];
      if (x.up) meta.push('<i class="fa fa-thumbs-up"></i> ' + x.up);
      if (x.doubt) meta.push('<i class="fa fa-question-circle"></i> ' + x.doubt);
      if (x.votes) meta.push('<i class="fa fa-caret-up"></i> ' + x.votes);
      if (x.comments) meta.push(x.comments + ' 条评论');
      var item = document.createElement('a');
      item.className = 'ac-hot-item';
      item.href = '#annot-' + p.hash;
      item.innerHTML =
        '<span class="ac-hot-quote">' + escapeHtml(p.exact) + '</span>' +
        '<span class="ac-hot-meta">' + meta.join(' · ') + '</span>';
      item.addEventListener('click', function (e) {
        e.preventDefault();
        var q = passageFor(p.ids);
        if (!q) return;
        scrollIntoViewInstant(q.marks[0]);
        flashMarks(q.marks);
        openThread(q.ids, q.marks[q.marks.length - 1]);
      });
      host.appendChild(item);
    });
  }

  // Head of the comment section: comment count and the GitHub link. Re-rendered
  // on its own after a views / load response so the editors below are left alone.
  function renderLikeBar() {
    if (!commentsHost) return;
    var head = commentsHost.querySelector('.ac-head');
    var total = comments.reduce(function (n, c) { return n + (c.deleted ? 0 : 1) + c.replies.length; }, 0);
    head.innerHTML =
      '<span class="ac-count">' + (loadError ? '<i class="fa fa-exclamation-circle"></i> 评论加载失败：' + escapeHtml(loadError.message)
        : !loaded ? '正在加载评论…' : '<i class="fa fa-comment-o"></i> ' + total + ' 条评论') + '</span>' +
      (discussion && discussion.url ? '<a class="ac-github" href="' + escapeAttr(discussion.url) + '" target="_blank" rel="noopener noreferrer" title="这个讨论串在 GitHub Discussions 上"><i class="fa fa-github"></i> GitHub</a>' : '');
    // Header meta badges (`.post-stats`) are painted by js/share.js; hand it
    // what we know (views once the worker answered, comment count once loaded).
    var stats = {};
    if (pageViews !== null) stats.views = pageViews;
    if (loaded) stats.comments = total;
    if (Object.keys(stats).length) { try { document.dispatchEvent(new CustomEvent('blog:stats', { detail: stats })); } catch (e) { /* old browsers */ } }
  }

  // Reply box right under the comment's replies (only one open at a time).
  function openInlineReply(c, mention, group) {
    var old = commentsHost.querySelector('.ac-reply-editor');
    if (old) old.parentNode.removeChild(old);
    var host = document.createElement('div');
    host.className = 'ap-editor ac-reply-editor';
    group.appendChild(host);
    var ta = renderEditor(host, {
      placeholder: '回复 @' + mention + '…',
      submitLabel: '回复',
      compact: true,
      inline: true,
      initialText: mention !== c.author.login ? '@' + mention + ' ' : '',
      onCancel: function () { host.parentNode.removeChild(host); },
      onSubmit: function (text) { return postReply(c, text); }
    });
    ta.focus();
  }

  function commentDraftKey() { return 'commentDraft:' + cfg.path; }
  function saveCommentDraft(text) {
    try { if (text && text.trim()) sessionStorage.setItem(commentDraftKey(), text); else sessionStorage.removeItem(commentDraftKey()); } catch (e) { /* ignore */ }
  }
  function readCommentDraft() { try { return sessionStorage.getItem(commentDraftKey()) || ''; } catch (e) { return ''; } }

  function logout() {
    try { localStorage.removeItem(SESSION_KEY); } catch (e) { /* ignore */ }
    token = null; viewer = null;
    try { document.dispatchEvent(new CustomEvent('blog:viewer', { detail: null })); } catch (e) { /* old browsers */ }
    var hosts = document.querySelectorAll('.ap-editor');
    for (var i = 0; i < hosts.length; i++) if (hosts[i].querySelector('.ap-user')) refreshAuthUI(hosts[i]);
    if (discussion) discussion.likes.mine = null; // the counts stay, our own vote marks go
    comments.forEach(function (c) { c.votes.mine = null; c.replies.forEach(function (r) { r.votes.mine = null; }); });
    renderCommentSection();
    if (panelState && panelState.kind === 'thread') refreshThreadPanel();
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
    { key: 'table', icon: 'fa-table', title: '表格 (3×3)' },
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
        if (opts.clearOnSubmit && host.isConnected) {
          ta.value = '';
          if (issueBox) issueBox.checked = false;
          switchTab(host, 'write');
          setStatus(host, '');
          updateSubmitState();
        }
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
      case 'table':
        var before = (s > 0 && v.charAt(s - 1) !== '\n') ? '\n' : '';
        var after = (e < v.length && v.charAt(e) !== '\n') ? '\n' : '';
        var table = '| 标题 | 标题 | 标题 |\n| --- | --- | --- |\n| 内容 | 内容 | 内容 |\n| 内容 | 内容 | 内容 |';
        ta.setRangeText(before + table + after, s, e, 'select');
        ta.setSelectionRange(s + before.length + 2, s + before.length + 4);
        break;
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
      userEl.innerHTML = '<span class="ap-muted">登录 GitHub 后即可发表</span>';
      loginBtn.style.display = '';
      submitBtn.style.display = 'none';
      return;
    }
    loginBtn.style.display = 'none';
    submitBtn.style.display = '';
    if (viewer) { renderViewer(userEl, viewer); return; }
    userEl.innerHTML = '<span class="ap-muted">正在连接 GitHub…</span>';
    fetchViewer();
  }

  var VIEWER_QUERY = '{ viewer { login avatarUrl url } }';
  var viewerPending = null;
  // One request at a time; onViewerKnown() updates every editor on success.
  function fetchViewer() {
    if (viewer || viewerPending) return;
    viewerPending = graphql(VIEWER_QUERY).then(function (data) {
      viewerPending = null;
      viewer = data.viewer;
      onViewerKnown();
    }).catch(function (err) {
      viewerPending = null;
      var hosts = document.querySelectorAll('.ap-editor');
      for (var i = 0; i < hosts.length; i++) {
        var u = hosts[i].querySelector('.ap-user');
        if (u) u.innerHTML = '<span class="ap-muted">' + escapeHtml(err.message) + '</span>';
        if (!getSession()) refreshAuthUI(hosts[i]);
      }
    });
  }

  // The reader's identity just became known: show it in every editor and
  // re-render the lists so their own comments get 编辑 / 删除.
  function onViewerKnown() {
    var hosts = document.querySelectorAll('.ap-editor');
    for (var i = 0; i < hosts.length; i++) if (hosts[i].querySelector('.ap-user')) refreshAuthUI(hosts[i]);
    if (panelState && panelState.kind === 'thread' && !panel.querySelector('.ap-text').value) refreshThreadPanel();
    if (commentsHost && !commentsHost.querySelector('.ac-reply-editor, .ap-inline-editor')) renderCommentSection();
    loadViewerReactions();
    // Other scripts (js/share.js: author-only buttons) want to know who is logged in.
    try { document.dispatchEvent(new CustomEvent('blog:viewer', { detail: viewer })); } catch (e) { /* old browsers */ }
  }

  function renderViewer(userEl, v) {
    userEl.innerHTML =
      '<img src="' + escapeAttr(v.avatarUrl) + '" alt="" width="22" height="22"> ' +
      '<a href="' + escapeAttr(v.url) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(v.login) + '</a>' +
      '<button type="button" class="ap-logout" title="退出 GitHub 登录">退出</button>';
    userEl.querySelector('.ap-logout').addEventListener('click', logout);
  }

  // --------------------------------------------------------- new annotation

  // The underlined passage (commented or reaction-only) that contains the
  // selection — smallest wins — or null. A selection inside it joins it instead
  // of starting a second thread / counter for the same place.
  function passageContaining(offsets) {
    var best = null;
    passages().forEach(function (p) {
      var range = p.list.length ? p.list[0].range : p.reaction.range;
      if (!range || range.start > offsets.start || range.end < offsets.end) return;
      if (!best || (range.end - range.start) < best.len) best = { p: p, len: range.end - range.start };
    });
    return best ? best.p : null;
  }

  function openComposer(sel, offsets, anchorNode, draftText) {
    var join = passageContaining(offsets);
    if (join) {
      // same passage -> one thread; the editor there defaults to a comment on the passage
      openThread(join.ids, join.marks[join.marks.length - 1]);
      var joinTa = panel && panel.querySelector('.ap-text');
      if (joinTa) { if (draftText) joinTa.value = draftText; joinTa.focus(); joinTa.dispatchEvent(new Event('input')); }
      return;
    }
    panelState = { kind: 'editor', selector: sel, offsets: offsets };
    ensurePanel();
    panel.className = 'annotation-panel is-editor';
    panel.innerHTML =
      '<div class="ap-head">' +
        '<i class="fa fa-quote-left"></i><span class="ap-quote" title="' + escapeAttr(sel.exact) + '">' + escapeHtml(sel.exact) + '</span>' +
        '<span class="ap-count">新评论</span>' +
        '<button type="button" class="ap-close" title="取消">×</button>' +
      '</div>' +
      '<div class="ap-editor"></div>';
    panel.querySelector('.ap-close').addEventListener('click', cancelComposer);
    var ta = renderEditor(panel.querySelector('.ap-editor'), {
      placeholder: '写下你对这段文字的评论…',
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
      var a = parseComment(c);
      if (!a.selector) { a.selector = sel; a.noteHTML = c.bodyHTML; a.issue = issue; } // GitHub rendered it unexpectedly
      comments.push(a);
      clearDraft();
      closePanel();
      syncViews();
      if (a.marks.length) {
        flashMarks(a.marks);
        openThread(groupIdsFor(a), a.marks[a.marks.length - 1]);
      }
      flashComment(a.id);
      showToast(issue ? '评论已发表，Issue #' + issue.number + ' 已创建' : '评论已发表');
    });
  }

  // A plain comment from the section at the bottom (no passage). With an
  // issue the body starts with `<sub>[⚑ Issue #N](url)</sub>` so it can be
  // recognised (parseBodyHeader) and shown with the flag badge.
  function postComment(text, withIssue) {
    var issue = null;
    return ensureDiscussion().then(function (id) {
      return (withIssue ? createIssue(null, text) : Promise.resolve(null)).then(function (is) {
        issue = is;
        var body = (issue ? '<sub>[⚑ Issue #' + issue.number + '](' + issue.url + ')</sub>\n\n' : '') + text.trim() + '\n';
        return graphql(ADD_COMMENT, { body: body, discussionId: id });
      });
    }).then(function (data) {
      var c = parseComment(data.addDiscussionComment.comment);
      if (issue && !c.issue) c.issue = issue;
      comments.push(c);
      syncViews();
      flashComment(c.id);
      showToast(issue ? '评论已发表，Issue #' + issue.number + ' 已创建' : '评论已发表');
    });
  }

  // `sel` is null for a plain comment.
  function createIssue(sel, text) {
    var titleEl = document.querySelector('.page-header .title, .post-heading h1, h1');
    var postTitle = (titleEl && titleEl.textContent) || document.title.split(/\s[-|]\s/)[0];
    var snippet = sel ? sel.exact : text.trim().split('\n')[0].replace(/^[#>*\-\s]+/, '');
    if (snippet.length > 40) snippet = snippet.slice(0, 40) + '…';
    var body = sel
      ? '> ' + escapeMarkdown(sel.exact) + '\n>\n> [§ 原文位置](' + threadLink(sel) + ')\n\n' + text.trim() + '\n'
      : text.trim() + '\n\n— 来自文章评论区：' + cfg.siteUrl + cfg.path + '#comments\n';
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
      a.replies = (a.replies || []).concat([{ id: c.id, url: c.url, author: c.author || GHOST, createdAt: c.createdAt, lastEditedAt: null, bodyHTML: c.bodyHTML, owner: c.authorAssociation === 'OWNER', votes: parseVotes(null) }]);
      a.replyCount = a.replies.length;
      syncViews(); // marker counts, panel, comment section
      flashComment(c.id);
      showToast('回复已发表');
    });
  }

  // Briefly tint every rendering of a comment (panel and bottom section).
  function flashComment(id) {
    var els = document.querySelectorAll('.ap-comment[data-comment-id="' + id + '"]');
    for (var i = 0; i < els.length; i++) els[i].classList.add('is-new');
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

  function loadDiscussion(fresh) {
    var qs = '?term=' + encodeURIComponent(cfg.path) + (fresh ? '&t=' + Date.now() : '');
    return api('/discussions' + qs).then(function (data) {
      var d = data && data.discussion;
      discussion = d ? { id: d.id, url: d.url, totalCommentCount: d.totalCommentCount, likes: parseVotes(d.reactions || d.reactionGroups) } : null;
      comments = d ? parseComments(d.comments || []) : [];
      loadError = null; loaded = true;
      syncViews();
      if (viewer) loadViewerReactions(); // the anonymous payload cannot know what *we* voted
      return annotations;
    }).catch(function (err) {
      console.warn('[annotations] load failed:', err.message);
      loadError = err; loaded = true;
      applyHighlights(); // reaction-only passages still get their underline
      renderCommentSection();
      return [];
    });
  }

  // `comments` is the source of truth; the article highlights and the comment
  // section at the bottom are two views of it. Call after every mutation.
  function syncViews() {
    annotations = comments.filter(function (c) { return c.selector; });
    applyHighlights();
    renderCommentSection();
  }

  function parseComments(comments) {
    var out = [];
    for (var i = 0; i < comments.length; i++) {
      var a = parseComment(comments[i]);
      if (a) out.push(a);
    }
    return out;
  }

  // Every top-level comment of the discussion becomes a record; the ones that
  // carry the quote header additionally get `selector` / `noteHTML` and are
  // highlighted in the article. Deleted-but-with-replies comments are kept as
  // placeholders (GitHub soft-deletes those) so their replies stay readable.
  function parseComment(c) {
    if (!c || c.isMinimized) return null;
    var replies = parseReplies(c.replies);
    if (c.deletedAt && !replies.length) return null;
    var rec = {
      id: c.id,
      url: c.url,
      author: c.author || GHOST,
      createdAt: c.createdAt,
      lastEditedAt: c.lastEditedAt || null,
      deleted: !!c.deletedAt,
      owner: c.authorAssociation === 'OWNER',
      votes: parseVotes(c.reactions || c.reactionGroups),
      replyCount: (c.replies && (c.replies.totalCount !== undefined ? c.replies.totalCount : c.replies.length)) || c.replyCount || 0,
      replies: replies,
      bodyHTML: c.deletedAt ? '' : (c.bodyHTML || ''),
      selector: null, noteHTML: null, issue: null
    };
    if (!rec.deleted) parseBodyHeader(rec);
    return rec;
  }

  // Fill rec.selector / noteHTML / issue from the comment's HTML (see file header).
  function parseBodyHeader(rec) {
    var doc = new DOMParser().parseFromString('<div>' + rec.bodyHTML + '</div>', 'text/html');
    var root = doc.body.firstChild;
    var first = root.firstElementChild;
    if (!first) return;
    var issueLink, issueNo;
    if (first.tagName !== 'BLOCKQUOTE') {
      // plain comment filed with 「同时提交 Issue」: leading <p><sub>[⚑ Issue #N](…)</sub></p>
      if (first.tagName === 'P' && first.querySelector('sub a[href*="/issues/"]') && first.textContent.trim().length < 40) {
        issueLink = first.querySelector('a[href*="/issues/"]');
        issueNo = /\/issues\/(\d+)/.exec(issueLink.getAttribute('href'));
        rec.issue = { url: issueLink.getAttribute('href'), number: issueNo ? +issueNo[1] : 0 };
      }
      return;
    }
    var quote = first;
    var links = quote.querySelectorAll('a[href*="#annot-"], a[href*=":~:text="]');
    var link = null;
    for (var i = 0; i < links.length; i++) {
      try {
        var u = new URL(links[i].getAttribute('href'));
        if (u.pathname === cfg.path) { link = links[i]; break; }
      } catch (e) { /* ignore */ }
    }
    if (!link) return;
    var fragment = parseTextFragment(link.getAttribute('href'));
    var linkBlock = link.closest('p, sub') || link;
    while (linkBlock.parentNode !== quote && linkBlock.parentNode !== root) linkBlock = linkBlock.parentNode;
    issueLink = linkBlock.querySelector('a[href*="/issues/"]');
    issueNo = issueLink && /\/issues\/(\d+)/.exec(issueLink.getAttribute('href'));
    linkBlock.parentNode.removeChild(linkBlock);
    var exact = quote.textContent.replace(/\s+/g, ' ').trim();
    if (!exact) return;
    root.removeChild(quote);
    rec.noteHTML = root.innerHTML;
    rec.selector = { exact: exact, prefix: fragment.prefix, suffix: fragment.suffix };
    rec.issue = issueLink ? { url: issueLink.getAttribute('href'), number: issueNo ? +issueNo[1] : 0 } : null;
  }

  // giscus' adapter returns replies as a plain array; GitHub GraphQL as {nodes}.
  function parseReplies(replies) {
    var list = Array.isArray(replies) ? replies : (replies && replies.nodes) || [];
    return list.filter(function (r) { return r && !r.deletedAt && !r.isMinimized; }).map(function (r) {
      return { id: r.id, url: r.url, createdAt: r.createdAt, lastEditedAt: r.lastEditedAt || null, bodyHTML: r.bodyHTML, author: r.author || GHOST,
        owner: r.authorAssociation === 'OWNER', votes: parseVotes(r.reactions || r.reactionGroups) };
    });
  }

  // 👍 / 👎 GitHub reactions are our up / down votes. giscus' adapter ships them as
  // { THUMBS_UP: { count, viewerHasReacted }, … }, GitHub GraphQL as
  // reactionGroups [{ content, viewerHasReacted, reactors { totalCount } }].
  function parseVotes(src) {
    var v = { up: 0, down: 0, mine: null };
    function take(content, n, mine) {
      var dir = content === 'THUMBS_UP' ? 'up' : content === 'THUMBS_DOWN' ? 'down' : null;
      if (!dir) return;
      v[dir] = n || 0;
      if (mine) v.mine = dir;
    }
    if (Array.isArray(src)) src.forEach(function (g) { take(g.content, g.reactors ? g.reactors.totalCount : (g.users ? g.users.totalCount : 0), g.viewerHasReacted); });
    else if (src) Object.keys(src).forEach(function (k) { take(k, src[k].count, src[k].viewerHasReacted); });
    return v;
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
    // A backslash only escapes ASCII punctuation: `1\.` not `\1.` (the latter
    // rendered literally and broke the quote's hash for passages like "1.5px").
    return s.replace(/[\\`*_\[\]<>~|]/g, '\\$&').replace(/^[#>+\-]/, '\\$&').replace(/^(\d+)([.)])/, '$1\\$2');
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
      '<button type="button" class="annotation-tb-up" title="赞这段话（不用登录）"><i class="fa fa-thumbs-o-up"></i> 赞</button>' +
      '<button type="button" class="annotation-tb-doubt" title="觉得这段话有问题？存疑（不用登录）"><i class="fa fa-question-circle-o"></i> 存疑</button>' +
      '<span class="annotation-tb-sep"></span>' +
      '<button type="button" class="annotation-tb-comment"><i class="fa fa-comment-o"></i> 评论</button>' +
      '<button type="button" class="annotation-tb-copy" title="复制选中的文字"><i class="fa fa-copy"></i> 复制</button>' +
      '<button type="button" class="annotation-tb-search" title="用 Google 搜这段文字"><i class="fa fa-search"></i> 搜一搜</button>' +
      '<button type="button" class="annotation-tb-share" title="分享这段话：微博 / X / 微信 / 复制链接（打开后自动定位这段文字）" aria-haspopup="true" aria-expanded="false"><i class="fa fa-share-alt"></i> 分享</button>' +
      '<span class="annotation-tb-arrow"></span>';
    toolbar.addEventListener('mousedown', function (e) { e.preventDefault(); }); // keep the selection
    ['up', 'doubt'].forEach(function (kind) {
      toolbar.querySelector('.annotation-tb-' + kind).addEventListener('click', function (e) {
        e.stopPropagation();
        var range = currentRange();
        var offsets = range && rangeToOffsets(range);
        hideToolbar();
        if (!offsets) return;
        // inside an underlined passage -> react on that passage, not on a new sub-range
        var p = passageContaining(offsets);
        var exact = p ? p.exact : selectorFromOffsets(offsets).exact;
        if (window.getSelection) window.getSelection().removeAllRanges();
        reactFromToolbar(exact, kind);
      });
    });
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
    toolbar.querySelector('.annotation-tb-copy').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      if (!range) return;
      copyText(range.toString()).then(function () { showToast('已复制'); });
      hideToolbar();
    });
    toolbar.querySelector('.annotation-tb-search').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      if (!range) return;
      var q = range.toString().replace(/\s+/g, ' ').trim().slice(0, 200);
      window.open('https://www.google.com/search?q=' + encodeURIComponent(q), '_blank', 'noopener');
      hideToolbar();
    });
    toolbar.querySelector('.annotation-tb-share').addEventListener('click', function (e) {
      e.stopPropagation();
      var range = currentRange();
      var offsets = range && rangeToOffsets(range);
      if (!offsets) return;
      // inside an underlined passage -> share that passage (its thread link)
      var p = passageContaining(offsets);
      sharePassage(e.currentTarget, p ? p.exact : selectorFromOffsets(offsets).exact, !!(p && p.list.length));
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
        if (!viewer && query !== VIEWER_QUERY) fetchViewer(); // an earlier attempt failed; the token works now
        return data.data;
      });
    });
  }

  // --------------------------------------------------------------- GitHub

  var ADD_COMMENT = 'mutation($body: String!, $discussionId: ID!, $replyToId: ID) {' +
    ' addDiscussionComment(input: {body: $body, discussionId: $discussionId, replyToId: $replyToId}) { comment {' +
    ' id url createdAt bodyHTML authorAssociation author { login avatarUrl url } replies { totalCount } } } }';

  // ------------------------------------------------ likes / votes (reactions)

  var REACTION_FIELDS = 'reactionGroups { content viewerHasReacted reactors { totalCount } }';
  var VIEWER_REACTIONS = 'query($id: ID!) { node(id: $id) { ... on Discussion { ' + REACTION_FIELDS +
    ' comments(first: 100) { nodes { id ' + REACTION_FIELDS + ' replies(first: 100) { nodes { id ' + REACTION_FIELDS + ' } } } } } } }';
  var ADD_REACTION = 'mutation($id: ID!, $content: ReactionContent!) { addReaction(input: {subjectId: $id, content: $content}) { reaction { id } } }';
  var REMOVE_REACTION = 'mutation($id: ID!, $content: ReactionContent!) { removeReaction(input: {subjectId: $id, content: $content}) { reaction { id } } }';
  var CONTENT = { up: 'THUMBS_UP', down: 'THUMBS_DOWN' };

  // The relay serves the discussion anonymously (cached), so once the reader is
  // known fetch the same reactions with their token to learn what they voted.
  var viewerReactionsPending = false;
  function loadViewerReactions() {
    if (!discussion || !discussion.id || viewerReactionsPending) return;
    viewerReactionsPending = true;
    graphql(VIEWER_REACTIONS, { id: discussion.id }).then(function (data) {
      viewerReactionsPending = false;
      var d = data.node;
      if (!d) return;
      var byId = {};
      comments.forEach(function (c) { byId[c.id] = c; c.replies.forEach(function (r) { byId[r.id] = r; }); });
      discussion.likes = parseVotes(d.reactionGroups);
      (d.comments.nodes || []).forEach(function (n) {
        if (byId[n.id]) byId[n.id].votes = parseVotes(n.reactionGroups);
        (n.replies.nodes || []).forEach(function (r) { if (byId[r.id]) byId[r.id].votes = parseVotes(r.reactionGroups); });
      });
      comments.forEach(function (c) { updateVoteEls(c); c.replies.forEach(updateVoteEls); });
      renderLikeBar();
    }).catch(function () { viewerReactionsPending = false; });
  }

  // Toggle the reader's 👍 / 👎 on a comment (`dir` = 'up' | 'down'): optimistic,
  // one vote per person, switching sides removes the other reaction first.
  function toggleVote(rec, dir) {
    if (!getSession()) { showToast('登录 GitHub 后即可投票'); return; }
    var v = rec.votes, prev = { up: v.up, down: v.down, mine: v.mine };
    var steps = [];
    if (v.mine === dir) { v[dir] = Math.max(0, v[dir] - 1); v.mine = null; steps.push([REMOVE_REACTION, dir]); }
    else {
      if (v.mine) { v[v.mine] = Math.max(0, v[v.mine] - 1); steps.push([REMOVE_REACTION, v.mine]); }
      v[dir] += 1; v.mine = dir; steps.push([ADD_REACTION, dir]);
    }
    updateVoteEls(rec);
    steps.reduce(function (p, s) {
      return p.then(function () { return graphql(s[0], { id: rec.id, content: CONTENT[s[1]] }); });
    }, Promise.resolve()).catch(function (err) {
      rec.votes = prev;
      updateVoteEls(rec);
      showToast('投票失败：' + err.message);
    });
  }

  function voteHtml(rec) {
    var v = rec.votes || parseVotes(null), score = v.up - v.down;
    return '<span class="ap-vote' + (v.mine ? ' is-' + v.mine : '') + '" title="' + v.up + ' 赞同 · ' + v.down + ' 反对">' +
      '<button type="button" class="ap-vote-up" title="赞同"><i class="fa fa-caret-up"></i></button>' +
      '<b class="ap-vote-score' + (score < 0 ? ' is-negative' : '') + '">' + score + '</b>' +
      '<button type="button" class="ap-vote-down" title="反对"><i class="fa fa-caret-down"></i></button></span>';
  }

  // Every rendering of a comment (panel + bottom section) shows the same votes.
  function updateVoteEls(rec) {
    if (commentsHost) renderHotPassages(); // ranking follows the votes
    var els = document.querySelectorAll('.ap-comment[data-comment-id="' + rec.id + '"] .ap-vote');
    for (var i = 0; i < els.length; i++) {
      var span = document.createElement('span');
      span.innerHTML = voteHtml(rec);
      bindVote(span.firstChild, rec);
      els[i].parentNode.replaceChild(span.firstChild, els[i]);
    }
  }

  function bindVote(voteEl, rec) {
    voteEl.querySelector('.ap-vote-up').addEventListener('click', function () { toggleVote(rec, 'up'); });
    voteEl.querySelector('.ap-vote-down').addEventListener('click', function () { toggleVote(rec, 'down'); });
  }

  // ------------------------------------------------ passage 赞 / 存疑 (anonymous)
  // Same trust model as page views: the worker keeps `passage_reactions(path,
  // hash, quote, up, doubt)` in D1, one row per passage; the browser remembers
  // its own choices in localStorage and sends toggles. No GitHub login — this
  // is "raising a hand", commenting is "speaking". A local preview only posts
  // when `localStorage.annotationsApi` points it at a dev worker.
  var reactLocalOnly = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) && !localStorage.getItem('annotationsApi');

  function reactKey(hash, kind) { return 'react:' + cfg.path + ':' + hash + ':' + kind; }
  function myReaction(hash, kind) { try { return localStorage.getItem(reactKey(hash, kind)) === '1'; } catch (e) { return false; } }
  function rememberReaction(hash, kind, on) { try { if (on) localStorage.setItem(reactKey(hash, kind), '1'); else localStorage.removeItem(reactKey(hash, kind)); } catch (e) { /* ignore */ } }

  function loadReactions() {
    return api('/reactions?path=' + encodeURIComponent(cfg.path)).then(function (data) {
      reactions = {};
      (data.items || []).forEach(function (it) {
        if (it && /^[0-9a-f]{8}$/.test(it.hash)) reactions[it.hash] = { hash: it.hash, quote: it.quote || '', up: it.up || 0, doubt: it.doubt || 0, share: it.share || 0, range: null, marks: [] };
      });
      if (loaded) { applyHighlights(); if (commentsHost) renderHotPassages(); }
    }).catch(function (err) { console.warn('[annotations] reactions unavailable:', err.message); });
  }

  // Toggle my 赞 / 存疑 on the passage with this exact text. Optimistic; the
  // underline / marker / open panel / ranking follow the new counts.
  function react(exact, kind) {
    var hash = annotHash(exact);
    var r = reactions[hash] || (reactions[hash] = { hash: hash, quote: exact, up: 0, doubt: 0, share: 0, range: null, marks: [] });
    var on = !myReaction(hash, kind), before = r[kind];
    r[kind] = Math.max(0, r[kind] + (on ? 1 : -1));
    rememberReaction(hash, kind, on);
    refreshReactionViews(hash);
    if (reactLocalOnly) return Promise.resolve({ on: on, r: r });
    return api('/reactions', { method: 'POST', body: { path: cfg.path, hash: hash, quote: exact, kind: kind, on: on } })
      .then(function (d) { r.up = d.up || 0; r.doubt = d.doubt || 0; r.share = d.share || 0; refreshReactionViews(hash); return { on: on, r: r }; })
      .catch(function (err) {
        r[kind] = before; rememberReaction(hash, kind, !on); refreshReactionViews(hash);
        showToast('操作失败：' + err.message);
        return null;
      });
  }

  // 分享 a passage: the same popover as the article's 「分享」 (js/share.js,
  // window.BlogShare) with the passage link and the quote as text. Every completed
  // share is one more `share` on the passage (worker also bumps the article's
  // share count). No toggle, no memory — a share is a share.
  function sharePassage(btn, exact, hasThread) {
    var sel = { exact: exact };
    var url = hasThread ? threadLink(sel) : shareLink(sel);
    var quote = exact.length > 120 ? exact.slice(0, 118) + '…' : exact;
    var bar = document.querySelector('.post-actions');
    var counted = function () { countPassageShare(exact); };
    if (!window.BlogShare) {
      copyText(url).then(function () { showToast('分享链接已复制：打开后会自动定位这段文字'); counted(); });
      return;
    }
    window.BlogShare.open(btn, {
      url: url,
      title: (bar && bar.getAttribute('data-title')) || document.title,
      text: '「' + quote + '」',
      onShared: counted,
      toast: showToast
    });
  }
  function countPassageShare(exact) {
    var hash = annotHash(exact);
    var r = reactions[hash] || (reactions[hash] = { hash: hash, quote: exact, up: 0, doubt: 0, share: 0, range: null, marks: [] });
    r.share = (r.share || 0) + 1;
    refreshReactionViews(hash);
    if (reactLocalOnly) return;
    api('/reactions', { method: 'POST', body: { path: cfg.path, hash: hash, quote: exact, kind: 'share' } })
      .then(function (d) { r.up = d.up || 0; r.doubt = d.doubt || 0; r.share = d.share || 0; refreshReactionViews(hash); if (typeof d.shares === 'number') document.dispatchEvent(new CustomEvent('blog:stats', { detail: { shares: d.shares } })); })
      .catch(function () { /* keep the optimistic number */ });
  }

  // Cheap path when the passage is already underlined (repaint its marker and the
  // panel's reaction row); otherwise re-anchor so the underline appears / goes.
  function refreshReactionViews(hash) {
    var p = passageFor(['r:' + hash]), r = reactions[hash];
    var alive = r && (r.up > 0 || r.doubt > 0);
    if (!p && !alive) return; // e.g. a share of a passage nobody underlined: nothing to paint
    if (!p || (!alive && !p.list.length)) { applyHighlights(); if (commentsHost) renderHotPassages(); return; }
    p.marks.forEach(function (m) { m.classList.toggle('has-doubt', !!(r && r.doubt > 0)); });
    var marker = container.querySelector('.annotation-marker[data-hash="' + hash + '"]');
    if (marker) { marker.innerHTML = markerHtml(p); marker.title = markerTitle(p); marker.classList.toggle('has-doubt', !!(r && r.doubt > 0)); }
    if (panelState && panelState.kind === 'thread' && passageFor(panelState.ids) && passageFor(panelState.ids).hash === hash) {
      var host = panel.querySelector('.ap-react');
      if (host) { host.innerHTML = reactBarHtml(p); bindReactBar(host, p); }
      var count = panel.querySelector('.ap-count');
      if (count) count.textContent = commentCount(p) ? commentCount(p) + ' 条评论' : '还没有评论';
    }
    if (commentsHost) renderHotPassages();
  }

  // From the selection toolbar: 赞 just underlines and confirms; 存疑 also opens
  // the passage panel, whose 「说说哪里不对 →」 leads into the editor.
  function reactFromToolbar(exact, kind) {
    var hash = annotHash(exact);
    var wasOn = myReaction(hash, kind);
    react(exact, kind);
    var p = passageFor(['r:' + hash]);
    if (!p || !p.marks.length) { showToast(wasOn ? '已取消' : (kind === 'up' ? '已赞' : '已标记存疑')); return; }
    flashMarks(p.marks);
    if (kind === 'doubt' && !wasOn) openThread(p.ids, p.marks[p.marks.length - 1]);
    else showToast(wasOn ? (kind === 'up' ? '已取消赞' : '已取消存疑') : '已赞这段话');
  }

  // ----------------------------------------------------------- page views

  // Count once per browser per post per day; local previews only read the number.
  function loadViews() {
    var key = 'viewed:' + cfg.path, now = Date.now(), last = 0;
    try { last = +localStorage.getItem(key) || 0; } catch (e) { /* ignore */ }
    var local = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);
    var count = !local && now - last > 86400000;
    var req = count ? api('/views', { method: 'POST', body: { path: cfg.path } }) : api('/views?path=' + encodeURIComponent(cfg.path));
    req.then(function (data) {
      if (count) { try { localStorage.setItem(key, String(now)); } catch (e) { /* ignore */ } }
      pageViews = typeof data.views === 'number' ? data.views : null;
      renderLikeBar();
    }).catch(function () { pageViews = null; });
  }

  function ensureDiscussion() {
    if (discussion && discussion.id) return Promise.resolve(discussion.id);
    return loadDiscussion(true).then(function () {
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
        discussion = { id: data.id, url: '', totalCommentCount: 0, likes: parseVotes(null) };
        return data.id;
      });
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
      if (el.tagName === 'A') {
        var samePage = null;
        try { var u = new URL(el.getAttribute('href') || '', location.href); if (u.pathname === cfg.path && u.hash) samePage = u.hash; } catch (e) { /* ignore */ }
        if (samePage) el.setAttribute('href', samePage); // `§ 原文位置` → handled by hashchange
        else { el.setAttribute('target', '_blank'); el.setAttribute('rel', 'noopener noreferrer nofollow'); }
      }
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
    box.innerHTML = '<i class="fa fa-unlink"></i> ' + orphans.length + ' 条划线评论未能定位到原文（原文可能已修改）：';
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
    reload: function () { return loadDiscussion(true); },
    anchor: anchor,
    buildIndex: buildIndex,
    annotHash: annotHash,
    threadLink: threadLink,
    shareLink: shareLink,
    buildCommentBody: buildCommentBody,
    parseComment: parseComment,
    parseVotes: parseVotes,
    openThread: openThread,
    closePanel: closePanel,
    logout: logout,
    list: function () { return annotations; },
    comments: function () { return comments; },
    viewer: function () { return viewer; },
    // Auth / API plumbing other scripts may reuse.
    core: { api: api, graphql: graphql, getSession: getSession, login: login, ensureToken: ensureToken }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
