/**
 * feedback-brief.js — one post's reader feedback, aggregated into a 修订简报
 * (Markdown for the author and for the AI that revises the post).
 *
 * Shared by the dashboard (js/dashboard.js, in the browser) and the weekly
 * GitHub Action (tools/feedback-queue.cjs, in Node), so the two never drift.
 * It has no I/O of its own: callers fetch the four inputs and hand over a
 * tiny DOM adapter (`dom.parse(html) -> Document`; DOMParser in the browser,
 * linkedom in Node) because comment bodies and the article arrive as HTML.
 *
 *   analyze({ path, fb, disc, issues, article }) -> analysis
 *   render(analysis, { site, date })              -> markdown
 *   buildBrief(opts)                              -> { markdown, analysis }
 *
 * Inputs:
 *   fb       worker GET /feedback?path=  { reactions: [{hash, quote, section, up, doubt, share, reasons}], views, up, shares }
 *            rows whose quote starts with `§ ` are section-level 有用 / 没看懂 (quote = `§ ` + heading text, h2–h6)
 *   disc     the post's Discussion { url, comments: [{ bodyHTML, author, url, createdAt, deletedAt, authorAssociation,
 *            reactions | reactionGroups, replies: [...] | { nodes } }] } (worker relay / giscus shape or GraphQL shape)
 *   issues   the repo's 划线评论 issues mentioning the post (REST shape: number, state, title, html_url, body)
 *   article  articleFromHtml(dom, html) — the live page: normalised text + heading offsets, or null
 *
 * `analysis.score` is what the queue thresholds on: 2×存疑 + 评论 + ▲ + 3×建议修改
 * over unresolved, still-anchored passages, + unresolved plain comments, +
 * chapter-level 没看懂.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FeedbackBrief = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var REASON_LABELS = { wrong: '有错误', unclear: '没看懂', outdated: '版本过时', example: '缺例子/图', conflict: '与前文矛盾' };
  var RESOLVED_RE = /已修正|已修复|已更正|已改正|已订正|已采纳/;
  var CHAPTER_PREFIX = '§ ';

  function norm(s) { return String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); }
  function short(s, n) { s = s || ''; return s.length > n ? s.slice(0, n) + '…' : s; }
  function reasonsText(reasons) {
    return Object.keys(reasons || {}).filter(function (k) { return reasons[k] > 0; }).sort(function (a, b) { return reasons[b] - reasons[a]; })
      .map(function (k) { return (REASON_LABELS[k] || k) + ' ' + reasons[k]; }).join(' · ');
  }
  function fragment(dom, html) {
    var doc = dom.parse('<div>' + (html || '') + '</div>');
    return doc.body.firstElementChild;
  }
  function textOf(dom, html) { return norm(fragment(dom, html).textContent); }

  // A comment of the discussion -> { quote, section, issueNo, note, suggest, author, votes, url, replies, resolvedByAuthor }
  function parseNote(dom, c) {
    var root = fragment(dom, c.bodyHTML);
    var out = { quote: '', section: '', issueNo: 0, suggest: false, author: c.author ? c.author.login : 'ghost', url: c.url, createdAt: c.createdAt,
      votes: 0, hooray: 0, replies: (Array.isArray(c.replies) ? c.replies : (c.replies && c.replies.nodes) || []) };
    var rx = c.reactions || {};
    if (Array.isArray(c.reactionGroups)) c.reactionGroups.forEach(function (g) { rx[g.content] = { count: g.reactors ? g.reactors.totalCount : 0 }; });
    out.votes = ((rx.THUMBS_UP || {}).count || 0) - ((rx.THUMBS_DOWN || {}).count || 0);
    out.hooray = (rx.HOORAY || {}).count || 0;
    var first = root.firstElementChild, issueLink;
    if (first && first.tagName === 'BLOCKQUOTE' && first.querySelector('a[href*="#annot-"]')) {
      var sub = first.querySelector('a[href*="#annot-"]');
      while (sub.parentNode !== first) sub = sub.parentNode;
      issueLink = sub.querySelector('a[href*="/issues/"]');
      var sec = /位于「(.+?)」/.exec(sub.textContent || '');
      out.section = sec ? sec[1] : '';
      sub.parentNode.removeChild(sub);
      out.quote = norm(first.textContent);
      root.removeChild(first);
      out.suggest = /^\s*建议改为/.test(root.textContent || '');
    } else if (first && first.tagName === 'P' && first.querySelector('sub a[href*="/issues/"]')) {
      issueLink = first.querySelector('a[href*="/issues/"]'); root.removeChild(first);
    }
    if (issueLink) { var m = /\/issues\/(\d+)/.exec(issueLink.getAttribute('href')); out.issueNo = m ? +m[1] : 0; }
    out.note = norm(root.textContent);
    out.resolvedByAuthor = out.hooray > 0 || out.replies.some(function (r) { return r.authorAssociation === 'OWNER' && RESOLVED_RE.test(textOf(dom, r.bodyHTML)); });
    return out;
  }

  // The article's normalised text and its h2/h3 offsets, from the page HTML.
  function articleFromHtml(dom, html, fallbackTitle) {
    var doc = dom.parse(html), box = doc.querySelector('.post-container');
    if (!box) return null;
    var junk = box.querySelectorAll('.comment, .footnotes, script, style, .series-toc, .series-nav, .series-context, .pager, .related-posts, .post-actions, .post-license');
    for (var i = 0; i < junk.length; i++) junk[i].parentNode.removeChild(junk[i]);
    var body = String(box.textContent || '').replace(/\s+/g, ' ');
    var heads = [], pos = 0, hs = box.querySelectorAll('h2, h3, h4, h5, h6');
    for (var j = 0; j < hs.length; j++) {
      var t = norm(hs[j].textContent); if (!t) continue;
      var at = body.indexOf(t, pos); if (at < 0) continue;
      heads.push({ at: at, title: t, level: +hs[j].tagName.charAt(1) }); pos = at + t.length;
    }
    var h1 = doc.querySelector('.page-header .title, h1');
    return { text: body, heads: heads, title: norm(h1 ? h1.textContent : '') || fallbackTitle || '' };
  }
  function sectionAt(article, quote) {
    if (!article) return { found: null, section: '' };
    var at = article.text.indexOf(quote);
    if (at < 0) return { found: false, section: '' };
    var sec = '';
    article.heads.forEach(function (hd) { if (hd.at <= at) sec = hd.title; });
    return { found: true, section: sec };
  }

  function analyze(o) {
    var dom = o.dom, path = o.path, fb = o.fb || {}, disc = o.disc, issues = o.issues || [], article = o.article;
    var issueByNo = {}; issues.forEach(function (is) { issueByNo[is.number] = is; });
    function issueResolved(n) { return n.issueNo ? (issueByNo[n.issueNo] || {}).state === 'closed' : n.resolvedByAuthor; }
    var comments = ((disc && disc.comments) || []).filter(function (c) { return !c.deletedAt; }).map(function (c) { return parseNote(dom, c); });
    var passages = {}, order = [], chapters = [];
    function passage(quote) { if (!passages[quote]) { passages[quote] = { quote: quote, section: '', up: 0, doubt: 0, share: 0, reasons: {}, notes: [] }; order.push(passages[quote]); } return passages[quote]; }
    (fb.reactions || []).forEach(function (r) {
      if (r.quote && r.quote.indexOf(CHAPTER_PREFIX) === 0) { chapters.push({ title: r.quote.slice(CHAPTER_PREFIX.length), up: r.up || 0, doubt: r.doubt || 0 }); return; }
      var p = passage(r.quote); p.up = r.up || 0; p.doubt = r.doubt || 0; p.share = r.share || 0; p.reasons = r.reasons || {}; p.hash = r.hash; if (r.section) p.section = r.section;
    });
    var plain = [];
    comments.forEach(function (n) {
      n.resolved = issueResolved(n);
      if (!n.quote) { plain.push(n); return; }
      var p = passage(n.quote); p.notes.push(n); if (n.section && !p.section) p.section = n.section;
    });
    order.forEach(function (p) {
      var loc = sectionAt(article, p.quote);
      p.found = loc.found; if (loc.section) p.section = loc.section;
      p.resolved = p.notes.length > 0 && p.notes.every(issueResolved);
      p.score = 2 * p.doubt + p.up * 0.5 + p.notes.reduce(function (s, n) { return s + 1 + Math.max(0, n.votes) + (n.suggest ? 3 : 0); }, 0);
    });
    // keep the article's order for chapters when we know it, else by 没看懂
    if (article && chapters.length) {
      var idx = {}, lvl = {}; article.heads.forEach(function (hd, i) { idx[hd.title] = i; lvl[hd.title] = hd.level; });
      chapters.forEach(function (c) { c.level = lvl[c.title] || 2; });
      chapters.sort(function (a, b) { return (idx[a.title] == null ? 1e9 : idx[a.title]) - (idx[b.title] == null ? 1e9 : idx[b.title]); });
    } else chapters.sort(function (a, b) { return b.doubt - a.doubt || b.up - a.up; });
    var todo = order.filter(function (p) { return p.found !== false && !p.resolved; }).sort(function (a, b) { return b.score - a.score; });
    var openIssues = issues.filter(function (is) { return is.state === 'open'; });
    var a = {
      path: path, title: (article && article.title) || o.title || path, discUrl: disc && disc.url || '',
      views: fb.views || 0, up: fb.up || 0, shares: fb.shares || 0,
      comments: comments, passages: order, chapters: chapters,
      todo: todo,
      done: order.filter(function (p) { return p.found !== false && p.resolved; }),
      lost: order.filter(function (p) { return p.found === false; }),
      plain: plain.sort(function (x, y) { return y.votes - x.votes; }),
      issues: issues, openIssues: openIssues,
      otherIssues: openIssues.filter(function (is) { return !comments.some(function (n) { return n.issueNo === is.number; }); }),
      articleLoaded: !!article
    };
    a.score = todo.reduce(function (s, p) { return s + p.score; }, 0) +
      plain.filter(function (n) { return !n.resolved; }).reduce(function (s, n) { return s + 1 + Math.max(0, n.votes); }, 0) +
      chapters.reduce(function (s, c) { return s + c.doubt; }, 0);
    return a;
  }

  function render(a, opt) {
    opt = opt || {};
    var site = String(opt.site || '').replace(/\/$/, ''), path = a.path, title = a.title;
    var issueByNo = {}; a.issues.forEach(function (is) { issueByNo[is.number] = is; });
    var L = [];
    function replyLines(n, indent) {
      n.replies.forEach(function (r) { L.push(indent + '- ↳ @' + (r.author ? r.author.login : 'ghost') + (r.authorAssociation === 'OWNER' ? '（作者）' : '') + '：' + short(r.bodyText != null ? norm(r.bodyText) : textOf(opt.dom, r.bodyHTML), 200)); });
    }
    function noteLines(p, indent) {
      p.notes.slice().sort(function (x, y) { return y.votes - x.votes; }).forEach(function (n) {
        var is = n.issueNo ? issueByNo[n.issueNo] : null;
        var meta = ' @' + n.author + (n.votes ? '（▲' + n.votes + '）' : '') + (n.issueNo ? ' · Issue #' + n.issueNo + (is ? (is.state === 'closed' ? '（已关闭）' : '（open）') : '') : '');
        L.push(indent + '- ' + (n.suggest ? '✎ 建议修改' : '💬') + meta + '：' + short(n.note, 400) + '  ');
        L.push(indent + '  ' + n.url);
        replyLines(n, indent + '  ');
      });
    }
    function passageBlock(p, i) {
      var sig = [];
      if (p.doubt) sig.push('存疑 ' + p.doubt + (reasonsText(p.reasons) ? '（' + reasonsText(p.reasons) + '）' : ''));
      if (p.up) sig.push('赞 ' + p.up);
      if (p.share) sig.push('分享 ' + p.share);
      if (p.notes.length) sig.push(p.notes.length + ' 条评论');
      L.push((i + 1) + '. ' + (p.section ? '【' + p.section + '】' : '') + '「' + p.quote + '」' + (p.hash ? '  \n   ' + site + path + '#annot-' + p.hash : ''));
      if (sig.length) L.push('   - ' + sig.join(' · '));
      noteLines(p, '   ');
    }
    L.push('# 修订简报：《' + title + '》');
    L.push('');
    L.push('生成于 ' + (opt.date || new Date().toISOString().slice(0, 10)) + ' · 阅读 ' + a.views + ' · 有用 ' + a.up + ' · 分享 ' + a.shares + ' · 评论 ' + a.comments.length + ' · Issue ' + a.openIssues.length + ' 开 / ' + (a.issues.length - a.openIssues.length) + ' 关');
    L.push('');
    L.push('文章：' + site + path + (a.discUrl ? '  \n讨论：' + a.discUrl : ''));
    L.push('');
    if (a.chapters.length) {
      L.push('## 章节热度');
      L.push('');
      L.push('_读者在各级标题旁点的「有用」/「没看懂」（缩进的是小节）。没看懂多的章节整体需要补解释或例子，有用多的保持现状。_');
      L.push('');
      L.push('| 章节 | 有用 | 没看懂 |');
      L.push('| --- | ---: | ---: |');
      a.chapters.forEach(function (c) { L.push('| ' + (c.level > 2 ? '　'.repeat(c.level - 2) + '└ ' : '') + c.title.replace(/\|/g, '\\|') + ' | ' + (c.up || '') + ' | ' + (c.doubt || '') + ' |'); });
      L.push('');
    }
    L.push('## 待处理段落（按关注度排序）');
    L.push('');
    if (!a.todo.length) L.push('_没有待处理的段落。_');
    a.todo.forEach(passageBlock);
    L.push('');
    if (a.plain.length) {
      L.push('## 普通评论');
      L.push('');
      a.plain.forEach(function (n) {
        var is = n.issueNo ? issueByNo[n.issueNo] : null;
        L.push('- ' + (n.resolved ? '✅ ' : '') + '@' + n.author + (n.votes ? '（▲' + n.votes + '）' : '') + (n.issueNo ? ' · Issue #' + n.issueNo + (is && is.state === 'closed' ? '（已关闭）' : '') : '') + '：' + short(n.note, 400) + '  \n  ' + n.url);
        replyLines(n, '  ');
      });
      L.push('');
    }
    if (a.otherIssues.length) {
      L.push('## 其他 open Issue');
      L.push('');
      a.otherIssues.forEach(function (is) { L.push('- #' + is.number + ' ' + is.title + '  \n  ' + is.html_url); });
      L.push('');
    }
    if (a.lost.length) {
      L.push('## 未定位的划线（原文已改，很可能已处理）');
      L.push('');
      L.push('_这些引文在现在的正文里找不到了。确认已修好的，去 Discussion 回复「已修正」（有 Issue 的关掉 Issue），文章里会显示为绿色「已修正」。_');
      L.push('');
      a.lost.forEach(passageBlock);
      L.push('');
    }
    if (a.done.length) {
      L.push('## 已修正');
      L.push('');
      a.done.forEach(function (p) { L.push('- ' + (p.section ? '【' + p.section + '】' : '') + '「' + short(p.quote, 80) + '」 · ' + p.notes.length + ' 条评论'); });
      L.push('');
    }
    L.push('## 给 AI 的修订指令');
    L.push('');
    L.push('以上是读者对《' + title + '》（`_posts/` 中 permalink 为 `' + path + '` 的文章）的反馈。请：');
    L.push('');
    L.push('1. 按「待处理段落」的顺序逐条核对：先判断读者说得对不对，对的改正文，不对的在讨论里回复说明理由；「建议修改」条目若采纳可直接应用其「建议改为」。');
    L.push('2. 「没看懂」多的段落补解释或例子/图；「版本过时」的核对版本并按更新说明规则处理；「与前文矛盾」的检查两处是否需要一起改。「章节热度」里没看懂明显多的章，从整章的铺垫和例子入手，而不是只改一句。');
    L.push('3. 改动只针对反馈涉及的段落，保持文章结构和口吻；不要删除或改写没有反馈的部分。');
    L.push('4. 修完后列出：每条反馈 → 做了什么 / 为什么不改；提交时在 commit message 里写 `Fixes #N` 关闭对应 Issue' + (opt.queueIssue ? '（包括本 Issue #' + opt.queueIssue + '）' : '') + '。');
    return L.join('\n');
  }

  function buildBrief(o) {
    var a = analyze(o);
    return { analysis: a, markdown: render(a, { site: o.site, date: o.date, dom: o.dom, queueIssue: o.queueIssue }) };
  }

  return {
    REASON_LABELS: REASON_LABELS, RESOLVED_RE: RESOLVED_RE, CHAPTER_PREFIX: CHAPTER_PREFIX,
    reasonsText: reasonsText, short: short, parseNote: parseNote, articleFromHtml: articleFromHtml, sectionAt: sectionAt,
    analyze: analyze, render: render, buildBrief: buildBrief
  };
});
