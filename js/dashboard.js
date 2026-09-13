/**
 * dashboard.js — the author's /admin/stats.html, all read-only:
 *   - 阅读趋势: worker /views/daily (per-day totals + the posts read most in the window)
 *   - 文章榜: worker /stats/top (views · 有用 · 分享 per post) + comment counts
 *     from /stats?paths= in chunks; sortable by clicking a header
 *   - 读者划出来的句子: worker /reactions/top?kind=doubt|up (passage 存疑 / 赞)
 *   - 修订简报 (#brief=/slug.html): one post's 存疑 + 原因 + 章节 (worker
 *     /feedback), its Discussion (worker /discussions relay), its 划线评论 issues
 *     (REST, state) and whether each quote still anchors in the live page —
 *     as Markdown to copy into an AI session. Single entry: this page.
 *   - recent comments: GitHub GraphQL with the giscus login (same session key
 *     as annotations.js); open issues: GitHub REST, public.
 */
(function () {
  'use strict';
  var root = document.querySelector('.dash');
  if (!root) return;
  var api = (localStorage.getItem('annotationsApi') || root.getAttribute('data-api') || '').replace(/\/$/, '');
  var repo = root.getAttribute('data-repo');
  var categoryId = root.getAttribute('data-category-id');
  var siteUrl = root.getAttribute('data-site-url');
  var titles = window.DASH_TITLES || {};
  var SESSION_KEY = 'giscus-session';

  function h(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function ago(iso) {
    var d = (Date.now() - new Date(iso).getTime()) / 1000;
    if (d < 3600) return Math.max(1, Math.floor(d / 60)) + ' 分钟前';
    if (d < 86400) return Math.floor(d / 3600) + ' 小时前';
    if (d < 86400 * 30) return Math.floor(d / 86400) + ' 天前';
    return iso.slice(0, 10);
  }
  function titleOf(path) { return titles[path] || path; }

  function getJson(url) { return fetch(url).then(function (r) { return r.json().then(function (d) { if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status)); return d; }); }); }
  function activate(links, el) { Array.prototype.forEach.call(links, function (x) { x.classList.remove('is-active'); }); el.classList.add('is-active'); }
  function fmt(n) { return n >= 10000 ? (n / 10000).toFixed(1).replace(/\.0$/, '') + ' 万' : String(n); }

  // ---- 阅读趋势 (views_daily)
  function loadTrend(days) {
    var bars = document.querySelector('#dash-trend .dash-bars'), top = document.querySelector('#dash-trend .dash-trend-top'), sum = document.querySelector('#dash-trend .dash-trend-sum');
    getJson(api + '/views/daily?days=' + days).then(function (data) {
      var byDay = {};
      (data.days || []).forEach(function (d) { byDay[d.day] = d.views; });
      // one bar per calendar day, zero-filled, oldest first (Beijing dates)
      var list = [], now = Date.now() + 8 * 3600e3;
      for (var i = days - 1; i >= 0; i--) { var day = new Date(now - i * 86400e3).toISOString().slice(0, 10); list.push({ day: day, views: byDay[day] || 0 }); }
      var max = Math.max.apply(null, list.map(function (d) { return d.views; }).concat([1]));
      var total = list.reduce(function (n, d) { return n + d.views; }, 0);
      sum.textContent = '这 ' + days + ' 天合计 ' + fmt(total) + ' 次，日均 ' + Math.round(total / days) + '。';
      bars.innerHTML = list.map(function (d) {
        var hgt = Math.max(d.views ? 2 : 0, Math.round(d.views / max * 100));
        return '<span class="dash-bar" title="' + d.day + '：' + d.views + ' 次"><i style="height:' + hgt + '%"></i></span>';
      }).join('') + '<span class="dash-bar-axis"><small>' + list[0].day + '</small><small>峰值 ' + max + '</small><small>' + list[list.length - 1].day + '</small></span>';
      var paths = (data.paths || []).slice(0, 8);
      top.innerHTML = paths.length ? '<h4>这段时间读得最多</h4><ol class="dash-top">' + paths.map(function (p) {
        return '<li><a href="' + h(p.path) + '">' + h(titleOf(p.path)) + '</a> <span class="dash-muted">' + fmt(p.views) + '</span></li>';
      }).join('') + '</ol>' : '';
    }).catch(function (err) { bars.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  var trendLinks = document.querySelectorAll('#dash-trend [data-days]');
  Array.prototype.forEach.call(trendLinks, function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); activate(trendLinks, a); loadTrend(parseInt(a.getAttribute('data-days'), 10)); });
  });
  loadTrend(30);

  // ---- 文章榜 (/stats/top + comment counts from /stats)
  var postRows = [], postSort = 'views', POSTS_TOP = 10, postsExpanded = false;
  function renderPosts() {
    var tbody = document.querySelector('#dash-posts tbody');
    if (!postRows.length) { tbody.innerHTML = '<tr><td colspan="8" class="dash-muted">还没有数据</td></tr>'; return; }
    var all = postRows.slice().sort(function (a, b) {
      if (postSort === 'recent') return (b.updated_at || '') < (a.updated_at || '') ? -1 : 1;
      if (postSort === 'rate') return b.rate - a.rate;
      return (b[postSort] || 0) - (a[postSort] || 0);
    });
    // the long tail (one view each) says nothing — TOP 20 unless expanded
    var rows = postsExpanded ? all : all.slice(0, POSTS_TOP);
    var t = { views: 0, up: 0, shares: 0, comments: 0 };
    all.forEach(function (r) { t.views += r.views; t.up += r.up; t.shares += r.shares; t.comments += r.comments || 0; });
    tbody.innerHTML = rows.map(function (r, i) {
      return '<tr><td class="dash-muted">' + (i + 1) + '</td><td><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a> <a class="dash-brief-link" href="#brief=' + h(r.path) + '" title="这篇文章的修订简报">简报</a></td>' +
        '<td class="num">' + fmt(r.views) + '</td><td class="num">' + (r.up || '') + '</td>' +
        '<td class="num dash-muted">' + (r.views >= 20 && r.up ? (r.rate * 100).toFixed(1) + '%' : '') + '</td>' +
        '<td class="num">' + (r.shares || '') + '</td><td class="num">' + (r.comments == null ? '<span class="dash-muted">…</span>' : (r.comments || '')) + '</td>' +
        '<td class="dash-muted">' + (r.updated_at ? ago(r.updated_at) : '') + '</td></tr>';
    }).join('') +
      (all.length > POSTS_TOP ? '<tr class="dash-expand"><td></td><td colspan="7"><a href="#" class="dash-toggle">' + (postsExpanded ? '只看 TOP ' + POSTS_TOP : '展开全部 ' + all.length + ' 篇') + '</a></td></tr>' : '') +
      '<tr class="dash-total"><td></td><td class="dash-muted">全部 ' + all.length + ' 篇合计</td><td class="num">' + fmt(t.views) + '</td><td class="num">' + t.up + '</td><td></td><td class="num">' + t.shares + '</td><td class="num">' + t.comments + '</td><td></td></tr>';
    var toggle = tbody.querySelector('.dash-toggle');
    if (toggle) toggle.addEventListener('click', function (e) { e.preventDefault(); postsExpanded = !postsExpanded; renderPosts(); });
  }
  function loadPosts() {
    var tbody = document.querySelector('#dash-posts tbody');
    getJson(api + '/stats/top?limit=100').then(function (data) {
      postRows = (data.rows || []).map(function (r) { return { path: r.path, views: r.views || 0, up: r.up || 0, shares: r.shares || 0, updated_at: r.updated_at, comments: null, rate: r.views ? (r.up || 0) / r.views : 0 }; });
      renderPosts();
      // comment counts: /stats takes 20 paths a call (giscus lookups, edge-cached)
      var paths = postRows.map(function (r) { return r.path; }), chunks = [];
      while (paths.length) chunks.push(paths.splice(0, 20));
      chunks.forEach(function (chunk) {
        getJson(api + '/stats?paths=' + encodeURIComponent(chunk.join(','))).then(function (d) {
          postRows.forEach(function (r) { var it = d.items && d.items[r.path]; if (it) r.comments = it.comments || 0; });
          renderPosts();
        }).catch(function () { postRows.forEach(function (r) { if (chunk.indexOf(r.path) >= 0 && r.comments == null) r.comments = 0; }); renderPosts(); });
      });
    }).catch(function (err) { tbody.innerHTML = '<tr><td colspan="8" class="dash-muted">加载失败：' + h(err.message) + '</td></tr>'; });
  }
  var sortHeads = document.querySelectorAll('#dash-posts th[data-sort]');
  Array.prototype.forEach.call(sortHeads, function (th) {
    th.addEventListener('click', function () { activate(sortHeads, th); postSort = th.getAttribute('data-sort'); renderPosts(); });
  });
  loadPosts();

  // ---- 读者划出来的句子 (/reactions/top)
  function loadPassages(kind) {
    var host = document.querySelector('#dash-passages .dash-list');
    getJson(api + '/reactions/top?kind=' + kind + '&limit=30').then(function (data) {
      var rows = data.rows || [];
      if (!rows.length) { host.innerHTML = '<p class="dash-muted">还没有人' + ({ doubt: '存疑', up: '点赞', share: '分享' }[kind] || '点赞') + '。</p>'; return; }
      host.innerHTML = '<ol class="dash-quotes">' + rows.map(function (r) {
        return '<li class="' + (kind === 'doubt' && r.doubt ? 'is-doubt' : '') + '"><a class="dash-quote" href="' + h(r.path) + '#annot-' + h(r.hash) + '">' + h(r.quote) + '</a>' +
          '<div class="dash-quote-meta"><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a>' + (r.section ? ' <span class="dash-muted">› ' + h(r.section) + '</span>' : '') + ' · ' +
          (r.doubt ? '<span class="is-doubt"><i class="fa fa-question-circle"></i> ' + r.doubt + (reasonsText(r.reasons) ? '（' + h(reasonsText(r.reasons)) + '）' : '') + '</span> ' : '') +
          (r.up ? '<span><i class="fa fa-thumbs-up"></i> ' + r.up + '</span> ' : '') +
          (r.share ? '<span><i class="fa fa-share-alt"></i> ' + r.share + '</span> ' : '') +
          (r.updated_at ? '<span class="dash-muted">· ' + ago(r.updated_at) + '</span>' : '') + '</div></li>';
      }).join('') + '</ol>';
    }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  var kindLinks = document.querySelectorAll('#dash-passages [data-kind]');
  Array.prototype.forEach.call(kindLinks, function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); activate(kindLinks, a); loadPassages(a.getAttribute('data-kind')); });
  });
  loadPassages('doubt');

  // ---- 修订简报 (#brief=/slug.html): everything readers said about one post, as
  // Markdown for the author — and for the AI that revises the post. Sources:
  // worker /feedback (存疑 + 原因 + 章节, D1), the post's Discussion via the
  // worker's anonymous /discussions relay, the repo's 划线评论 issues (REST,
  // state = 「已修正」), and the live page itself to tell which quotes still anchor.
  var REASON_LABELS = { wrong: '有错误', unclear: '没看懂', outdated: '版本过时', example: '缺例子/图', conflict: '与前文矛盾' };
  var RESOLVED_RE = /已修正|已修复|已更正|已改正|已订正|已采纳/;
  var briefBody = document.querySelector('#dash-brief .dash-brief-body');
  var briefPick = document.querySelector('#dash-brief .dash-brief-pick');
  (window.DASH_POSTS || []).forEach(function (p) { var o = document.createElement('option'); o.value = p[0]; o.textContent = p[1]; briefPick.appendChild(o); });
  briefPick.addEventListener('change', function () { if (briefPick.value) location.hash = 'brief=' + briefPick.value; });

  function text(html) { var d = new DOMParser().parseFromString('<div>' + html + '</div>', 'text/html'); return (d.body.textContent || '').replace(/\s+/g, ' ').trim(); }
  function reasonsText(reasons) {
    return Object.keys(reasons || {}).filter(function (k) { return reasons[k] > 0; }).sort(function (a, b) { return reasons[b] - reasons[a]; })
      .map(function (k) { return (REASON_LABELS[k] || k) + ' ' + reasons[k]; }).join(' · ');
  }
  // A comment of the discussion -> { quote, section, issueNo, note, suggest, author, votes, url, replies, resolvedByAuthor }
  function parseNote(c) {
    var doc = new DOMParser().parseFromString('<div>' + (c.bodyHTML || '') + '</div>', 'text/html'), root = doc.body.firstChild;
    var out = { quote: '', section: '', issueNo: 0, suggest: false, author: c.author ? c.author.login : 'ghost', url: c.url, createdAt: c.createdAt,
      votes: 0, hooray: 0, replies: (Array.isArray(c.replies) ? c.replies : (c.replies && c.replies.nodes) || []) };
    var rx = c.reactions || {};
    if (Array.isArray(c.reactionGroups)) c.reactionGroups.forEach(function (g) { rx[g.content] = { count: g.reactors ? g.reactors.totalCount : 0 }; });
    out.votes = ((rx.THUMBS_UP || {}).count || 0) - ((rx.THUMBS_DOWN || {}).count || 0);
    out.hooray = (rx.HOORAY || {}).count || 0;
    var first = root.firstElementChild, issueLink;
    if (first && first.tagName === 'BLOCKQUOTE' && first.querySelector('a[href*="#annot-"]')) {
      var sub = first.querySelector('a[href*="#annot-"]').closest('p, sub') || first.querySelector('a[href*="#annot-"]');
      while (sub.parentNode !== first) sub = sub.parentNode;
      issueLink = sub.querySelector('a[href*="/issues/"]');
      var sec = /位于「(.+?)」/.exec(sub.textContent || '');
      out.section = sec ? sec[1] : '';
      sub.parentNode.removeChild(sub);
      out.quote = (first.textContent || '').replace(/\s+/g, ' ').trim();
      root.removeChild(first);
      out.suggest = /^\s*建议改为/.test(root.textContent || '');
    } else if (first && first.tagName === 'P' && first.querySelector('sub a[href*="/issues/"]')) {
      issueLink = first.querySelector('a[href*="/issues/"]'); root.removeChild(first);
    }
    if (issueLink) { var m = /\/issues\/(\d+)/.exec(issueLink.getAttribute('href')); out.issueNo = m ? +m[1] : 0; }
    out.note = (root.textContent || '').replace(/\s+/g, ' ').trim();
    out.resolvedByAuthor = out.hooray > 0 || out.replies.some(function (r) { return r.authorAssociation === 'OWNER' && RESOLVED_RE.test(text(r.bodyHTML || '')); });
    return out;
  }
  // The article's normalised text and its h2/h3 offsets, from the live page.
  function loadArticle(path) {
    return fetch(path).then(function (r) { return r.text(); }).then(function (html) {
      var doc = new DOMParser().parseFromString(html, 'text/html'), box = doc.querySelector('.post-container');
      if (!box) return null;
      Array.prototype.forEach.call(box.querySelectorAll('.comment, .footnotes, script, style, .series-toc, .pager, .related-posts'), function (n) { n.parentNode.removeChild(n); });
      var body = (box.textContent || '').replace(/\s+/g, ' ');
      var heads = [], pos = 0;
      Array.prototype.forEach.call(box.querySelectorAll('h2, h3'), function (hd) {
        var t = (hd.textContent || '').replace(/\s+/g, ' ').trim(); if (!t) return;
        var at = body.indexOf(t, pos); if (at < 0) return;
        heads.push({ at: at, title: t }); pos = at + t.length;
      });
      return { text: body, heads: heads, title: (doc.querySelector('.page-header .title, h1') || {}).textContent || titleOf(path) };
    }).catch(function () { return null; });
  }
  function sectionAt(article, quote) {
    if (!article) return { found: null, section: '' };
    var at = article.text.indexOf(quote);
    if (at < 0) return { found: false, section: '' };
    var sec = '';
    article.heads.forEach(function (hd) { if (hd.at <= at) sec = hd.title; });
    return { found: true, section: sec };
  }
  function short(s, n) { s = s || ''; return s.length > n ? s.slice(0, n) + '…' : s; }

  function buildBrief(path, fb, disc, issues, article) {
    var title = (article && article.title.trim()) || titleOf(path), site = (siteUrl || location.origin).replace(/\/$/, '');
    var issueByNo = {}; issues.forEach(function (is) { issueByNo[is.number] = is; });
    var comments = ((disc && disc.comments) || []).filter(function (c) { return !c.deletedAt; }).map(parseNote);
    var passages = {}, order = [];
    function passage(quote) { var k = quote; if (!passages[k]) { passages[k] = { quote: quote, section: '', up: 0, doubt: 0, share: 0, reasons: {}, notes: [] }; order.push(passages[k]); } return passages[k]; }
    (fb.reactions || []).forEach(function (r) { var p = passage(r.quote); p.up = r.up; p.doubt = r.doubt; p.share = r.share; p.reasons = r.reasons || {}; p.hash = r.hash; if (r.section) p.section = r.section; });
    var plain = [];
    comments.forEach(function (n) {
      if (!n.quote) { plain.push(n); return; }
      var p = passage(n.quote); p.notes.push(n); if (n.section && !p.section) p.section = n.section;
    });
    order.forEach(function (p) {
      var loc = sectionAt(article, p.quote);
      p.found = loc.found; if (loc.section) p.section = loc.section;
      p.resolved = p.notes.length > 0 && p.notes.every(function (n) { return n.issueNo ? (issueByNo[n.issueNo] || {}).state === 'closed' : n.resolvedByAuthor; });
      p.score = 2 * p.doubt + p.up * 0.5 + p.notes.reduce(function (s, n) { return s + 1 + Math.max(0, n.votes) + (n.suggest ? 3 : 0); }, 0);
    });
    var todo = order.filter(function (p) { return p.found !== false && !p.resolved; }).sort(function (a, b) { return b.score - a.score; });
    var done = order.filter(function (p) { return p.found !== false && p.resolved; });
    var lost = order.filter(function (p) { return p.found === false; });
    var openIssues = issues.filter(function (is) { return is.state === 'open'; });
    var L = [];
    L.push('# 修订简报：《' + title + '》');
    L.push('');
    L.push('生成于 ' + new Date().toISOString().slice(0, 10) + ' · 阅读 ' + (fb.views || 0) + ' · 有用 ' + (fb.up || 0) + ' · 分享 ' + (fb.shares || 0) + ' · 评论 ' + comments.length + ' · Issue ' + openIssues.length + ' 开 / ' + (issues.length - openIssues.length) + ' 关');
    L.push('');
    L.push('文章：' + site + path + (disc && disc.url ? '  \n讨论：' + disc.url : ''));
    L.push('');
    function noteLines(p, indent) {
      p.notes.sort(function (a, b) { return b.votes - a.votes; }).forEach(function (n) {
        var is = n.issueNo ? issueByNo[n.issueNo] : null;
        var tag = n.suggest ? '✎ 建议修改' : '💬';
        var meta = ' @' + n.author + (n.votes ? '（▲' + n.votes + '）' : '') + (n.issueNo ? ' · Issue #' + n.issueNo + (is ? (is.state === 'closed' ? '（已关闭）' : '（open）') : '') : '');
        L.push(indent + '- ' + tag + meta + '：' + short(n.note, 400) + '  ');
        L.push(indent + '  ' + n.url);
        n.replies.forEach(function (r) { L.push(indent + '  - ↳ @' + (r.author ? r.author.login : 'ghost') + (r.authorAssociation === 'OWNER' ? '（作者）' : '') + '：' + short(text(r.bodyHTML || ''), 200)); });
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
    L.push('## 待处理段落（按关注度排序）');
    L.push('');
    if (!todo.length) L.push('_没有待处理的段落。_');
    todo.forEach(passageBlock);
    L.push('');
    if (plain.length) {
      L.push('## 普通评论');
      L.push('');
      plain.sort(function (a, b) { return b.votes - a.votes; }).forEach(function (n) {
        var is = n.issueNo ? issueByNo[n.issueNo] : null;
        L.push('- @' + n.author + (n.votes ? '（▲' + n.votes + '）' : '') + (n.issueNo ? ' · Issue #' + n.issueNo + (is && is.state === 'closed' ? '（已关闭）' : '') : '') + '：' + short(n.note, 400) + '  \n  ' + n.url);
        n.replies.forEach(function (r) { L.push('  - ↳ @' + (r.author ? r.author.login : 'ghost') + (r.authorAssociation === 'OWNER' ? '（作者）' : '') + '：' + short(text(r.bodyHTML || ''), 200)); });
      });
      L.push('');
    }
    var otherIssues = openIssues.filter(function (is) { return !comments.some(function (n) { return n.issueNo === is.number; }); });
    if (otherIssues.length) {
      L.push('## 其他 open Issue');
      L.push('');
      otherIssues.forEach(function (is) { L.push('- #' + is.number + ' ' + is.title + '  \n  ' + is.html_url); });
      L.push('');
    }
    if (lost.length) {
      L.push('## 未定位的划线（原文已改，很可能已处理）');
      L.push('');
      L.push('_这些引文在现在的正文里找不到了。确认已修好的，去 Discussion 回复「已修正」（有 Issue 的关掉 Issue），文章里会显示为绿色「已修正」。_');
      L.push('');
      lost.forEach(passageBlock);
      L.push('');
    }
    if (done.length) {
      L.push('## 已修正');
      L.push('');
      done.forEach(function (p) { L.push('- ' + (p.section ? '【' + p.section + '】' : '') + '「' + short(p.quote, 80) + '」 · ' + p.notes.length + ' 条评论'); });
      L.push('');
    }
    L.push('## 给 AI 的修订指令');
    L.push('');
    L.push('以上是读者对《' + title + '》（`_posts/` 中 permalink 为 `' + path + '` 的文章）的反馈。请：');
    L.push('');
    L.push('1. 按「待处理段落」的顺序逐条核对：先判断读者说得对不对，对的改正文，不对的在讨论里回复说明理由；「建议修改」条目若采纳可直接应用其「建议改为」。');
    L.push('2. 「没看懂」多的段落补解释或例子/图；「版本过时」的核对版本并按更新说明规则处理；「与前文矛盾」的检查两处是否需要一起改。');
    L.push('3. 改动只针对反馈涉及的段落，保持文章结构和口吻；不要删除或改写没有反馈的部分。');
    L.push('4. 修完后列出：每条反馈 → 做了什么 / 为什么不改；提交时在 commit message 里写 `Fixes #N` 关闭对应 Issue。');
    return L.join('\n');
  }

  function openBrief(path) {
    if (!path || !/^\/[\w\-./%]+\.html$/.test(path)) return;
    briefPick.value = path;
    briefBody.innerHTML = '<p class="dash-muted">生成中…</p>';
    document.getElementById('dash-brief').scrollIntoView({ block: 'start' });
    var labels = encodeURIComponent('划线评论');
    Promise.all([
      getJson(api + '/feedback?path=' + encodeURIComponent(path)).catch(function () { return { reactions: [] }; }),
      getJson(api + '/discussions?term=' + encodeURIComponent(path) + '&t=' + Date.now()).then(function (d) { return d.discussion || null; }).catch(function () { return null; }),
      fetch('https://api.github.com/repos/' + repo + '/issues?state=all&per_page=100&labels=' + labels).then(function (r) { return r.ok ? r.json() : []; }).catch(function () { return []; }),
      loadArticle(path)
    ]).then(function (res) {
      var issues = res[2].filter(function (is) { return !is.pull_request && (is.body || '').indexOf(path) >= 0; });
      var md = buildBrief(path, res[0], res[1], issues, res[3]);
      briefBody.innerHTML = '<div class="dash-brief-bar"><button type="button" class="dash-brief-copy"><i class="fa fa-clipboard"></i> 复制 Markdown</button> ' +
        '<a href="' + h(path) + '">打开文章</a>' + (res[1] && res[1].url ? ' · <a target="_blank" rel="noopener" href="' + h(res[1].url) + '">Discussion ↗</a>' : '') +
        (!res[3] ? ' <span class="dash-muted">（读不到文章正文，无法判断引文是否仍能定位）</span>' : '') + '</div>' +
        '<pre class="dash-brief-md"></pre>';
      briefBody.querySelector('.dash-brief-md').textContent = md;
      briefBody.querySelector('.dash-brief-copy').addEventListener('click', function () {
        var btn = this;
        (navigator.clipboard ? navigator.clipboard.writeText(md) : Promise.reject()).then(function () { btn.textContent = '已复制'; setTimeout(function () { btn.innerHTML = '<i class="fa fa-clipboard"></i> 复制 Markdown'; }, 1500); })
          .catch(function () { var r = document.createRange(); r.selectNodeContents(briefBody.querySelector('.dash-brief-md')); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); });
      });
    }).catch(function (err) { briefBody.innerHTML = '<p class="dash-muted">生成失败：' + h(err.message) + '</p>'; });
  }
  function briefFromHash() { var m = /^#brief=(.+)$/.exec(location.hash || ''); if (m) openBrief(decodeURIComponent(m[1])); }
  window.addEventListener('hashchange', briefFromHash);
  briefFromHash();

  // ---- issues (public REST)
  function loadIssues() {
    var host = document.querySelector('#dash-issues .dash-list');
    var base = 'https://api.github.com/repos/' + repo + '/issues?state=open&per_page=30&labels=';
    Promise.all([fetch(base + encodeURIComponent('划线评论')), fetch(base + 'dead-links')].map(function (p) { return p.then(function (r) { return r.ok ? r.json() : []; }); }))
      .then(function (res) {
        var errata = res[0], links = res[1];
        var html = '<h4>读者报的错 <span class="dash-count">' + errata.length + '</span> <a class="dash-more" target="_blank" rel="noopener" href="https://github.com/' + repo + '/issues?q=is%3Aissue+is%3Aopen+label%3A%E5%88%92%E7%BA%BF%E8%AF%84%E8%AE%BA">全部 ↗</a></h4>';
        html += errata.length ? '<ul>' + errata.map(function (is) {
          return '<li><a target="_blank" rel="noopener" href="' + h(is.html_url) + '">#' + is.number + ' ' + h(is.title) + '</a> <span class="dash-muted">' + ago(is.created_at) + '</span></li>';
        }).join('') + '</ul>' : '<p class="dash-muted">没有待处理的勘误。</p>';
        html += '<h4>死链报告 <span class="dash-count">' + links.length + '</span></h4>';
        html += links.length ? '<ul>' + links.map(function (is) {
          return '<li><a target="_blank" rel="noopener" href="' + h(is.html_url) + '">#' + is.number + ' ' + h(is.title) + '</a> <span class="dash-muted">更新于 ' + ago(is.updated_at) + '</span></li>';
        }).join('') + '</ul>' : '<p class="dash-muted">没有。</p>';
        host.innerHTML = html;
      }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  loadIssues();

  // ---- recent comments (GraphQL, needs the giscus login)
  var authEl = document.querySelector('#dash-comments .dash-auth');
  var listEl = document.querySelector('#dash-comments .dash-list');
  function session() { try { var raw = localStorage.getItem(SESSION_KEY); return raw ? JSON.parse(raw) : ''; } catch (e) { return ''; } }
  function login() {
    var url = new URL(location.href); url.hash = ''; url.searchParams.delete('giscus');
    location.href = 'https://giscus.app/api/oauth/authorize?redirect_uri=' + encodeURIComponent(url.toString());
  }
  (function takeSession() {
    var url = new URL(location.href), s = url.searchParams.get('giscus');
    if (!s) return;
    try { localStorage.setItem(SESSION_KEY, JSON.stringify(s)); } catch (e) { /* ignore */ }
    url.searchParams.delete('giscus'); history.replaceState(null, '', url.toString());
  })();

  var QUERY = 'query($owner: String!, $name: String!, $cat: ID!) { repository(owner: $owner, name: $name) {' +
    ' discussions(first: 20, categoryId: $cat, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes {' +
    ' title url updatedAt comments { totalCount } reactionGroups { content reactors { totalCount } }' +
    ' recent: comments(last: 3) { nodes { url createdAt bodyText author { login } replies(last: 2) { nodes { url createdAt bodyText author { login } } } } } } } } }';

  function loadComments() {
    if (!session()) {
      authEl.innerHTML = '<a href="#" class="dash-login">登录 GitHub</a>';
      authEl.querySelector('.dash-login').addEventListener('click', function (e) { e.preventDefault(); login(); });
      return;
    }
    authEl.textContent = '加载中…';
    fetch(api + '/token', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: session() }) })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.token) throw new Error('登录已过期');
        var parts = repo.split('/');
        return fetch('https://api.github.com/graphql', { method: 'POST', headers: { Authorization: 'Bearer ' + d.token, 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: QUERY, variables: { owner: parts[0], name: parts[1], cat: categoryId } }) }).then(function (r) { return r.json(); });
      })
      .then(function (res) {
        if (res.errors) throw new Error(res.errors[0].message);
        var nodes = res.data.repository.discussions.nodes;
        authEl.innerHTML = '<a href="https://github.com/' + repo + '/discussions" target="_blank" rel="noopener">全部讨论 ↗</a>';
        if (!nodes.length) { listEl.innerHTML = '<p class="dash-muted">还没有评论。</p>'; return; }
        listEl.innerHTML = nodes.map(function (d) {
          var likes = (d.reactionGroups || []).filter(function (g) { return g.content === 'THUMBS_UP'; }).map(function (g) { return g.reactors.totalCount; })[0] || 0;
          var items = [];
          (d.recent.nodes || []).forEach(function (c) {
            items.push(c);
            (c.replies.nodes || []).forEach(function (r) { items.push(r); });
          });
          items.sort(function (a, b) { return a.createdAt < b.createdAt ? 1 : -1; });
          var path = d.title.indexOf('/') === 0 ? d.title : null;
          return '<div class="dash-thread">' +
            '<div class="dash-thread-head"><a href="' + h(path || d.url) + '">' + h(path ? titleOf(path) : d.title) + '</a>' +
            ' <span class="dash-muted">' + d.comments.totalCount + ' 条评论 · 👍 ' + likes + ' · ' + ago(d.updatedAt) + '</span>' +
            ' <a class="dash-more" target="_blank" rel="noopener" href="' + h(d.url) + '">GitHub ↗</a></div>' +
            items.slice(0, 3).map(function (c) {
              return '<div class="dash-comment"><b>' + h(c.author ? c.author.login : 'ghost') + '</b> <span class="dash-muted">' + ago(c.createdAt) + '</span>' +
                '<a target="_blank" rel="noopener" href="' + h(c.url) + '"> ' + h(c.bodyText.replace(/\s+/g, ' ').slice(0, 140)) + (c.bodyText.length > 140 ? '…' : '') + '</a></div>';
            }).join('') + '</div>';
        }).join('');
      })
      .catch(function (err) {
        authEl.innerHTML = '<a href="#" class="dash-login">重新登录</a>';
        authEl.querySelector('.dash-login').addEventListener('click', function (e) { e.preventDefault(); try { localStorage.removeItem(SESSION_KEY); } catch (x) { /* ignore */ } login(); });
        listEl.innerHTML = '<p class="dash-muted">' + h(err.message) + '</p>';
      });
  }
  loadComments();
})();
