/**
 * dashboard.js — the author's /admin/stats.html, all read-only:
 *   - 阅读趋势: worker /views/daily (per-day totals + the posts read most in the window)
 *   - 文章榜: worker /stats/top (views · 有用 · 分享 per post) + comment counts
 *     from /stats?paths= in chunks; sortable by clicking a header
 *   - 读者划出来的句子: worker /reactions/top?kind=doubt|up (passage 存疑 / 赞)
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
  var postRows = [], postSort = 'views';
  function renderPosts() {
    var tbody = document.querySelector('#dash-posts tbody');
    if (!postRows.length) { tbody.innerHTML = '<tr><td colspan="8" class="dash-muted">还没有数据</td></tr>'; return; }
    var rows = postRows.slice().sort(function (a, b) {
      if (postSort === 'recent') return (b.updated_at || '') < (a.updated_at || '') ? -1 : 1;
      if (postSort === 'rate') return b.rate - a.rate;
      return (b[postSort] || 0) - (a[postSort] || 0);
    });
    var t = { views: 0, up: 0, shares: 0, comments: 0 };
    rows.forEach(function (r) { t.views += r.views; t.up += r.up; t.shares += r.shares; t.comments += r.comments || 0; });
    tbody.innerHTML = rows.map(function (r, i) {
      return '<tr><td class="dash-muted">' + (i + 1) + '</td><td><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a></td>' +
        '<td class="num">' + fmt(r.views) + '</td><td class="num">' + (r.up || '') + '</td>' +
        '<td class="num dash-muted">' + (r.views >= 20 && r.up ? (r.rate * 100).toFixed(1) + '%' : '') + '</td>' +
        '<td class="num">' + (r.shares || '') + '</td><td class="num">' + (r.comments == null ? '<span class="dash-muted">…</span>' : (r.comments || '')) + '</td>' +
        '<td class="dash-muted">' + (r.updated_at ? ago(r.updated_at) : '') + '</td></tr>';
    }).join('') + '<tr class="dash-total"><td></td><td class="dash-muted">以上 ' + rows.length + ' 篇合计</td><td class="num">' + fmt(t.views) + '</td><td class="num">' + t.up + '</td><td></td><td class="num">' + t.shares + '</td><td class="num">' + t.comments + '</td><td></td></tr>';
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
      if (!rows.length) { host.innerHTML = '<p class="dash-muted">还没有人' + (kind === 'doubt' ? '存疑' : '点赞') + '。</p>'; return; }
      host.innerHTML = '<ol class="dash-quotes">' + rows.map(function (r) {
        return '<li class="' + (kind === 'doubt' && r.doubt ? 'is-doubt' : '') + '"><a class="dash-quote" href="' + h(r.path) + '#annot-' + h(r.hash) + '">' + h(r.quote) + '</a>' +
          '<div class="dash-quote-meta"><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a> · ' +
          (r.doubt ? '<span class="is-doubt"><i class="fa fa-question-circle"></i> ' + r.doubt + '</span> ' : '') +
          (r.up ? '<span><i class="fa fa-thumbs-up"></i> ' + r.up + '</span> ' : '') +
          (r.updated_at ? '<span class="dash-muted">· ' + ago(r.updated_at) + '</span>' : '') + '</div></li>';
      }).join('') + '</ol>';
    }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  var kindLinks = document.querySelectorAll('#dash-passages [data-kind]');
  Array.prototype.forEach.call(kindLinks, function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); activate(kindLinks, a); loadPassages(a.getAttribute('data-kind')); });
  });
  loadPassages('doubt');

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
