/**
 * dashboard.js — the author's /admin/stats.html: view ranking (worker /views/top),
 * recent comments (GitHub GraphQL with the giscus login, same session key as
 * annotations.js), open issues (GitHub REST, public). All read-only.
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

  // ---- views
  function loadViews(order) {
    var tbody = document.querySelector('#dash-views tbody');
    fetch(api + '/views/top?limit=100&order=' + order).then(function (r) { return r.json(); }).then(function (data) {
      if (!data.rows || !data.rows.length) { tbody.innerHTML = '<tr><td colspan="4" class="dash-muted">还没有数据</td></tr>'; return; }
      var total = data.rows.reduce(function (n, r) { return n + r.views; }, 0);
      tbody.innerHTML = data.rows.map(function (r, i) {
        return '<tr><td class="dash-muted">' + (i + 1) + '</td><td><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a></td>' +
          '<td class="num">' + r.views + '</td><td class="dash-muted">' + (r.updated_at ? ago(r.updated_at) : '') + '</td></tr>';
      }).join('') + '<tr><td></td><td class="dash-muted">以上 ' + data.rows.length + ' 篇合计</td><td class="num">' + total + '</td><td></td></tr>';
    }).catch(function (err) { tbody.innerHTML = '<tr><td colspan="4" class="dash-muted">加载失败：' + h(err.message) + '</td></tr>'; });
  }
  Array.prototype.forEach.call(document.querySelectorAll('#dash-views [data-order]'), function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      Array.prototype.forEach.call(document.querySelectorAll('#dash-views [data-order]'), function (x) { x.classList.remove('is-active'); });
      a.classList.add('is-active');
      loadViews(a.getAttribute('data-order'));
    });
  });
  loadViews('top');

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
