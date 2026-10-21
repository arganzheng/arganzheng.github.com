/**
 * dashboard.js — the author's /admin/stats.html, all read-only:
 *   - 阅读趋势: worker /views/daily (per-day totals + the posts read most in the window)
 *   - 文章榜: worker /stats/top (views · 点赞 · 分享 per post) + comment counts
 *     from /stats?paths= in chunks; sortable by clicking a header
 *   - 读者划出来的句子: worker /reactions/top?kind=up|share (passage 赞 / 分享)
 *   - 修订简报 (#brief=/slug.html): one post's reactions (worker
 *     /feedback), its Discussion (worker /discussions relay), its 划线评论 issues
 *     (REST, state) and whether each quote still anchors in the live page —
 *     as Markdown to copy into an AI session. Single entry: this page. The
 *     aggregation itself is js/feedback-brief.js, shared with the weekly
 *     tools/feedback-queue.cjs Action that files 「待修订」 issues. Above the
 *     picker, 待修订的文章 lists posts linked from open 划线评论 / 待修订
 *     issues; `/feedback` supplies views and passage metrics.
 *   - recent comments: GitHub GraphQL with the giscus login (same session key
 *     as annotations.js); open issues (待修订 · 划线评论 · dead-links): GitHub REST, public.
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
  var FB = window.FeedbackBrief; // js/feedback-brief.js, loaded before this file
  var DOM = { parse: function (html) { return new DOMParser().parseFromString(html, 'text/html'); } };

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
  function setKpi(key, value, kind) {
    var card = document.querySelector('.dash-kpi[data-kpi="' + key + '"]');
    if (!card) return;
    var number = card.querySelector('strong'), note = card.querySelector('.dash-kpi-note');
    var count = kind === 'count', failed = value == null;
    number.textContent = failed ? '—' : String(value);
    card.classList.toggle('is-ok', count && !failed && Number(value) === 0);
    card.classList.toggle('is-warn', count && !failed && Number(value) > 0);
    card.classList.toggle('is-error', failed);
    if (note) {
      note.hidden = !(count && !failed && Number(value) === 0);
      note.textContent = note.hidden ? '' : '✓';
    }
  }
  function initDashboardControls() {
    Array.prototype.forEach.call(root.querySelectorAll('.dash-help'), function (button) {
      var text = document.getElementById(button.getAttribute('aria-controls'));
      if (!text) return;
      button.addEventListener('click', function () {
        var open = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', String(!open));
        text.hidden = open;
      });
    });
    var links = root.querySelectorAll('.dash-gnav a'), groups = root.querySelectorAll('.dash-group');
    function markGroup(id) {
      Array.prototype.forEach.call(links, function (link) {
        var active = link.getAttribute('href') === '#' + id;
        link.classList.toggle('is-active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
    if ('IntersectionObserver' in window) {
      var observer = new IntersectionObserver(function (entries) {
        var visible = entries.filter(function (entry) { return entry.isIntersecting; });
        if (visible.length) {
          visible.sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
          markGroup(visible[0].target.id);
        }
      }, { rootMargin: '-100px 0px -68% 0px', threshold: 0 });
      Array.prototype.forEach.call(groups, function (group) { observer.observe(group); });
    }
    var nav = document.querySelector('.navbar-custom');
    function updateNavOffset() {
      var visible = nav && nav.classList.contains('is-fixed') && nav.classList.contains('is-visible');
      root.style.setProperty('--dash-nav-offset', visible ? Math.ceil(nav.getBoundingClientRect().height) + 'px' : '0px');
    }
    window.addEventListener('scroll', updateNavOffset, { passive: true });
    if (nav && 'MutationObserver' in window) new MutationObserver(updateNavOffset).observe(nav, { attributes: true, attributeFilter: ['class'] });
    updateNavOffset();
  }
  initDashboardControls();

  // ---- 阅读趋势 (views_daily)
  function loadTrend(days) {
    var bars = document.querySelector('#dash-trend .dash-bars'), top = document.querySelector('#dash-trend-top .dash-trend-top'), sum = document.querySelector('#dash-trend .dash-trend-sum');
    getJson(api + '/views/daily?days=' + days).then(function (data) {
      var byDay = {};
      (data.days || []).forEach(function (d) { byDay[d.day] = d.views; });
      // one bar per calendar day, zero-filled, oldest first (Beijing dates)
      var list = [], now = Date.now() + 8 * 3600e3;
      for (var i = days - 1; i >= 0; i--) { var day = new Date(now - i * 86400e3).toISOString().slice(0, 10); list.push({ day: day, views: byDay[day] || 0 }); }
      var max = Math.max.apply(null, list.map(function (d) { return d.views; }).concat([1]));
      var total = list.reduce(function (n, d) { return n + d.views; }, 0);
      sum.textContent = '这 ' + days + ' 天合计 ' + fmt(total) + ' 次，日均 ' + Math.round(total / days) + '。';
      if (days === 30) {
        setKpi('views30', fmt(total));
        setKpi('daily30', fmt(Math.round(total / 30)));
      }
      bars.innerHTML = list.map(function (d) {
        var hgt = Math.max(d.views ? 2 : 0, Math.round(d.views / max * 100));
        return '<span class="dash-bar" title="' + d.day + '：' + d.views + ' 次"><i style="height:' + hgt + '%"></i></span>';
      }).join('') + '<span class="dash-bar-axis"><small>' + list[0].day + '</small><small>峰值 ' + max + '</small><small>' + list[list.length - 1].day + '</small></span>';
      var paths = (data.paths || []).slice(0, 8);
      top.innerHTML = paths.length ? '<h4>这段时间读得最多</h4><ol class="dash-top">' + paths.map(function (p) {
        return '<li><a href="' + h(p.path) + '">' + h(titleOf(p.path)) + '</a> <span class="dash-muted">' + fmt(p.views) + '</span></li>';
      }).join('') + '</ol>' : '';
    }).catch(function (err) {
      bars.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>';
      if (days === 30) {
        setKpi('views30', null);
        setKpi('daily30', null);
      }
    });
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
  // ---- 系列漏斗 (/stats/top)
  var funnelRows = [], funnelExpanded = false, FUNNEL_TOP = 8;
  function renderFunnel() {
    var host = document.querySelector('#dash-funnel .dash-list');
    if (!host) return;
    var rows = funnelExpanded ? funnelRows : funnelRows.slice(0, FUNNEL_TOP);
    host.innerHTML = rows.map(function (series) {
      var name = series.name || series.key;
      var summaryStats = '首篇 ' + fmt(series.first) + ' → 末篇 ' + fmt(series.last) +
        '（' + Math.round(series.last / series.first * 100) + '%）';
      if (series.biggest) summaryStats += ' · 流失最多：第 ' + (series.biggest.index + 1) + ' 篇（−' + series.biggest.percent + '%）';
      var summary = '<span class="dash-funnel-summary-title"><span class="dash-funnel-summary-name">' + h(name) +
        '</span><span class="dash-funnel-summary-count">' + h(series.posts.length) + ' 篇</span></span>' +
        '<span class="dash-funnel-summary-stats">' + h(summaryStats) + '</span>';
      var steps = series.posts.map(function (post, index) {
        var views = series.postViews[index], previous = index ? series.postViews[index - 1] : 0;
        var change = index && previous ? Math.round((views / previous - 1) * 100) : null;
        var changeText = change == null ? '—' : (change > 0 ? '+' + change : change < 0 ? '−' + Math.abs(change) : '0') + '%';
        var width = Math.max(0, Math.min(100, views / series.first * 100));
        var isDrop = series.biggest && series.biggest.index === index;
        return '<li class="dash-funnel-row' + (isDrop ? ' is-drop' : '') + '">' +
          '<span class="dash-funnel-index">' + (index + 1) + '</span>' +
          '<span class="dash-funnel-title-wrap"><a class="dash-funnel-title" href="' + h(post.path) + '" title="' + h(titleOf(post.path)) + '">' + h(post.subtitle || titleOf(post.path)) + '</a>' +
          (post.recap ? '<span class="dash-funnel-recap">自测</span>' : '') + '</span>' +
          '<span class="dash-funnel-bar" role="img" aria-label="阅读 ' + fmt(views) + ' / ' + fmt(series.first) + '"><i style="width:' + width + '%"></i></span>' +
          '<span class="dash-funnel-views">' + fmt(views) + '</span>' +
          '<span class="dash-funnel-change">' + changeText + '</span></li>';
      }).join('');
      return '<details class="dash-funnel-item"><summary>' + summary + '</summary><ol class="dash-funnel-steps">' + steps + '</ol></details>';
    }).join('') + (funnelRows.length > FUNNEL_TOP
      ? '<p class="dash-funnel-expand"><a href="#" class="dash-toggle">' + (funnelExpanded ? '只看 TOP ' + FUNNEL_TOP : '展开全部 ' + funnelRows.length + ' 个系列') + '</a></p>'
      : '');
    var toggle = host.querySelector('.dash-toggle');
    if (toggle) toggle.addEventListener('click', function (event) {
      event.preventDefault();
      funnelExpanded = !funnelExpanded;
      renderFunnel();
    });
  }
  function loadFunnel() {
    var host = document.querySelector('#dash-funnel .dash-list');
    if (!host) return;
    getJson(api + '/stats/top?limit=500').then(function (data) {
      var views = Object.create(null);
      (data.rows || []).forEach(function (row) {
        views[row.path] = Math.max(0, Number(row.views) || 0);
      });
      funnelRows = (window.DASH_SERIES || []).map(function (series) {
        var posts = (series.posts || []).map(function (post) {
          return { path: post[0], subtitle: post[1], recap: !!post[2] };
        });
        if (posts.length < 2) return null;
        var first = views[posts[0].path] || 0;
        if (!first) return null;
        var postViews = posts.map(function (post) { return views[post.path] || 0; });
        var lastIndex = posts.length - 1;
        for (var i = posts.length - 1; i >= 0; i--) {
          if (!posts[i].recap) { lastIndex = i; break; }
        }
        var biggest = null;
        for (var step = 1; step < postViews.length; step++) {
          var previous = postViews[step - 1];
          if (previous < 20) continue;
          var drop = 1 - postViews[step] / previous;
          var percent = Math.round(drop * 100);
          if (drop > 0 && percent > 0 && (!biggest || drop > biggest.drop)) {
            biggest = { index: step, drop: drop, percent: percent };
          }
        }
        return {
          key: series.key,
          name: series.name,
          posts: posts,
          first: first,
          last: postViews[lastIndex],
          postViews: postViews,
          biggest: biggest
        };
      }).filter(function (series) { return !!series; });
      funnelRows.sort(function (a, b) { return b.first - a.first || String(a.name || a.key).localeCompare(String(b.name || b.key)); });
      if (!funnelRows.length) {
        host.innerHTML = '<p class="dash-muted">还没有阅读量足够的系列。</p>';
        return;
      }
      renderFunnel();
    }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  loadFunnel();
  // ---- 值得翻新的老文章: views × log(age) for tech posts whose date / updated is >= 3 years old
  var meta = window.DASH_META || {}, REFRESH_YEARS = 3, REFRESH_TOP = 15;
  var srcBase = 'https://github.com/' + repo + '/edit/master/';
  function editUrl(path) { return srcBase + String(path || '').split('/').map(encodeURIComponent).join('/'); }

  // ---- 写作排期 (future posts are read from raw _posts at build time)
  function renderWritingSchedule() {
    var section = document.getElementById('dash-schedule');
    var tools = window.WritingSchedule;
    if (!section || !tools) return;
    var summary = section.querySelector('.dash-schedule-summary');
    var empty = section.querySelector('.dash-schedule-empty');
    var calendar = section.querySelector('.dash-schedule-calendar');
    var legend = section.querySelector('.dash-schedule-legend');
    var groupsHost = section.querySelector('.dash-schedule-groups');
    var data = window.DASH_SCHEDULE || {};
    var today = tools.beijingToday();
    var posts = tools.upcoming(data.scheduled || []);
    var grouped = tools.groupSchedule(posts);
    var paletteBySeries = Object.create(null);
    grouped.series.forEach(function (group) { paletteBySeries[group.key] = group.paletteIndex; });
    var byDate = Object.create(null);
    posts.forEach(function (post) { (byDate[post.date] || (byDate[post.date] = [])).push(post); });

    function itemId(post) { return 'dash-schedule-item-' + String(post.path).replace(/[^a-zA-Z0-9_-]/g, '-'); }
    function postRows(items, isDraft) {
      return '<ol class="dash-schedule-posts">' + items.map(function (post) {
        return '<li class="dash-schedule-post' + (isDraft ? ' is-draft' : '') + '" id="' + h(itemId(post)) + '">' +
          (post.date ? '<time datetime="' + h(post.date) + '">' + h(post.date) + '</time>' : '') +
          '<span class="dash-schedule-post-title">' + h(post.title) + '</span>' +
          '<a class="dash-brief-link" href="' + h(editUrl(post.path)) + '" target="_blank" rel="noopener" title="在 GitHub 编辑源文件">GitHub 编辑</a></li>';
      }).join('') + '</ol>';
    }
    function levelChip(level) {
      var labels = { 1: '入门', 2: '进阶', 3: '深入' }, label = labels[Number(level)];
      return label ? '<span class="st-level l' + Number(level) + '">' + label + '</span>' : '';
    }
    function seriesRows() {
      var metaBySeries = data.series || {}, published = data.published || {};
      return grouped.series.map(function (group) {
        var meta = metaBySeries[group.key] || {};
        var count = Number(published[group.key]) || 0;
        var planned = count + group.posts.length;
        var progress = planned ? Math.round(count / planned * 100) : 0;
        var recap = group.posts.filter(function (post) { return post.recap; })[0];
        var endDate = recap ? recap.date : group.posts[group.posts.length - 1].date;
        var endLabel = '完结 ' + endDate + (recap ? '' : '（末篇）');
        return '<details class="dash-schedule-series"><summary>' +
          '<span class="dash-schedule-series-name">' + h(meta.name || group.key) + '</span>' + levelChip(meta.level) +
          '<span class="dash-schedule-progress" role="progressbar" aria-valuemin="0" aria-valuemax="' + planned + '" aria-valuenow="' + count + '" aria-label="已发 ' + count + ' / 计划 ' + planned + '">' +
          '<span style="width:' + progress + '%"></span></span>' +
          '<span class="dash-schedule-progress-text">已发 ' + count + ' / 计划 ' + planned + '</span>' +
          '<span class="dash-schedule-next">下一篇 ' + h(group.nextDate) + '</span>' +
          '<span class="dash-schedule-end">' + h(endLabel) + '</span></summary>' +
          postRows(group.posts) + '</details>';
      }).join('');
    }
    function calendarHtml() {
      if (!posts.length) return '';
      var start = tools.addDays ? tools.addDays(today, 1) : new Date(Date.parse(today + 'T00:00:00Z') + 86400e3).toISOString().slice(0, 10);
      var last = posts[posts.length - 1].date;
      var collisions = Object.create(null), gaps = Object.create(null);
      tools.collisionDates(posts).forEach(function (date) { collisions[date] = true; });
      tools.gapDates(posts, today, last).forEach(function (date) { gaps[date] = true; });
      var cursor = new Date(Date.parse(start.slice(0, 7) + '-01T00:00:00Z'));
      var finalMonth = last.slice(0, 7), months = [];
      while (cursor.toISOString().slice(0, 7) <= finalMonth) {
        var year = cursor.getUTCFullYear(), month = cursor.getUTCMonth(), key = cursor.toISOString().slice(0, 7);
        var days = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
        var displayStart = key === start.slice(0, 7) ? Number(start.slice(8, 10)) : 1;
        var displayEnd = key === last.slice(0, 7) ? Number(last.slice(8, 10)) : days;
        var firstOffset = (new Date(Date.UTC(year, month, displayStart)).getUTCDay() + 6) % 7;
        var cells = [];
        for (var blank = 0; blank < firstOffset; blank++) cells.push('<span class="dash-cal-blank" aria-hidden="true"></span>');
        for (var day = displayStart; day <= displayEnd; day++) {
          var date = key + '-' + String(day).padStart(2, '0');
          var items = byDate[date] || [];
          var className = 'dash-cal-day';
          if (items.length) {
            var colored = items.filter(function (post) { return post.series; })[0];
            className += colored ? ' series-color-' + paletteBySeries[colored.series] : ' is-unassigned';
          }
          if (collisions[date]) className += ' is-collision';
          if (gaps[date]) className += ' is-gap';
          var titles = items.map(function (post) { return post.title; });
          var tip = date + (titles.length ? '\n' + titles.join('\n') : '');
          cells.push('<button type="button" class="' + className + '" data-date="' + date + '"' +
            (items.length ? '' : ' disabled') + ' title="' + h(tip) + '" aria-label="' + h(tip) + '">' +
            '<time datetime="' + date + '">' + day + '</time>' +
            (items.length > 1 ? '<small>' + items.length + ' 篇</small>' : '') + '</button>');
        }
        while (cells.length % 7) cells.push('<span class="dash-cal-blank" aria-hidden="true"></span>');
        months.push('<div class="dash-schedule-month"><h4>' + year + ' 年 ' + (month + 1) + ' 月</h4>' +
          '<div class="dash-cal-grid"><span class="dash-cal-weekday">一</span><span class="dash-cal-weekday">二</span><span class="dash-cal-weekday">三</span><span class="dash-cal-weekday">四</span><span class="dash-cal-weekday">五</span><span class="dash-cal-weekday">六</span><span class="dash-cal-weekday">日</span>' +
          cells.join('') + '</div></div>');
        cursor.setUTCMonth(cursor.getUTCMonth() + 1);
      }
      return '<div class="dash-schedule-months">' + months.join('') + '</div>';
    }

    if (posts.length) {
      var last = posts[posts.length - 1].date;
      var daysLeft = Math.round((Date.parse(last + 'T00:00:00Z') - Date.parse(today + 'T00:00:00Z')) / 86400e3);
      summary.textContent = '已排期 ' + posts.length + ' 篇，排到 ' + last + '（还有 ' + daysLeft + ' 天）';
      empty.hidden = true;
      calendar.innerHTML = calendarHtml();
      legend.innerHTML = '<span class="dash-schedule-legend-title">图例</span>' +
        grouped.series.map(function (group) {
          return '<span><i class="dash-schedule-swatch series-color-' + group.paletteIndex + '"></i>' + h((data.series && data.series[group.key] && data.series[group.key].name) || group.key) + '</span>';
        }).join('') +
        (grouped.unassigned.length ? '<span><i class="dash-schedule-swatch is-unassigned"></i>未归入系列</span>' : '') +
        '<span><i class="dash-schedule-swatch is-collision"></i>冲突</span><span><i class="dash-schedule-swatch is-gap"></i>空档 ≥3 天</span>';
    } else {
      summary.textContent = '已排期 0 篇';
      empty.hidden = false;
      calendar.innerHTML = '';
      legend.innerHTML = '';
    }

    var groups = [];
    if (grouped.series.length) groups.push('<div class="dash-schedule-series-list"><h4>系列进度</h4>' + seriesRows() + '</div>');
    if (grouped.unassigned.length) groups.push('<section class="dash-schedule-group"><h4>未归入系列 <small>' + grouped.unassigned.length + ' 篇</small></h4>' + postRows(grouped.unassigned) + '</section>');
    var drafts = data.drafts || [];
    if (drafts.length) groups.push('<section class="dash-schedule-group dash-schedule-drafts"><h4>草稿（_drafts） <small>' + drafts.length + ' 篇</small></h4>' + postRows(drafts, true) + '</section>');
    groupsHost.innerHTML = groups.join('');

    calendar.addEventListener('click', function (event) {
      var cell = event.target.closest('.dash-cal-day');
      if (!cell || cell.disabled) return;
      var items = byDate[cell.getAttribute('data-date')] || [];
      var target = items.length && document.getElementById(itemId(items[0]));
      if (!target) return;
      var details = target.closest('details');
      if (details) details.open = true;
      window.setTimeout(function () {
        var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        target.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'center' });
        target.classList.remove('is-flashing');
        void target.offsetWidth;
        target.classList.add('is-flashing');
        window.setTimeout(function () { target.classList.remove('is-flashing'); }, 1800);
      }, 40);
    });
  }
  renderWritingSchedule();

  function loadRefresh() {
    var host = document.querySelector('#dash-refresh .dash-list');
    if (!host) return;
    getJson(api + '/stats/top?limit=500').then(function (data) {
      var now = Date.now(), rows = [];
      (data.rows || []).forEach(function (r) {
        var m = meta[r.path];
        if (!m || m[3] === 'life' || !r.views) return;
        var ref = m[1] || m[0], years = (now - new Date(ref).getTime()) / 31557600000;
        if (years < REFRESH_YEARS) return;
        rows.push({ path: r.path, views: r.views, up: r.up || 0, years: years, ref: ref, updated: !!m[1], src: m[2], score: r.views * Math.log(1 + years) });
      });
      rows.sort(function (a, b) { return b.score - a.score; });
      if (!rows.length) { host.innerHTML = '<p class="dash-muted">没有三年以上还在被读的文章。</p>'; return; }
      host.innerHTML = '<ol class="dash-top dash-refresh">' + rows.slice(0, REFRESH_TOP).map(function (r) {
        return '<li><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a> <span class="dash-muted">' + (r.updated ? '更新于' : '写于') + ' ' + h(r.ref.slice(0, 7)) + ' · ' + Math.floor(r.years) + ' 年 · 阅读 ' + fmt(r.views) + (r.up ? ' · 有用 ' + r.up : '') + '</span> ' +
          '<a class="dash-brief-link" href="#brief=' + h(r.path) + '" title="这篇文章的修订简报">简报</a> <a class="dash-brief-link" href="' + h(editUrl(r.src)) + '" target="_blank" rel="noopener" title="在 GitHub 编辑源文件">编辑</a></li>';
      }).join('') + '</ol>' + (rows.length > REFRESH_TOP ? '<p class="dash-muted">共 ' + rows.length + ' 篇符合条件，只列前 ' + REFRESH_TOP + '。</p>' : '');
    }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  loadRefresh();

  var sortHeads = document.querySelectorAll('#dash-posts th[data-sort]');
  Array.prototype.forEach.call(sortHeads, function (th) {
    th.addEventListener('click', function () { activate(sortHeads, th); postSort = th.getAttribute('data-sort'); renderPosts(); });
  });
  loadPosts();

  // ---- 读者划出来的句子 (/reactions/top)
  var passageRows = [], passageKind = 'up', passagesExpanded = false, PASSAGES_TOP = 10;
  function renderPassages(host) {
    var rows = passageRows, kind = passageKind;
    if (!rows.length) {
      host.innerHTML = '<p class="dash-muted">还没有人' + ({ up: '点赞', share: '分享' }[kind] || '点赞') + '。</p>';
      return;
    }
    var shown = passagesExpanded ? rows : rows.slice(0, PASSAGES_TOP);
    host.innerHTML = '<ol class="dash-quotes">' + shown.map(function (r) {
      var ch = isChapter(r), quote = ch ? r.quote.slice(FB.CHAPTER_PREFIX.length) : r.quote;
      return '<li><a class="dash-quote" title="' + h(quote) + '" href="' + h(r.path) + (ch ? '' : '#annot-' + h(r.hash)) + '">' + h(quote) + '</a>' +
        '<div class="dash-quote-meta">' + (ch ? '<span class="dash-tag">章节</span> · ' : '') + '<a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a>' +
        (r.section && !ch ? ' <span class="dash-muted">› ' + h(r.section) + '</span>' : '') + ' · ' +
        (r.up ? '<span><i class="fa fa-thumbs-up"></i> ' + r.up + (ch ? ' 点赞' : '') + '</span> ' : '') +
        (r.share ? '<span><i class="fa fa-share-alt"></i> ' + r.share + '</span> ' : '') +
        (r.updated_at ? '<span class="dash-muted">· ' + ago(r.updated_at) + '</span>' : '') +
        '</div></li>';
    }).join('') + '</ol>' +
      (rows.length > PASSAGES_TOP ? '<p class="dash-passage-toggle"><a href="#" class="dash-toggle">' + (passagesExpanded ? '收起' : '展开全部 ' + rows.length + ' 条') + '</a></p>' : '');
  }
  function loadPassages(kind) {
    var host = document.querySelector('#dash-passages .dash-list');
    getJson(api + '/reactions/top?kind=' + kind + '&limit=30').then(function (data) {
      passageRows = data.rows || [];
      passageKind = kind;
      passagesExpanded = false;
      renderPassages(host);
    }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  document.querySelector('#dash-passages .dash-list').addEventListener('click', function (e) {
    var toggle = e.target.closest('.dash-toggle');
    if (!toggle) return;
    e.preventDefault();
    passagesExpanded = !passagesExpanded;
    renderPassages(e.currentTarget);
  });
  var kindLinks = document.querySelectorAll('#dash-passages [data-kind]');
  Array.prototype.forEach.call(kindLinks, function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); activate(kindLinks, a); loadPassages(a.getAttribute('data-kind')); });
  });
  loadPassages('up');

  // ---- 修订简报 (#brief=/slug.html): everything readers said about one post, as
  // Markdown for the author — and for the AI that revises the post. Sources:
  // worker /feedback (D1 reactions), the post's Discussion via the
  // worker's anonymous /discussions relay, the repo's 划线评论 issues (REST,
  // state = 「已修正」), and the live page itself to tell which quotes still anchor.
  var briefBody = document.querySelector('#dash-brief .dash-brief-body');
  var briefPick = document.querySelector('#dash-brief .dash-brief-pick');
  (window.DASH_POSTS || []).forEach(function (p) { var o = document.createElement('option'); o.value = p[0]; o.textContent = p[1]; briefPick.appendChild(o); });
  briefPick.addEventListener('change', function () { if (briefPick.value) location.hash = 'brief=' + briefPick.value; });

  function isChapter(r) { return (r.quote || '').indexOf(FB.CHAPTER_PREFIX) === 0; }
  // `path` is the data path (= `comments_path` for a renamed post); the live
  // page is DASH_META[path][4]. A jekyll-redirect-from stub is followed once.
  function loadArticle(path) {
    var m = meta[path], url = (m && m[4]) || path;
    return fetch(url).then(function (r) { return r.text(); }).then(function (html) {
      var re = /<meta http-equiv="refresh" content="0; url=([^"]+)">/.exec(html);
      return re ? fetch(re[1]).then(function (r) { return r.text(); }) : html;
    }).then(function (html) { return FB.articleFromHtml(DOM, html, titleOf(path)); }).catch(function () { return null; });
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
      var md = FB.buildBrief({ dom: DOM, path: path, title: titleOf(path), fb: res[0], disc: res[1], issues: issues, article: res[3], site: siteUrl || location.origin }).markdown;
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

  // open issues by label, public REST, fetched once and shared by 待修订的文章 and 待处理
  var openIssuesCache = {}, openIssueErrors = {};
  function openIssues(label) {
    if (!openIssuesCache[label]) openIssuesCache[label] = fetch('https://api.github.com/repos/' + repo + '/issues?state=open&per_page=30&labels=' + encodeURIComponent(label))
      .then(function (r) {
        if (!r.ok) { openIssueErrors[label] = true; return []; }
        return r.json();
      }).then(function (list) { return list.filter(function (is) { return !is.pull_request; }); }).catch(function () { openIssueErrors[label] = true; return []; });
    return openIssuesCache[label];
  }
  function queueIssuePath(is) { var m = /<!-- feedback-queue: (\S+) -->/.exec(is.body || ''); return m ? m[1] : ''; }

  // ---- 待修订的文章: posts with open 划线评论 issues or an open 待修订 issue.
  // Discussion comments are not fetched here (one relay call per post); the full brief has them.
  var QUEUE_TOP = 15;
  function loadBriefQueue() {
    var host = document.querySelector('#dash-brief .dash-brief-queue');
    if (!host) return;
    Promise.all([getJson(api + '/feedback').then(function (d) { return d.posts || {}; }), openIssues('划线评论'), openIssues('待修订')]).then(function (res) {
      var posts = res[0], errata = res[1], queued = {};
      res[2].forEach(function (is) { var p = queueIssuePath(is); if (p) queued[p] = is; });
      var paths = Object.keys(posts);
      Object.keys(queued).forEach(function (p) { if (paths.indexOf(p) < 0) paths.push(p); });
      errata.forEach(function (is) {
        var m = /(?:https?:\/\/[^/\s)]+)?(\/[^\s)#]+\.html)(?:#[^\s)]*)?/.exec(is.body || '');
        var path = m && m[1];
        if (path && paths.indexOf(path) < 0) paths.push(path);
      });
      var rows = [];
      paths.forEach(function (path) {
        var issues = errata.filter(function (is) { return (is.body || '').indexOf(path) >= 0; });
        if (!issues.length && !queued[path]) return;
        var post = posts[path] || {};
        rows.push({ path: path, issues: issues, queued: queued[path], views: post.views || 0, score: issues.length + (queued[path] ? 1 : 0) });
      });
      rows.sort(function (a, b) { return b.score - a.score || b.views - a.views; });
      if (!rows.length) { host.innerHTML = '<p class="dash-muted">没有待修订的文章。</p>'; return; }
      host.innerHTML = '<h4>待修订的文章 <span class="dash-count">' + rows.length + '</span> <small class="dash-muted">有未关闭的划线评论 Issue 或已开「待修订」Issue</small></h4>' +
        '<ol class="dash-top dash-queue">' + rows.slice(0, QUEUE_TOP).map(function (r) {
          var m = meta[r.path], bits = [];
          if (r.issues.length) bits.push('报错 ' + r.issues.map(function (is) { return '<a target="_blank" rel="noopener" href="' + h(is.html_url) + '">#' + is.number + '</a>'; }).join(' '));
          if (r.queued) bits.push('<a target="_blank" rel="noopener" href="' + h(r.queued.html_url) + '">待修订 #' + r.queued.number + '</a>');
          if (r.views) bits.push('阅读 ' + fmt(r.views));
          return '<li><a href="' + h(r.path) + '">' + h(titleOf(r.path)) + '</a> <span class="dash-muted">' + bits.join(' · ') + '</span> ' +
            '<a class="dash-brief-link" href="#brief=' + h(r.path) + '" title="这篇文章的修订简报">简报</a>' + (m ? ' <a class="dash-brief-link" href="' + h(srcBase + m[2]) + '" target="_blank" rel="noopener" title="在 GitHub 编辑源文件">编辑</a>' : '') + '</li>';
        }).join('') + '</ol>' + (rows.length > QUEUE_TOP ? '<p class="dash-muted">共 ' + rows.length + ' 篇，只列前 ' + QUEUE_TOP + '。</p>' : '');
    }).catch(function (err) { host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>'; });
  }
  loadBriefQueue();

  // ---- issues (public REST)
  function loadIssues() {
    var host = document.querySelector('#dash-issues .dash-list');
    Promise.all([openIssues('待修订'), openIssues('划线评论'), openIssues('dead-links')])
      .then(function (res) {
        var queue = res[0], errata = res[1], links = res[2];
        setKpi('queue', openIssueErrors['待修订'] ? null : queue.length, 'count');
        setKpi('errata', openIssueErrors['划线评论'] ? null : errata.length, 'count');
        setKpi('deadlinks', openIssueErrors['dead-links'] ? null : links.length, 'count');
        if (!queue.length && !errata.length && !links.length) {
          host.innerHTML = Object.keys(openIssueErrors).length
            ? '<p class="dash-muted">Issue 状态暂时无法读取。</p>'
            : '<p class="dash-empty-ok">✓ 都清了 · 待修订 0 · 读者报错 0 · 死链 0</p>';
          return;
        }
        // 待修订: one issue per post, opened by the weekly feedback-queue Action; its body is the brief
        var html = '<h4>待修订的文章 <span class="dash-count">' + queue.length + '</span> <a class="dash-more" target="_blank" rel="noopener" href="https://github.com/' + repo + '/issues?q=is%3Aissue+is%3Aopen+label%3A%E5%BE%85%E4%BF%AE%E8%AE%A2">全部 ↗</a></h4>';
        html += queue.length ? '<ul>' + queue.map(function (is) {
          var qp = queueIssuePath(is);
          return '<li><a target="_blank" rel="noopener" href="' + h(is.html_url) + '">#' + is.number + ' ' + h(is.title) + '</a>' + (qp ? ' <a class="dash-brief-link" href="#brief=' + h(qp) + '">简报</a>' : '') + ' <span class="dash-muted">更新于 ' + ago(is.updated_at) + '</span></li>';
        }).join('') + '</ul>' : '<p class="dash-muted">没有。每周一 Action 会把反馈够多的文章开成 Issue。</p>';
        html += '<h4>读者报的错 <span class="dash-count">' + errata.length + '</span> <a class="dash-more" target="_blank" rel="noopener" href="https://github.com/' + repo + '/issues?q=is%3Aissue+is%3Aopen+label%3A%E5%88%92%E7%BA%BF%E8%AF%84%E8%AE%BA">全部 ↗</a></h4>';
        html += errata.length ? '<ul>' + errata.map(function (is) {
          return '<li><a target="_blank" rel="noopener" href="' + h(is.html_url) + '">#' + is.number + ' ' + h(is.title) + '</a> <span class="dash-muted">' + ago(is.created_at) + '</span></li>';
        }).join('') + '</ul>' : '<p class="dash-muted">没有待处理的勘误。</p>';
        html += '<h4>死链报告 <span class="dash-count">' + links.length + '</span></h4>';
        html += links.length ? '<ul>' + links.map(function (is) {
          return '<li><a target="_blank" rel="noopener" href="' + h(is.html_url) + '">#' + is.number + ' ' + h(is.title) + '</a> <span class="dash-muted">更新于 ' + ago(is.updated_at) + '</span></li>';
        }).join('') + '</ul>' : '<p class="dash-muted">没有。</p>';
        host.innerHTML = html;
      }).catch(function (err) {
        host.innerHTML = '<p class="dash-muted">加载失败：' + h(err.message) + '</p>';
        setKpi('queue', null, 'count');
        setKpi('errata', null, 'count');
        setKpi('deadlinks', null, 'count');
      });
  }
  loadIssues();

  // ---- recent comments (GraphQL, needs the giscus login)
  var authEl = document.querySelector('#dash-comments .dash-auth');
  var listEl = document.querySelector('#dash-comments .dash-list');
  function session() { try { var raw = localStorage.getItem(SESSION_KEY); return raw ? JSON.parse(raw) : ''; } catch (e) { return ''; } }
  function getToken() {
    if (!session()) return Promise.reject(new Error('先在下面「最近评论」处登录 GitHub'));
    return fetch(api + '/token', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: session() }) })
      .then(function (r) { return r.json(); }).then(function (d) { if (!d.token) throw new Error('登录已过期'); return d.token; });
  }
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

  // Every top-level comment of the 30 most recently active threads, with its
  // isMinimized flag: a thread is 已处理 once the author minimizes it as
  // RESOLVED on GitHub (js/annotations.js hides minimized comments from the
  // article page too), so 「未处理」 is simply the non-minimized ones.
  var QUERY = 'query($owner: String!, $name: String!, $cat: ID!) { repository(owner: $owner, name: $name) {' +
    ' discussions(first: 30, categoryId: $cat, orderBy: {field: UPDATED_AT, direction: DESC}) { nodes {' +
    ' title url updatedAt comments { totalCount } reactionGroups { content reactors { totalCount } }' +
    ' recent: comments(last: 60) { nodes { id url createdAt bodyText isMinimized author { login } replies(last: 3) { nodes { url createdAt bodyText author { login } } } } } } } } }';
  var MINIMIZE = 'mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: RESOLVED}) { minimizedComment { isMinimized } } }';

  var modeLinks = document.querySelectorAll('#dash-comments [data-mode]');
  var mode = 'open', threads = null;
  Array.prototype.forEach.call(modeLinks, function (a) {
    a.addEventListener('click', function (e) { e.preventDefault(); activate(modeLinks, a); mode = a.getAttribute('data-mode'); renderComments(); });
  });

  function gql(query, variables) {
    return getToken().then(function (token) {
      return fetch('https://api.github.com/graphql', { method: 'POST', headers: { Authorization: 'Bearer ' + token, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query, variables: variables }) }).then(function (r) { return r.json(); });
    }).then(function (res) { if (res.errors) throw new Error(res.errors[0].message); return res.data; });
  }

  function renderComments() {
    if (!threads) return;
    var openTotal = threads.reduce(function (n, d) {
      return n + d.recent.nodes.filter(function (c) { return !c.isMinimized; }).length;
    }, 0);
    setKpi('comments', openTotal, 'count');
    var shown = threads.map(function (d) {
      var open = d.recent.nodes.filter(function (c) { return !c.isMinimized; });
      return { d: d, comments: mode === 'open' ? open : d.recent.nodes.slice(), open: open.length };
    }).filter(function (t) { return t.comments.length; });
    var total = shown.reduce(function (n, t) { return n + t.comments.length; }, 0);
    authEl.innerHTML = '<span class="dash-count">' + total + '</span> <a href="https://github.com/' + repo + '/discussions" target="_blank" rel="noopener">全部讨论 ↗</a>';
    if (!shown.length) { listEl.innerHTML = '<p class="dash-muted">' + (mode === 'open' ? '没有未处理的评论。' : '还没有评论。') + '</p>'; return; }
    listEl.innerHTML = shown.map(function (t) {
      var d = t.d;
      var likes = (d.reactionGroups || []).filter(function (g) { return g.content === 'THUMBS_UP'; }).map(function (g) { return g.reactors.totalCount; })[0] || 0;
      var path = d.title.indexOf('/') === 0 ? d.title : null;
      var items = t.comments.slice().sort(function (a, b) { return a.createdAt < b.createdAt ? 1 : -1; });
      return '<div class="dash-thread">' +
        '<div class="dash-thread-head"><a href="' + h(path || d.url) + '">' + h(path ? titleOf(path) : d.title) + '</a>' +
        ' <span class="dash-muted">' + t.open + ' 条未处理 / ' + d.comments.totalCount + ' 条评论 · 👍 ' + likes + ' · ' + ago(d.updatedAt) + '</span>' +
        ' <a class="dash-more" target="_blank" rel="noopener" href="' + h(d.url) + '">GitHub ↗</a></div>' +
        items.map(function (c) {
          var last = (c.replies.nodes || []).slice(-1)[0];
          return '<div class="dash-comment' + (c.isMinimized ? ' is-done' : '') + '" data-id="' + h(c.id) + '"><b>' + h(c.author ? c.author.login : 'ghost') + '</b> <span class="dash-muted">' + ago(c.createdAt) + '</span>' +
            (c.isMinimized ? ' <span class="dash-muted"><i class="fa fa-check-circle"></i> 已处理</span>' : ' <span class="dash-resolve"><button type="button" data-action="done" title="改好了：在 GitHub 上折叠为 resolved，文章页不再显示"><i class="fa fa-check"></i> 已处理</button></span>') +
            '<a target="_blank" rel="noopener" href="' + h(c.url) + '"> ' + h(c.bodyText.replace(/\s+/g, ' ').slice(0, 140)) + (c.bodyText.length > 140 ? '…' : '') + '</a>' +
            (last ? '<div class="dash-muted">↳ ' + h(last.author ? last.author.login : 'ghost') + '：' + h(last.bodyText.replace(/\s+/g, ' ').slice(0, 100)) + (last.bodyText.length > 100 ? '…' : '') + '</div>' : '') +
            '</div>';
        }).join('') + '</div>';
    }).join('');
  }

  listEl.addEventListener('click', function (e) {
    var btn = e.target.closest('button[data-action="done"]');
    if (!btn) return;
    var row = btn.closest('.dash-comment'), id = row.getAttribute('data-id');
    btn.disabled = true; btn.textContent = '…';
    gql(MINIMIZE, { id: id }).then(function () {
      threads.forEach(function (d) { d.recent.nodes.forEach(function (c) { if (c.id === id) c.isMinimized = true; }); });
      renderComments();
    }).catch(function (err) { btn.disabled = false; btn.innerHTML = '<i class="fa fa-check"></i> 已处理'; alert('折叠失败：' + err.message); });
  });

  function loadComments() {
    if (!session()) {
      var commentKpi = document.querySelector('.dash-kpi[data-kpi="comments"] .dash-kpi-note');
      setKpi('comments', null);
      if (commentKpi) { commentKpi.textContent = '登录后显示'; commentKpi.hidden = false; }
      authEl.textContent = '';
      listEl.innerHTML = '<p class="dash-login-line dash-muted">需要 GitHub 登录查看评论 <a href="#" class="dash-login">登录 GitHub</a></p>';
      listEl.querySelector('.dash-login').addEventListener('click', function (e) { e.preventDefault(); login(); });
      return;
    }
    setKpi('comments', '…');
    authEl.textContent = '';
    var parts = repo.split('/');
    gql(QUERY, { owner: parts[0], name: parts[1], cat: categoryId })
      .then(function (data) { threads = data.repository.discussions.nodes; renderComments(); })
      .catch(function (err) {
        setKpi('comments', null);
        authEl.innerHTML = '<a href="#" class="dash-login">重新登录</a>';
        authEl.querySelector('.dash-login').addEventListener('click', function (e) { e.preventDefault(); try { localStorage.removeItem(SESSION_KEY); } catch (x) { /* ignore */ } login(); });
        listEl.innerHTML = '<p class="dash-muted">' + h(err.message) + '</p>';
      });
  }
  loadComments();
})();
