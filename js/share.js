/*!
 * share.js — drives every `.post-actions` bar (_includes/post-actions.html):
 *   - 分享 popover: system share sheet (Web Share API), Weibo / X / LinkedIn
 *     intent links, WeChat QR (js/vendor/qrcode.min.js, loaded on first use),
 *     copy link. One popover element, re-anchored to whichever button opened it.
 *   - 赞同 / 反对: anonymous per-post counters in the worker's D1 (POST /votes),
 *     one vote per browser (localStorage), no login. List pages get all their
 *     numbers (votes, views, comments) from one GET /stats?paths=…; on the post
 *     page js/annotations.js supplies views + comment count.
 *   - post page: author-only 「复制为公众号格式」 (lazy js/wechat-export.js).
 * window.PostActions.render(bar, {up, down, mine, views, comments}) paints a bar.
 */
(function () {
  'use strict';

  var bars = Array.prototype.slice.call(document.querySelectorAll('.post-actions'));
  if (!bars.length) return;
  var enc = encodeURIComponent;
  var version = (document.currentScript && (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src + (version ? '?v=' + version : '');
      s.onload = resolve; s.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(s);
    });
  }
  function fmt(n) { return n >= 10000 ? (n / 10000).toFixed(1).replace(/\.0$/, '') + ' 万' : String(n); }

  // ------------------------------------------------------------------ paint
  function render(bar, st) {
    var up = bar.querySelector('.pa-up'), down = bar.querySelector('.pa-down'), vote = bar.querySelector('.pa-vote');
    if (st.up != null) {
      bar.querySelector('.pa-up-n').textContent = st.up ? ' ' + fmt(st.up) : '';
      down.title = st.down ? '反对（' + st.down + '）' : '反对';
      vote.classList.toggle('is-up', st.mine === 'up');
      vote.classList.toggle('is-down', st.mine === 'down');
      up.title = st.mine === 'up' ? '取消赞同' : '赞同';
      if (!bar.classList.contains('is-compact')) {
        var meta = document.querySelector('.post-likes');
        if (meta) meta.textContent = st.up ? ' · ' + fmt(st.up) + ' 人赞同' : '';
      }
    }
    var views = bar.querySelector('.pa-views');
    if (st.views != null) { views.querySelector('b').textContent = fmt(st.views); views.hidden = false; }
    var c = bar.querySelector('.pa-comments b');
    if (st.comments != null) c.textContent = fmt(st.comments) + ' 条';
  }

  function toast(bar, msg, ms) {
    var t = bar.querySelector('.pa-toast');
    if (!t) return;
    t.textContent = msg; t.hidden = false;
    clearTimeout(t._timer);
    t._timer = setTimeout(function () { t.hidden = true; }, ms || 2000);
  }

  // ---------------------------------------------------------------- copying
  function copyText(s) {
    if (navigator.clipboard && navigator.clipboard.writeText && window.isSecureContext && document.hasFocus()) {
      return navigator.clipboard.writeText(s).catch(function () { return legacyCopy(s); });
    }
    return legacyCopy(s);
  }
  function legacyCopy(s) {
    var area = document.createElement('textarea');
    area.value = s; area.setAttribute('readonly', '');
    area.style.position = 'fixed'; area.style.opacity = '0';
    document.body.appendChild(area); area.select();
    var ok = document.execCommand('copy');
    document.body.removeChild(area);
    return ok ? Promise.resolve() : Promise.reject(new Error('copy failed'));
  }

  // ---------------------------------------------------------- share popover
  var pop = null, popFor = null;
  function ensurePop() {
    if (pop) return pop;
    pop = document.createElement('div');
    pop.className = 'pa-share-pop';
    pop.setAttribute('role', 'menu');
    pop.innerHTML =
      (navigator.share ? '<button type="button" class="pa-sp-item" data-k="native"><i class="fa fa-share-alt"></i>系统分享…</button>' : '') +
      '<a class="pa-sp-item" data-k="weibo" target="_blank" rel="noopener noreferrer"><i class="fa fa-weibo"></i>微博</a>' +
      '<a class="pa-sp-item" data-k="x" target="_blank" rel="noopener noreferrer"><i class="fa fa-twitter"></i>X</a>' +
      '<a class="pa-sp-item" data-k="linkedin" target="_blank" rel="noopener noreferrer"><i class="fa fa-linkedin"></i>LinkedIn</a>' +
      '<button type="button" class="pa-sp-item" data-k="wechat"><i class="fa fa-weixin"></i>微信扫一扫</button>' +
      '<button type="button" class="pa-sp-item" data-k="copy"><i class="fa fa-link"></i>复制链接</button>' +
      '<div class="pa-sp-qr" hidden><span class="pa-sp-qr-img"></span><span class="pa-sp-qr-hint">微信扫一扫，分享给朋友或朋友圈</span></div>';
    document.body.appendChild(pop);
    pop.addEventListener('click', function (e) {
      var item = e.target.closest('.pa-sp-item');
      if (!item || !popFor) return;
      var k = item.getAttribute('data-k'), d = dataOf(popFor);
      if (k === 'native') { navigator.share({ title: d.title, text: d.text || d.title, url: d.url }).catch(function () { /* cancelled */ }); closePop(); }
      else if (k === 'wechat') { e.preventDefault(); showQR(d.url); }
      else if (k === 'copy') { copyText(d.url).then(function () { toast(popFor, '已复制链接'); }, function () { toast(popFor, '复制失败'); }); closePop(); }
      else closePop();
    });
    document.addEventListener('click', function (e) { if (pop && !pop.hidden && !pop.contains(e.target) && !e.target.closest('.pa-share')) closePop(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closePop(); });
    window.addEventListener('resize', closePop);
    return pop;
  }
  function dataOf(bar) {
    return {
      url: bar.getAttribute('data-url') || location.href.replace(/[#?].*$/, ''),
      title: bar.getAttribute('data-title') || document.title,
      text: bar.getAttribute('data-text') || ''
    };
  }
  function openPop(bar, btn) {
    var p = ensurePop(), d = dataOf(bar);
    popFor = bar;
    p.querySelector('[data-k="weibo"]').href = 'https://service.weibo.com/share/share.php?url=' + enc(d.url) + '&title=' + enc(d.title + (d.text ? ' — ' + d.text : ''));
    p.querySelector('[data-k="x"]').href = 'https://twitter.com/intent/tweet?url=' + enc(d.url) + '&text=' + enc(d.title);
    p.querySelector('[data-k="linkedin"]').href = 'https://www.linkedin.com/sharing/share-offsite/?url=' + enc(d.url);
    var qr = p.querySelector('.pa-sp-qr'); qr.hidden = true; qr.querySelector('.pa-sp-qr-img').innerHTML = '';
    p.hidden = false; p.style.visibility = 'hidden';
    var r = btn.getBoundingClientRect(), pw = p.offsetWidth, ph = p.offsetHeight;
    var left = Math.min(Math.max(8, r.left + window.scrollX), window.scrollX + document.documentElement.clientWidth - pw - 8);
    var below = r.bottom + 8 + ph < window.innerHeight || r.top < ph + 8;
    p.style.left = left + 'px';
    p.style.top = (below ? r.bottom + window.scrollY + 8 : r.top + window.scrollY - ph - 8) + 'px';
    p.classList.toggle('is-above', !below);
    p.style.visibility = '';
    btn.setAttribute('aria-expanded', 'true');
  }
  function closePop() {
    if (!pop || pop.hidden) return;
    pop.hidden = true;
    if (popFor) popFor.querySelector('.pa-share').setAttribute('aria-expanded', 'false');
    popFor = null;
  }
  function showQR(url) {
    var ready = window.qrcode ? Promise.resolve() : loadScript('/js/vendor/qrcode.min.js');
    ready.then(function () {
      var qr = window.qrcode(0, 'M'); qr.addData(url); qr.make();
      var box = pop.querySelector('.pa-sp-qr');
      box.querySelector('.pa-sp-qr-img').innerHTML = qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
      box.hidden = false;
    }, function () { if (popFor) toast(popFor, '二维码加载失败'); });
  }

  // ------------------------------------------------------------- votes
  // Article 赞同 / 反对 are anonymous counters kept by the worker (D1), like page
  // views: no GitHub login, one vote per browser remembered in localStorage.
  // (Comment votes stay GitHub reactions — those need an identity.)
  function apiBase(bar) { return (localStorage.getItem('annotationsApi') || bar.getAttribute('data-annotations-api') || '').replace(/\/$/, ''); }
  function myVote(path) { try { return localStorage.getItem('vote:' + path) || null; } catch (e) { return null; } }
  function remember(path, dir) { try { if (dir) localStorage.setItem('vote:' + path, dir); else localStorage.removeItem('vote:' + path); } catch (e) { /* ignore */ } }
  var local = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

  function vote(bar, dir) {
    var st = bar._stats, api = apiBase(bar), path = bar.getAttribute('data-path');
    if (!st || bar._pending || !api) return;
    var prev = st.mine, next = prev === dir ? null : dir;
    var before = { up: st.up, down: st.down, mine: st.mine };
    if (prev) st[prev] = Math.max(0, st[prev] - 1);
    if (next) st[next] += 1;
    st.mine = next;
    render(bar, st);
    if (local) { remember(path, next); return; } // previews don't count
    bar._pending = true;
    fetch(api + '/votes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path, dir: next, prev: prev }) })
      .then(function (r) { return r.json().then(function (d) { if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status)); return d; }); })
      .then(function (d) { st.up = d.up; st.down = d.down; remember(path, next); render(bar, st); })
      .catch(function (err) { st.up = before.up; st.down = before.down; st.mine = before.mine; render(bar, st); toast(bar, '投票失败：' + err.message, 3000); })
      .then(function () { bar._pending = false; });
  }
  function bindVotes(bar) {
    bar.querySelector('.pa-up').addEventListener('click', function () { vote(bar, 'up'); });
    bar.querySelector('.pa-down').addEventListener('click', function () { vote(bar, 'down'); });
  }
  // Post page: one GET /votes for this article.
  function loadVotes(bar) {
    var api = apiBase(bar), path = bar.getAttribute('data-path');
    if (!api) return;
    fetch(api + '/votes?path=' + enc(path)).then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)); }).then(function (d) {
      bar._stats = { up: d.up || 0, down: d.down || 0, mine: myVote(path) };
      render(bar, bar._stats);
    }).catch(function () { bar._stats = { up: 0, down: 0, mine: myVote(path) }; render(bar, bar._stats); });
  }
  // List pages: one GET /stats for every bar (views, comments, votes).
  function loadStats(list) {
    var api = apiBase(list[0]);
    if (!api) return;
    var byPath = {};
    list.forEach(function (b) { byPath[b.getAttribute('data-path')] = b; });
    var paths = Object.keys(byPath), chunks = [];
    while (paths.length) chunks.push(paths.splice(0, 20));
    chunks.forEach(function (chunk) {
      fetch(api + '/stats?paths=' + enc(chunk.join(','))).then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)); }).then(function (data) {
        chunk.forEach(function (p) {
          var it = data.items && data.items[p], bar = byPath[p];
          if (!it || !bar) return;
          bar._stats = { up: it.up || 0, down: it.down || 0, mine: myVote(p), views: it.views || 0, comments: it.comments || 0 };
          render(bar, bar._stats);
        });
      }).catch(function () { /* counters stay blank */ });
    });
  }

  // ------------------------------------------------------------------- wire
  var compact = bars.filter(function (b) { return b.classList.contains('is-compact'); });
  bars.forEach(function (bar) {
    var shareBtn = bar.querySelector('.pa-share');
    shareBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (pop && !pop.hidden && popFor === bar) closePop(); else openPop(bar, shareBtn);
    });
    bindVotes(bar);
  });
  if (compact.length) loadStats(compact);
  bars.filter(function (b) { return !b.classList.contains('is-compact'); }).forEach(loadVotes);

  // ------------------------------------------------- author: WeChat export
  var exportBtn = document.querySelector('.post-actions:not(.is-compact) .pa-export');
  if (exportBtn) {
    var bar = exportBtn.closest('.post-actions'), author = bar.getAttribute('data-author') || '';
    var onViewer = function (v) { exportBtn.hidden = !(v && author && v.login === author); };
    document.addEventListener('blog:viewer', function (e) { onViewer(e.detail); });
    if (window.BlogAnnotations && window.BlogAnnotations.viewer) onViewer(window.BlogAnnotations.viewer());
    exportBtn.addEventListener('click', function () {
      exportBtn.disabled = true;
      var d = dataOf(bar);
      var ready = window.WechatExport ? Promise.resolve() : loadScript('/js/wechat-export.js');
      ready.then(function () { return window.WechatExport.copy({ url: d.url, title: d.title }); }).then(function (info) {
        toast(bar, '已复制（' + info.images + ' 图 · ' + info.refs + ' 条参考链接），到公众号 / 知乎编辑器里粘贴', 4000);
      }, function (err) {
        toast(bar, '失败：' + (err && err.message || err), 4000);
      }).then(function () { exportBtn.disabled = false; });
    });
  }

  window.PostActions = { render: render };
})();
