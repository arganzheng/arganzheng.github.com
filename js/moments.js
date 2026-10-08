/*! moments.js — the 随笔 pages (_layouts/moments.html): the sidebar's live cards
 * — 随机漫步 (a random entry, ⇄ draws another) and 每日回顾 (entries written on
 * this day in earlier years, else on this day of earlier months), both read
 * once from /moments/index.json (_plugins/moments.rb) — and each card's foot:
 * 分享 (the share popover of js/share.js, window.BlogShare; a completed share
 * counts on the month, POST /shares) and, once js/annotations.js reports the
 * signed-in author (`blog:viewer`), 编辑 (a link to the 发布页's edit mode) and
 * 删除 (worker DELETE /moments { month, id }, with the giscus token). */
(function () {
  var random = document.querySelector('.ms-random');
  var list = document.querySelector('.moments-list');
  if (!random && !list) return;
  var review = document.querySelector('.ms-review');
  var base = (random || list).getAttribute('data-base') || '';
  var all = [], last = null;

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function card(e, label) {
    return '<a class="ms-card" href="' + esc(base + e.url) + '">' +
      (label ? '<span class="ms-card-label">' + esc(label) + '</span>' : '') +
      '<time>' + esc(e.date + (e.time ? ' ' + e.time : '')) + (e.place ? ' · ' + esc(e.place) : '') + '</time>' +
      (e.img ? '<img src="' + esc(base + e.img) + '" alt="" loading="lazy">' : '') +
      (e.text ? '<p>' + esc(e.text) + '</p>' : '') + '</a>';
  }

  function shuffle() {
    if (all.length < 1) return;
    var pick = all[Math.floor(Math.random() * all.length)];
    if (all.length > 1 && pick === last) return shuffle();
    last = pick;
    random.querySelector('.ms-slot').innerHTML = card(pick);
  }

  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function daily() {
    if (!review) return;
    var now = new Date(), y = now.getFullYear(), m = now.getMonth() + 1, d = now.getDate();
    var md = '-' + pad(m) + '-' + pad(d), out = [];
    all.forEach(function (e) {          // 那年今日
      var ey = +e.date.slice(0, 4);
      if (e.date.slice(4) === md && ey < y) out.push({ e: e, label: (y - ey) + ' 年前的今天' });
    });
    if (!out.length) all.forEach(function (e) {   // else this day of an earlier month
      var ey = +e.date.slice(0, 4), em = +e.date.slice(5, 7), ed = +e.date.slice(8, 10), ago = (y - ey) * 12 + (m - em);
      if (ed === d && ago > 0) out.push({ e: e, label: ago + ' 个月前的今天' });
    });
    if (!out.length) return;
    review.querySelector('.ms-slot').innerHTML = out.slice(0, 3).map(function (x) { return card(x.e, x.label); }).join('');
    review.hidden = false;
  }

  if (random) {
    random.querySelector('.ms-shuffle').addEventListener('click', shuffle);
    fetch(random.getAttribute('data-src')).then(function (r) { return r.ok ? r.json() : []; }).then(function (d) {
      all = d || [];
      if (!all.length) { random.hidden = true; return; }
      shuffle();
      daily();
    }).catch(function () { random.hidden = true; });
  }

  // ---- the card foot: 分享 · (author) 编辑 / 删除
  if (!list) return;
  var api = (localStorage.getItem('annotationsApi') || list.getAttribute('data-annotations-api') || '').replace(/\/$/, '');
  var path = list.getAttribute('data-path') || '', author = list.getAttribute('data-author') || '';
  var local = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

  function toast(card, msg, ms) {
    var t = card.querySelector('.moment-toast');
    if (!t) return;
    t.textContent = msg; t.hidden = false;
    clearTimeout(t._timer);
    t._timer = setTimeout(function () { t.hidden = true; }, ms || 2500);
  }
  function countShare() {
    if (local || !api || !path) return;
    fetch(api + '/shares', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path }) }).catch(function () { /* ignore */ });
  }
  function ensureToken() {
    var core = window.BlogAnnotations && window.BlogAnnotations.core;
    if (core && core.ensureToken) return core.ensureToken();
    var s; try { s = JSON.parse(localStorage.getItem('giscus-session') || '""'); } catch (e) { s = ''; }
    if (!s) return Promise.reject(new Error('尚未登录 GitHub'));
    return fetch(api + '/token', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: s }) })
      .then(function (r) { return r.json(); }).then(function (d) { if (!d.token) throw new Error('no token'); return d.token; });
  }
  function remove(card, btn) {
    var title = card.getAttribute('data-title') || '', month = card.getAttribute('data-month') || '';
    if (!window.confirm('删除这条随笔？\n' + title + '\n\n会提交一个删除的 commit，图片一起删掉，1–2 分钟后生效。')) return;
    btn.disabled = true; toast(card, '删除中…', 60000);
    ensureToken().then(function (t) {
      return fetch(api + '/moments', { method: 'DELETE', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + t }, body: JSON.stringify({ month: month, id: card.id }) })
        .then(function (r) { return r.json().catch(function () { return {}; }).then(function (d) { if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status)); return d; }); });
    }).then(function () {
      card.classList.add('is-deleted');
      toast(card, '已删除，正在重新部署，1–2 分钟后消失', 60000);
    }).catch(function (e) {
      btn.disabled = false;
      toast(card, '删除失败：' + e.message, 4000);
    });
  }
  list.addEventListener('click', function (e) {
    var share = e.target.closest('.moment-share');
    if (share && window.BlogShare) {
      e.stopPropagation();
      var card = share.closest('.moment');
      window.BlogShare.open(share, {
        url: share.getAttribute('data-url') || location.href.replace(/[#?].*$/, '') + '#' + card.id,
        title: share.getAttribute('data-title') || document.title,
        text: share.getAttribute('data-text') || '',
        onShared: countShare,
        toast: function (msg) { toast(card, msg); }
      });
      return;
    }
    var del = e.target.closest('.moment-del');
    if (del && !del.disabled) remove(del.closest('.moment'), del);
  });
  function onViewer(v) {
    var mine = !!(v && author && v.login === author && api);
    Array.prototype.forEach.call(list.querySelectorAll('.moment-own'), function (n) { n.hidden = !mine; });
  }
  document.addEventListener('blog:viewer', function (e) { onViewer(e.detail); });
  if (window.BlogAnnotations && window.BlogAnnotations.viewer) onViewer(window.BlogAnnotations.viewer());
})();
