/*! moments.js — the 随笔 sidebar's live cards (_layouts/moments.html): 随机漫步
 * (a random entry, ⇄ draws another) and 每日回顾 (entries written on this day
 * in earlier years, else on this day of earlier months), both read once from
 * /moments/index.json (_plugins/moments.rb). Nothing happens off /moments/. */
(function () {
  var random = document.querySelector('.ms-random');
  if (!random || !window.fetch) return;
  var review = document.querySelector('.ms-review');
  var base = random.getAttribute('data-base') || '';
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

  random.querySelector('.ms-shuffle').addEventListener('click', shuffle);
  fetch(random.getAttribute('data-src')).then(function (r) { return r.ok ? r.json() : []; }).then(function (d) {
    all = d || [];
    if (!all.length) { random.hidden = true; return; }
    shuffle();
    daily();
  }).catch(function () { random.hidden = true; });
})();
