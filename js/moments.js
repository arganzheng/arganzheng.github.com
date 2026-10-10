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
  var list = document.querySelector('.post-container.moments > .moments-list[data-path]') ||
    document.querySelector('.moments-list');
  if (!random && !list) return;
  var stream = list && list.closest('.post-container.moments');
  var review = document.querySelector('.ms-review');
  var base = (random || list).getAttribute('data-base') || '';
  var all = [], last = null, indexPromise = null;

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function loadIndex(src) {
    if (!indexPromise) {
      indexPromise = fetch(src).then(function (r) { return r.ok ? r.json() : []; }).then(function (d) { return d || []; });
    }
    return indexPromise;
  }
  window.MomentsIndex = { load: loadIndex };

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
    loadIndex(random.getAttribute('data-src')).then(function (d) {
      all = d;
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
  var qrScript;
  function qrImage(url) {
    if (!window.qrcode) {
      if (!qrScript) qrScript = new Promise(function (resolve, reject) {
        var script = document.createElement('script');
        script.src = base + '/js/vendor/qrcode.min.js' + ((document.currentScript && document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1] ? '?v=' + RegExp.$1 : '');
        script.onload = resolve; script.onerror = reject; document.head.appendChild(script);
      });
      return qrScript.then(function () { return qrImage(url); });
    }
    var qr = window.qrcode(0, 'M'); qr.addData(url); qr.make();
    var svg = qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
    var img = new Image();
    img.src = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
    return new Promise(function (resolve) { img.onload = function () { URL.revokeObjectURL(img.src); resolve(img); }; img.onerror = function () { resolve(null); }; });
  }
  function loadImage(url) {
    return new Promise(function (resolve) {
      if (!url || new URL(url, location.href).origin !== location.origin) return resolve(null);
      var img = new Image(); img.onload = function () { resolve(img); }; img.onerror = function () { resolve(null); }; img.src = url;
    });
  }
  function wrap(ctx, text, width, limit) {
    var tokens = [], word = '';
    function flush() { if (word) { tokens.push(word); word = ''; } }
    Array.from(text).forEach(function (c) {
      if (/\s/.test(c)) { flush(); if (c === '\n') tokens.push('\n'); else if (tokens[tokens.length - 1] !== ' ') tokens.push(' '); }
      else if (/[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]/.test(c)) { flush(); tokens.push(c); }
      else word += c;
    });
    flush();
    var lines = [], line = '';
    tokens.forEach(function (token) {
      if (token === '\n') { lines.push(line.trim()); line = ''; return; }
      if (token === ' ' && !line) return;
      var next = line + token;
      if (line && ctx.measureText(next).width > width) { lines.push(line.trim()); line = token.trimStart(); }
      else line = next;
    });
    if (line) lines.push(line.trim());
    if (lines.length > limit) { lines = lines.slice(0, limit); lines[limit - 1] = lines[limit - 1].replace(/\s+$/, '') + '…'; }
    return lines;
  }
  function shareCard(card, url) {
    return Promise.all([document.fonts ? document.fonts.ready : Promise.resolve(), loadImage(card.querySelector('.moment-pic') && card.querySelector('.moment-pic').href), qrImage(url)]).then(function (assets) {
      var body = card.querySelector('.moment-body'), quote = body.querySelector('.moment-quote');
      var paragraphs = Array.prototype.filter.call(body.children, function (el) { return el.tagName === 'P'; }).map(function (p) { return p.textContent.trim(); }).filter(Boolean).join('\n\n');
      var time = card.querySelector('.moment-when').textContent.trim(), place = card.querySelector('.moment-place');
      var qText = quote && quote.querySelector('p') ? quote.querySelector('p').textContent.trim() : '';
      var cite = quote && quote.querySelector('.moment-cite') ? quote.querySelector('.moment-cite').textContent.trim() : '';
      var canvas = document.createElement('canvas'), ctx = canvas.getContext('2d'), width = 1080, pad = 72, inner = width - pad * 2;
      ctx.font = '40px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
      var lines = wrap(ctx, paragraphs, inner, 16), quoteLines = [];
      if (qText) { ctx.font = '32px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'; quoteLines = wrap(ctx, qText, inner - 44, 5); }
      var image = assets[1], qr = assets[2], imageH = image ? Math.min(1080, inner * image.naturalHeight / image.naturalWidth) : 0;
      var headerH = 48, bodyH = lines.length * 68, quoteH = quoteLines.length ? quoteLines.length * 50 + (cite ? 72 : 0) + 48 : 0, footerH = 230;
      var height = Math.min(2400, pad + headerH + (lines.length ? 28 + bodyH : 0) + quoteH + (imageH ? 28 + imageH : 0) + 48 + footerH + pad);
      canvas.width = width; canvas.height = height; ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#7a8794'; ctx.font = '30px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
      ctx.fillText(time + (place ? ' · ' + place.textContent.trim() : ''), pad, pad + 32);
      var y = pad + headerH + (lines.length ? 28 : 0);
      ctx.fillStyle = '#2b3238'; ctx.font = '40px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
      lines.forEach(function (line) { if (y < height - pad - footerH) { ctx.fillText(line, pad, y + 40); y += 68; } });
      if (quoteLines.length && y < height - pad - footerH) {
        y += 20; ctx.fillStyle = '#1f8a9c'; ctx.fillRect(pad, y, 6, quoteLines.length * 50 + 8);
        ctx.fillStyle = '#2b3238'; ctx.font = '32px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
        quoteLines.forEach(function (line) { ctx.fillText(line, pad + 28, y + 34); y += 50; });
        if (cite) { ctx.fillStyle = '#7a8794'; ctx.textAlign = 'right'; ctx.fillText(cite.replace(/^——\s*/, ''), width - pad, y + 28); ctx.textAlign = 'left'; y += 44; }
        y += 28;
      }
      if (image && y + imageH < height - footerH) {
        ctx.save(); ctx.beginPath(); ctx.roundRect(pad, y, inner, imageH, 16); ctx.clip();
        var ratio = Math.max(inner / image.naturalWidth, imageH / image.naturalHeight), sw = inner / ratio, sh = imageH / ratio;
        ctx.drawImage(image, (image.naturalWidth - sw) / 2, (image.naturalHeight - sh) / 2, sw, sh, pad, y, inner, imageH); ctx.restore(); y += imageH + 28;
      }
      var footY = height - pad - footerH; ctx.strokeStyle = '#e4e9ee'; ctx.beginPath(); ctx.moveTo(pad, footY); ctx.lineTo(width - pad, footY); ctx.stroke();
      var brand = document.querySelector('.navbar-brand'), siteTitle = brand ? brand.textContent.trim() : document.title;
      ctx.fillStyle = '#2b3238'; ctx.font = '28px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'; ctx.fillText('随笔 · ' + siteTitle, pad, footY + 48);
      ctx.fillStyle = '#7a8794'; ctx.font = '22px -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'; ctx.fillText(new URL(url, location.href).host, pad, footY + 82);
      if (qr) ctx.drawImage(qr, width - pad - 200, footY + 16, 200, 200);
      return new Promise(function (resolve, reject) { canvas.toBlob(function (blob) { blob ? resolve(blob) : reject(new Error('PNG encoding failed')); }, 'image/png'); });
    });
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
  stream.addEventListener('click', function (e) {
    var share = e.target.closest('.moment-share');
    if (share && window.BlogShare) {
      e.stopPropagation();
      var card = share.closest('.moment');
      window.BlogShare.open(share, {
        url: share.getAttribute('data-url') || location.href.replace(/[#?].*$/, '') + '#' + card.id,
        title: share.getAttribute('data-title') || document.title,
        text: share.getAttribute('data-text') || '',
        id: card.id,
        card: function () { return shareCard(card, share.getAttribute('data-url') || location.href); },
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
    Array.prototype.forEach.call(stream.querySelectorAll('.moment-own'), function (n) { n.hidden = !mine; });
  }
  document.addEventListener('blog:viewer', function (e) { onViewer(e.detail); });
  if (window.BlogAnnotations && window.BlogAnnotations.viewer) onViewer(window.BlogAnnotations.viewer());
  if (list.getAttribute('data-entry-page') === 'true') {
    ensureToken().then(function (t) {
      return fetch('https://api.github.com/user', { headers: { Authorization: 'Bearer ' + t, Accept: 'application/vnd.github+json' } });
    }).then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('not signed in')); })
      .then(onViewer).catch(function () {});
  }
})();
