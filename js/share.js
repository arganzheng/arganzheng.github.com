/*!
 * share.js — the action bar above a post's comments, `.post-stats` strips
 * (阅读 · 点赞 · 评论 · 分享) in post headers and lists, and author-only controls
 * revealed when `js/annotations.js` identifies the signed-in viewer.
 *   - 点赞 (heart): anonymous per-post counter in the worker's D1 (POST /votes,
 *     dir 'up' | null), one per browser (localStorage), no login.
 *   - 分享 popover: system share sheet (Web Share API), Weibo / X / LinkedIn
 *     intent links, WeChat QR (js/vendor/qrcode.min.js, loaded on first use),
 *     copy link. One popover element, re-anchored to whichever button opened it.
 *     Every completed share is one POST /shares; Moments can additionally
 *     provide a PNG card generator and a long-press / download overlay.
 *   - stats: list pages get everything from one GET /stats?paths=…; on the post
 *     page 点赞 / 分享 come from GET /votes and views / comment count arrive from
 *     js/annotations.js as `blog:stats` events ({views} / {comments}).
 *   - author-only 「复制为公众号格式」 (lazy js/wechat-export.js).
 */
(function () {
  'use strict';

  var bars = Array.prototype.slice.call(document.querySelectorAll('.post-actions'));
  var strips = Array.prototype.slice.call(document.querySelectorAll('.post-stats'));
  var historySpans = Array.prototype.slice.call(document.querySelectorAll('.post-history'));
  if (!bars.length && !strips.length && !historySpans.length && !document.querySelector('.moment-share')) return; // 随笔 cards reuse the popover (BlogShare.open)
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
  // A `.post-stats` strip: four text labels with counts. Numbers arrive
  // piecemeal (votes / stats / annotations.js), so each strip keeps its own
  // partial state and is repainted with what is known so far.
  var STATS = [
    { k: 'views', label: '阅读' },
    { k: 'up', label: '点赞' },
    { k: 'comments', label: '评论' },
    { k: 'shares', label: '分享' }
  ];
  function paintStrip(el, patch) {
    var st = el._stats || (el._stats = {});
    Object.keys(patch).forEach(function (k) { if (patch[k] != null) st[k] = patch[k]; });
    el.innerHTML = STATS.filter(function (s) { return st[s.k] != null; }).map(function (s) {
      var n = st[s.k];
      return '<span class="ps ps-' + s.k + (n ? '' : ' is-zero') + '" title="' + s.label + '"><span class="ps-label">' + s.label + '</span> <b>' + fmt(n) + '</b></span>';
    }).join('');
  }
  function paintStrips(path, patch) {
    strips.forEach(function (el) { if (el.getAttribute('data-path') === path) paintStrip(el, patch); });
  }

  function render(bar, st) {
    var btn = bar.querySelector('.pa-like');
    bar.querySelector('.pa-like-n').textContent = st.up ? ' ' + fmt(st.up) : '';
    btn.classList.toggle('is-active', st.mine === 'up');
    btn.querySelector('.fa').className = 'fa ' + (st.mine === 'up' ? 'fa-heart' : 'fa-regular fa-heart');
    btn.title = st.mine === 'up' ? '取消点赞' : '给这篇文章点个赞（不用登录）';
    paintStrips(bar.getAttribute('data-path'), { up: st.up, shares: st.shares });
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
  function absoluteSitePath(siteUrl, path) {
    return siteUrl.replace(/\/+$/, '') + (path.charAt(0) === '/' ? path : '/' + path);
  }
  function stripFrontMatter(source) {
    var match = source.match(/^---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$)/);
    if (!match) throw new Error('无法识别文章 front matter');
    return source.slice(match[0].length).replace(/^\s+|\s+$/g, '');
  }
  function rewriteMarkdownPaths(markdown, siteUrl) {
    return markdown
      .replace(/(\]\()\/(?!\/)([^)\s]+)(?=[^)]*\))/g, function (_, prefix, path) {
        return prefix + absoluteSitePath(siteUrl, '/' + path);
      })
      .replace(/(\bsrc=)(["'])\/(?!\/)([^"']*)\2/gi, function (_, attr, quote, path) {
        return attr + quote + absoluteSitePath(siteUrl, '/' + path) + quote;
      });
  }
  function fetchMarkdownSource(bar) {
    var repo = bar.getAttribute('data-repo') || '';
    var sourcePath = bar.getAttribute('data-source') || '';
    if (!repo || !sourcePath) return Promise.reject(new Error('缺少文章源文件信息'));
    var encodedPath = sourcePath.split('/').map(encodeURIComponent).join('/');
    var apiUrl = 'https://api.github.com/repos/' + repo + '/contents/' + encodedPath + '?ref=master';
    var rawUrl = 'https://raw.githubusercontent.com/' + repo + '/master/' + encodedPath;
    return fetch(apiUrl, { headers: { Accept: 'application/vnd.github.raw' } }).then(function (response) {
      if (!response.ok) throw new Error('GitHub API HTTP ' + response.status);
      return response.text();
    }).catch(function () {
      return fetch(rawUrl).then(function (response) {
        if (!response.ok) throw new Error('GitHub raw HTTP ' + response.status);
        return response.text();
      });
    });
  }

  // ---------------------------------------------------------- share popover
  var pop = null, popFor = null, cardOverlay = null, cardObjectUrl = null;
  function ensurePop() {
    if (pop) return pop;
    pop = document.createElement('div');
    pop.className = 'pa-share-pop';
    pop.setAttribute('role', 'menu');
    pop.innerHTML =
      (navigator.share ? '<button type="button" class="pa-sp-item" data-k="native"><i class="fa fa-share-alt"></i>系统分享…</button>' : '') +
      '<a class="pa-sp-item" data-k="weibo" target="_blank" rel="noopener noreferrer"><i class="fa fa-brands fa-weibo"></i>微博</a>' +
      '<a class="pa-sp-item" data-k="x" target="_blank" rel="noopener noreferrer"><i class="fa fa-brands fa-twitter"></i>X</a>' +
      '<a class="pa-sp-item" data-k="linkedin" target="_blank" rel="noopener noreferrer"><i class="fa fa-brands fa-linkedin"></i>LinkedIn</a>' +
      '<button type="button" class="pa-sp-item" data-k="wechat"><i class="fa fa-brands fa-weixin"></i>微信扫一扫</button>' +
      '<button type="button" class="pa-sp-item" data-k="image" hidden><i class="fa fa-image"></i>生成图片</button>' +
      '<button type="button" class="pa-sp-item" data-k="copy"><i class="fa fa-link"></i>复制链接</button>' +
      '<div class="pa-sp-qr" hidden><span class="pa-sp-qr-img"></span><span class="pa-sp-qr-hint">微信扫一扫，分享给朋友或朋友圈</span></div>';
    document.body.appendChild(pop);
    pop.addEventListener('click', function (e) {
      var item = e.target.closest('.pa-sp-item');
      if (!item || !popFor) return;
      var k = item.getAttribute('data-k'), d = popFor, done = function () { d.onShared(k); };
      if (k === 'native') { navigator.share({ title: d.title, text: d.text || d.title, url: d.url }).then(done, function () { /* cancelled */ }); closePop(); }
      else if (k === 'wechat') { e.preventDefault(); showQR(d.url); done(); }
      else if (k === 'image') { closePop(); showCard(d); }
      else if (k === 'copy') { copyText(d.url).then(function () { d.toast('已复制链接'); done(); }, function () { d.toast('复制失败'); }); closePop(); }
      else { done(); closePop(); }
    });
    document.addEventListener('click', function (e) { if (pop && !pop.hidden && !pop.contains(e.target) && !(popFor && popFor.btn.contains(e.target))) closePop(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { closePop(); closeCard(); } });
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
  // Open the popover anchored to `btn`; `ctx.card` optionally returns a PNG Blob.
  // js/annotations.js reuses it for passage links (BlogShare.open).
  function openPop(ctx, btn) {
    var p = ensurePop(), d = ctx;
    if (popFor) closePop();
    popFor = ctx; ctx.btn = btn;
    ctx.onShared = ctx.onShared || function () {}; ctx.toast = ctx.toast || function () {};
    p.querySelector('[data-k="weibo"]').href = 'https://service.weibo.com/share/share.php?url=' + enc(d.url) + '&title=' + enc(d.title + (d.text ? ' — ' + d.text : ''));
    p.querySelector('[data-k="x"]').href = 'https://twitter.com/intent/tweet?url=' + enc(d.url) + '&text=' + enc(d.title + (d.text ? ' — ' + d.text : ''));
    p.querySelector('[data-k="linkedin"]').href = 'https://www.linkedin.com/sharing/share-offsite/?url=' + enc(d.url);
    p.querySelector('[data-k="image"]').hidden = typeof d.card !== 'function';
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
    if (popFor && popFor.btn) popFor.btn.setAttribute('aria-expanded', 'false');
    popFor = null;
  }
  function showCard(ctx) {
    ctx.card().then(function (blob) {
      closeCard();
      cardObjectUrl = URL.createObjectURL(blob);
      var name = 'moment-' + (ctx.id || 'share') + '.png';
      var file = new File([blob], name, { type: 'image/png' });
      cardOverlay = document.createElement('div');
      cardOverlay.className = 'pa-share-image';
      cardOverlay.setAttribute('role', 'dialog');
      cardOverlay.setAttribute('aria-modal', 'true');
      var canShare = false;
      try { canShare = !!(navigator.canShare && navigator.canShare({ files: [file] })); } catch (e) { /* unsupported */ }
      cardOverlay.innerHTML =
        '<div class="pa-share-image-panel">' +
          '<button type="button" class="pa-share-image-close" aria-label="关闭">×</button>' +
          '<img class="pa-share-image-preview" alt="随笔分享图片">' +
          '<p>长按图片保存，或</p><div class="pa-share-image-actions">' +
          '<a class="pa-share-image-download" download="' + name + '">下载</a>' +
          (canShare ? '<button type="button" class="pa-share-image-share">分享图片</button>' : '') +
          '</div></div>';
      cardOverlay.querySelector('img').src = cardObjectUrl;
      cardOverlay.querySelector('.pa-share-image-download').href = cardObjectUrl;
      cardOverlay.addEventListener('click', function (e) {
        if (e.target === cardOverlay || e.target.closest('.pa-share-image-close')) closeCard();
        if (e.target.closest('.pa-share-image-share')) navigator.share({ files: [file], title: ctx.title }).catch(function () {});
      });
      document.body.appendChild(cardOverlay);
      ctx.onShared('image');
    }).catch(function () { ctx.toast('图片生成失败'); });
  }
  function closeCard() {
    if (cardOverlay) cardOverlay.remove();
    cardOverlay = null;
    if (cardObjectUrl) URL.revokeObjectURL(cardObjectUrl);
    cardObjectUrl = null;
  }
  function isOpenFor(btn) { return !!(pop && !pop.hidden && popFor && popFor.btn === btn); }
  function showQR(url) {
    var ready = window.qrcode ? Promise.resolve() : loadScript('/js/vendor/qrcode.min.js');
    ready.then(function () {
      var qr = window.qrcode(0, 'M'); qr.addData(url); qr.make();
      var box = pop.querySelector('.pa-sp-qr');
      box.querySelector('.pa-sp-qr-img').innerHTML = qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
      box.hidden = false;
    }, function () { if (popFor) popFor.toast('二维码加载失败'); });
  }
  function shareCtx(bar) {
    var d = dataOf(bar);
    d.onShared = function () { countShare(bar); };
    d.toast = function (msg) { toast(bar, msg); };
    return d;
  }
  window.BlogShare = {
    open: function (btn, ctx) { if (isOpenFor(btn)) closePop(); else openPop(ctx, btn); },
    close: closePop,
    copyText: copyText
  };

  // ------------------------------------------------------------- 点赞
  // An anonymous counter kept by the worker (D1), like page views: no GitHub
  // login, one per browser remembered in localStorage. (Comment votes stay
  // GitHub reactions — those need an identity.)
  function apiBase(bar) { return (localStorage.getItem('annotationsApi') || bar.getAttribute('data-annotations-api') || '').replace(/\/$/, ''); }
  function myVote(path) { try { return localStorage.getItem('vote:' + path) || null; } catch (e) { return null; } }
  function remember(path, dir) { try { if (dir) localStorage.setItem('vote:' + path, dir); else localStorage.removeItem('vote:' + path); } catch (e) { /* ignore */ } }
  var local = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

  function vote(bar) {
    var st = bar._stats, api = apiBase(bar), path = bar.getAttribute('data-path');
    if (!st || bar._pending || !api) return;
    var prev = st.mine, next = prev ? null : 'up';
    var before = { up: st.up, mine: st.mine };
    st.up = Math.max(0, st.up + (next ? 1 : -1));
    st.mine = next;
    render(bar, st);
    if (local) { remember(path, next); return; } // previews don't count
    bar._pending = true;
    fetch(api + '/votes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path, dir: next, prev: prev }) })
      .then(function (r) { return r.json().then(function (d) { if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status)); return d; }); })
      .then(function (d) { st.up = d.up; remember(path, next); render(bar, st); })
      .catch(function (err) { st.up = before.up; st.mine = before.mine; render(bar, st); toast(bar, '操作失败：' + err.message, 3000); })
      .then(function () { bar._pending = false; });
  }
  function bindVotes(bar) {
    bar.querySelector('.pa-like').addEventListener('click', function () { vote(bar); });
  }
  // One completed share (any channel) = POST /shares. Optimistic; previews don't count.
  function countShare(bar) {
    var st = bar._stats, api = apiBase(bar), path = bar.getAttribute('data-path');
    if (!st || !api) return;
    st.shares = (st.shares || 0) + 1;
    render(bar, st);
    if (local) return;
    fetch(api + '/shares', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path }) })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d && typeof d.shares === 'number') { st.shares = d.shares; render(bar, st); } })
      .catch(function () { /* keep the optimistic number */ });
  }
  // Post page: one GET /votes for this article (点赞 + 分享 counts).
  function loadVotes(bar) {
    var api = apiBase(bar), path = bar.getAttribute('data-path');
    if (!api) return;
    fetch(api + '/votes?path=' + enc(path)).then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)); }).then(function (d) {
      bar._stats = { up: d.up || 0, shares: d.shares || 0, mine: myVote(path) };
      render(bar, bar._stats);
    }).catch(function () { bar._stats = { up: 0, shares: 0, mine: myVote(path) }; render(bar, bar._stats); });
  }
  // List pages: one GET /stats for all the strips (views, 点赞, comments, shares).
  function loadStats(list) {
    var api = apiBase(list[0]);
    if (!api) return;
    var paths = [];
    list.forEach(function (el) { var p = el.getAttribute('data-path'); if (paths.indexOf(p) < 0) paths.push(p); });
    var chunks = [];
    while (paths.length) chunks.push(paths.splice(0, 20));
    chunks.forEach(function (chunk) {
      fetch(api + '/stats?paths=' + enc(chunk.join(','))).then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)); }).then(function (data) {
        chunk.forEach(function (p) {
          var it = data.items && data.items[p];
          if (it) paintStrips(p, { views: it.views || 0, up: it.up || 0, comments: it.comments || 0, shares: it.shares || 0 });
        });
      }).catch(function () { /* stays blank */ });
    });
  }

  // ------------------------------------------------------------------- wire
  bars.forEach(function (bar) {
    var shareBtn = bar.querySelector('.pa-share');
    shareBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (isOpenFor(shareBtn)) closePop(); else openPop(shareCtx(bar), shareBtn);
    });
    bindVotes(bar);
    loadVotes(bar);
  });
  // Post page: the header strip gets views / comment count from js/annotations.js;
  // list pages: one /stats round trip for every strip.
  if (bars.length) {
    document.addEventListener('blog:stats', function (e) {
      if (!e.detail) return;
      // a passage share (annotations.js) is an article share too
      if (typeof e.detail.shares === 'number' && bars[0]._stats) bars[0]._stats.shares = e.detail.shares;
      paintStrips(bars[0].getAttribute('data-path'), e.detail);
    });
  } else if (strips.length) {
    loadStats(strips);
  }

  // ------------------------------------------------- author: export / edit
  var exportBtn = document.querySelector('.post-actions .pa-export');
  var printBtn = document.querySelector('.post-actions .pa-print');
  var markdownBtn = document.querySelector('.post-actions .pa-markdown');
  var editBtn = document.querySelector('.post-actions .pa-edit');
  if (exportBtn || printBtn || markdownBtn || editBtn || historySpans.length) {
    var bar = document.querySelector('.post-actions[data-author]');
    var author = (bar && bar.getAttribute('data-author')) ||
      (historySpans[0] && historySpans[0].getAttribute('data-author')) || '';
    var isAuthor = false;
    var onViewer = function (v) {
      isAuthor = !!(v && author && v.login === author);
      if (exportBtn) exportBtn.hidden = !isAuthor;
      if (printBtn) printBtn.hidden = !isAuthor;
      if (markdownBtn) markdownBtn.hidden = !isAuthor;
      if (editBtn) editBtn.hidden = !isAuthor;
      historySpans.forEach(function (span) { span.hidden = !isAuthor; });
    };
    document.addEventListener('blog:viewer', function (e) { onViewer(e.detail); });
    if (window.BlogAnnotations && window.BlogAnnotations.viewer) onViewer(window.BlogAnnotations.viewer());
    if (exportBtn) {
      exportBtn.addEventListener('click', function () {
        if (!isAuthor) return;
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
    if (printBtn) {
      printBtn.addEventListener('click', function () {
        if (!isAuthor || printBtn.disabled) return;
        var root = document.documentElement;
        var hadTheme = root.hasAttribute('data-theme');
        var oldTheme = root.getAttribute('data-theme');
        var hadPrintClass = root.classList.contains('print-export');
        var restored = false;
        function restorePrintState() {
          if (restored) return;
          restored = true;
          window.removeEventListener('afterprint', restorePrintState);
          if (hadTheme) root.setAttribute('data-theme', oldTheme);
          else root.removeAttribute('data-theme');
          if (!hadPrintClass) root.classList.remove('print-export');
          printBtn.disabled = false;
        }
        printBtn.disabled = true;
        window.addEventListener('afterprint', restorePrintState);
        root.setAttribute('data-theme', 'light');
        root.classList.add('print-export');
        try { window.print(); }
        catch (err) {
          restorePrintState();
          toast(bar, '打印失败：' + (err && err.message || err), 4000);
        }
      });
    }
    if (markdownBtn) {
      markdownBtn.addEventListener('click', function () {
        if (!isAuthor || markdownBtn.disabled) return;
        markdownBtn.disabled = true;
        var d = dataOf(bar);
        fetchMarkdownSource(bar).then(function (source) {
          var body = stripFrontMatter(source);
          var siteUrl = bar.getAttribute('data-site-url') || new URL(d.url).origin;
          body = rewriteMarkdownPaths(body, siteUrl);
          var subtitle = bar.getAttribute('data-subtitle') || '';
          var markdown = '# ' + d.title + '\n\n' +
            (subtitle ? '> ' + subtitle + '\n\n' : '') +
            body + '\n\n原文：' + d.url + '\n';
          return copyText(markdown).then(function () {
            toast(bar, '已复制 Markdown（' + markdown.length + ' 字符）', 4000);
          });
        }).catch(function (err) {
          toast(bar, '失败：' + (err && err.message || err), 4000);
        }).then(function () { markdownBtn.disabled = false; });
      });
    }
  }

})();
