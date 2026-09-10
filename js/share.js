/*!
 * share.js — the share bar under each article (_includes/share.html).
 * System share sheet (Web Share API), Weibo / X / LinkedIn intent links,
 * WeChat QR code (js/vendor/qrcode.min.js, loaded on first click), copy link,
 * and — for the blog author only — 「复制为公众号格式」 which lazy-loads
 * js/wechat-export.js. No third-party script, no counters.
 */
(function () {
  'use strict';

  var bar = document.querySelector('.post-share');
  if (!bar) return;

  var url = bar.getAttribute('data-url') || location.href.replace(/[#?].*$/, '');
  var title = bar.getAttribute('data-title') || document.title;
  var text = bar.getAttribute('data-text') || '';
  var author = bar.getAttribute('data-author') || '';
  var version = (document.currentScript && (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';
  var enc = encodeURIComponent;

  // ------------------------------------------------------------ intent links
  var links = {
    weibo: 'https://service.weibo.com/share/share.php?url=' + enc(url) + '&title=' + enc(title + (text ? ' — ' + text : '')),
    x: 'https://twitter.com/intent/tweet?url=' + enc(url) + '&text=' + enc(title),
    linkedin: 'https://www.linkedin.com/sharing/share-offsite/?url=' + enc(url)
  };
  Object.keys(links).forEach(function (k) {
    var a = bar.querySelector('[data-share="' + k + '"]');
    if (a) a.href = links[k];
  });

  // --------------------------------------------------------------- native
  var nativeBtn = bar.querySelector('[data-share="native"]');
  if (nativeBtn && navigator.share) {
    nativeBtn.hidden = false;
    nativeBtn.addEventListener('click', function () {
      navigator.share({ title: title, text: text || title, url: url }).catch(function () { /* user cancelled */ });
    });
  }

  // ----------------------------------------------------------------- toast
  function toast(btn, msg, ms) {
    var t = btn.querySelector('.post-share-toast');
    if (!t) return;
    t.textContent = msg;
    btn.classList.add('is-toast');
    clearTimeout(t._timer);
    t._timer = setTimeout(function () { btn.classList.remove('is-toast'); t.textContent = ''; }, ms || 1800);
  }

  // ------------------------------------------------------------- copy link
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
  var copyBtn = bar.querySelector('[data-share="copy"]');
  if (copyBtn) copyBtn.addEventListener('click', function () {
    copyText(url).then(function () { toast(copyBtn, '已复制链接'); }, function () { toast(copyBtn, '复制失败'); });
  });

  // ------------------------------------------------------------- wechat QR
  var wechatBtn = bar.querySelector('[data-share="wechat"]');
  var qrBox = bar.querySelector('.post-share-qr');
  var qrDrawn = false;
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src + (version ? '?v=' + version : '');
      s.onload = resolve; s.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(s);
    });
  }
  function drawQR() {
    if (qrDrawn) return Promise.resolve();
    var ready = window.qrcode ? Promise.resolve() : loadScript('/js/vendor/qrcode.min.js');
    return ready.then(function () {
      var qr = window.qrcode(0, 'M');
      qr.addData(url); qr.make();
      // cellSize 4 => ~120-150 px, margin 2 cells; the SVG scales with CSS.
      qrBox.querySelector('.post-share-qr-img').innerHTML = qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
      qrDrawn = true;
    });
  }
  function closeQR() {
    if (!qrBox || qrBox.hidden) return;
    qrBox.hidden = true; wechatBtn.setAttribute('aria-expanded', 'false');
  }
  if (wechatBtn && qrBox) {
    wechatBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      if (!qrBox.hidden) return closeQR();
      drawQR().then(function () {
        qrBox.hidden = false; wechatBtn.setAttribute('aria-expanded', 'true');
      }, function () { toast(copyBtn, '二维码加载失败'); });
    });
    document.addEventListener('click', function (e) { if (!qrBox.contains(e.target)) closeQR(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeQR(); });
  }

  // ------------------------------------------------- author: WeChat export
  var exportBtn = bar.querySelector('[data-share="export"]');
  function onViewer(v) {
    if (!exportBtn) return;
    exportBtn.hidden = !(v && author && v.login === author);
  }
  if (exportBtn) {
    document.addEventListener('blog:viewer', function (e) { onViewer(e.detail); });
    if (window.BlogAnnotations && window.BlogAnnotations.viewer) onViewer(window.BlogAnnotations.viewer());
    exportBtn.addEventListener('click', function () {
      exportBtn.disabled = true;
      var ready = window.WechatExport ? Promise.resolve() : loadScript('/js/wechat-export.js');
      ready.then(function () {
        return window.WechatExport.copy({ url: url, title: title });
      }).then(function (info) {
        toast(exportBtn, '已复制（' + info.images + ' 图 · ' + info.refs + ' 条参考链接），到公众号 / 知乎编辑器里粘贴', 4000);
      }, function (err) {
        toast(exportBtn, '失败：' + (err && err.message || err), 4000);
      }).then(function () { exportBtn.disabled = false; });
    });
  }
})();
