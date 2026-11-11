/*!
 * figure-export.js — the whole figure as an image file, whatever the viewport
 * (lazy-loaded by js/figures.js, not part of blog.min.js).
 *
 * A screenshot cannot hold a Mermaid diagram taller than the window, so:
 *   - png(fig)  -> Promise<Blob>  full figure at 2× its natural size.
 *       Mermaid: the *live* SVG (what the reader sees, KaTeX labels included)
 *       is cloned, the computed styles of the HTML labels inlined, the KaTeX
 *       faces it uses embedded as data: URLs, and drawn into a canvas.
 *       Browsers that taint a canvas for <foreignObject> (Safari) or any other
 *       failure fall back to WechatExport.mermaidToSvg — a pure-SVG re-render
 *       (htmlLabels:false, $$…$$ as Unicode).
 *       <img>: the original file at natural size (SVG files at 2×).
 *   - svg(fig)  -> Promise<Blob|null>  Mermaid: the pure-SVG re-render (no
 *       foreignObject, opens in Figma / Keynote / Inkscape); SVG <img>: the file.
 *   - original(fig) -> Promise<Blob|null>  raster <img>: the file as served.
 *   - openMenu(btn, fig)  the author's 「下载图片」 menu (js/figures.js only
 *       creates the button for the author; the menu checks again).
 * png() is also what a figure's passage share card uses (js/annotations.js
 * sharePassage) — the card carries title, link and QR; raw files are author-only.
 */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var SCALE = 2, MAX_SIDE = 16000, MAX_AREA = 64e6;
  var KATEX_FONTS = 'https://cdn.jsdelivr.net/npm/katex@0.18.7/dist/fonts/';
  // what decides how an HTML label looks; copied from getComputedStyle
  var PROPS = ['display', 'box-sizing', 'position', 'top', 'left', 'right', 'bottom', 'width', 'height', 'min-width', 'max-width',
    'margin-top', 'margin-right', 'margin-bottom', 'margin-left', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
    'border-top', 'border-right', 'border-bottom', 'border-left', 'border-radius',
    'font-family', 'font-size', 'font-weight', 'font-style', 'font-variant', 'line-height', 'letter-spacing', 'word-spacing',
    'color', 'background-color', 'text-align', 'text-decoration', 'text-transform', 'text-indent', 'white-space', 'word-break', 'overflow-wrap',
    'vertical-align', 'flex-direction', 'justify-content', 'align-items', 'gap', 'transform', 'transform-origin', 'opacity', 'overflow',
    'table-layout', 'border-collapse', 'fill', 'stroke'];

  function slug() {
    var p = location.pathname.replace(/\/+$/, '').split('/').pop() || 'figure';
    return p.replace(/\.html?$/i, '') || 'figure';
  }
  function captionOf(fig) {
    if (fig.matches('figure')) return fig.querySelector('figcaption');
    var n = fig.nextElementSibling;
    return n && n.classList.contains('post-figcaption') ? n : null;
  }
  function filename(fig, ext) {
    var cap = captionOf(fig), no = cap && cap.querySelector('.fig-no');
    var m = no && /(\d+)/.exec(no.textContent);
    return slug() + '-fig-' + (m ? m[1] : 'x') + '.' + ext;
  }
  function liveSvg(fig) { return fig.querySelector(':scope > .fig-media > svg, :scope > svg'); }
  function imgOf(fig) { return fig.matches('figure') ? fig.querySelector('.fig-media > img, img') : null; }
  function isSvgUrl(u) { return /\.svg(?:[?#]|$)/i.test(u || ''); }

  function loadScript(src, v) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src + (v ? '?v=' + v : '');
      s.onload = resolve; s.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(s);
    });
  }
  function wechat() {
    return window.WechatExport ? Promise.resolve(window.WechatExport) : loadScript('/js/wechat-export.js', window.BlogFigures && window.BlogFigures.wechatVersion).then(function () { return window.WechatExport; });
  }

  function fitScale(w, h, want) {
    return Math.max(0.1, Math.min(want, MAX_SIDE / w, MAX_SIDE / h, Math.sqrt(MAX_AREA / (w * h))));
  }
  function loadImage(src, cross) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      if (cross) img.crossOrigin = 'anonymous';
      img.decoding = 'sync';
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error('image load failed')); };
      img.src = src;
    });
  }
  function canvasBlob(canvas) {
    return new Promise(function (resolve, reject) {
      try {
        canvas.toBlob(function (b) { if (b) resolve(b); else reject(new Error('empty canvas')); }, 'image/png');
      } catch (e) { reject(e); } // SecurityError: tainted by <foreignObject>
    });
  }
  // draw `src` (w × h CSS px) at `scale` on white
  function rasterize(src, w, h, scale, cross) {
    return loadImage(src, cross).then(function (img) {
      var c = document.createElement('canvas');
      c.width = Math.round(w * scale); c.height = Math.round(h * scale);
      var ctx = c.getContext('2d');
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      return canvasBlob(c);
    });
  }
  function svgDataUrl(str) { return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(str); }
  function serialize(svg) {
    svg.setAttribute('xmlns', SVGNS);
    svg.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    return new XMLSerializer().serializeToString(svg);
  }
  function svgSize(svg) {
    var vb = svg.viewBox && svg.viewBox.baseVal;
    if (vb && vb.width && vb.height) return { w: vb.width, h: vb.height };
    var r = svg.getBoundingClientRect();
    return { w: r.width || 800, h: r.height || 400 };
  }

  // ------------------------------------------------ Mermaid, live rendering
  function katexFile(family, weight, style) {
    var bold = parseInt(weight, 10) >= 600 || weight === 'bold', italic = style === 'italic';
    return family + '-' + (bold && italic ? 'BoldItalic' : bold ? 'Bold' : italic ? 'Italic' : 'Regular');
  }
  function toDataUrl(blob) {
    return new Promise(function (resolve) {
      var r = new FileReader();
      r.onload = function () { resolve(r.result); };
      r.onerror = function () { resolve(null); };
      r.readAsDataURL(blob);
    });
  }
  // @font-face rules with the KaTeX faces the labels use (an SVG drawn as an
  // image cannot load web fonts, only data: URLs)
  function katexFaces(used) {
    var keys = Object.keys(used);
    return Promise.all(keys.map(function (k) {
      var f = used[k];
      return fetch(KATEX_FONTS + katexFile(f.family, f.weight, f.style) + '.woff2', { mode: 'cors' })
        .then(function (r) { if (!r.ok) throw new Error(r.status); return r.blob(); })
        .then(toDataUrl)
        .then(function (url) {
          return url ? '@font-face{font-family:' + f.family + ';src:url(' + url + ') format("woff2");font-weight:' +
            (parseInt(f.weight, 10) >= 600 ? 700 : 400) + ';font-style:' + (f.style === 'italic' ? 'italic' : 'normal') + ';}' : '';
        }, function () { return ''; });
    })).then(function (rules) { return rules.join(''); });
  }

  function liveMermaidSvg(fig) {
    var live = liveSvg(fig);
    if (!live) return Promise.reject(new Error('no svg'));
    var size = svgSize(live);
    var clone = live.cloneNode(true);
    var fonts = {};
    // parallel walk: inline the computed style of every HTML node inside a <foreignObject>
    (function walk(a, b, inHtml) {
      if (a.nodeType !== 1) return;
      var html = inHtml || a.namespaceURI !== SVGNS;
      if (html) {
        var cs = getComputedStyle(a), out = [];
        for (var i = 0; i < PROPS.length; i++) {
          var v = cs.getPropertyValue(PROPS[i]);
          if (v) out.push(PROPS[i] + ':' + v);
        }
        b.setAttribute('style', out.join(';'));
        var fam = (cs.fontFamily || '').split(',')[0].replace(/["']/g, '').trim();
        if (/^KaTeX_/.test(fam) && a.closest('.katex-html')) {
          var key = fam + '|' + cs.fontWeight + '|' + cs.fontStyle;
          fonts[key] = { family: fam, weight: cs.fontWeight, style: cs.fontStyle };
        }
      }
      for (var j = 0, ca = a.children, cb = b.children; j < ca.length && j < cb.length; j++) walk(ca[j], cb[j], html);
    })(live, clone, false);
    // reader chrome out: highlight marks unwrapped, markers / MathML dropped
    Array.prototype.forEach.call(clone.querySelectorAll('mark'), function (m) {
      while (m.firstChild) m.parentNode.insertBefore(m.firstChild, m);
      m.parentNode.removeChild(m);
    });
    Array.prototype.forEach.call(clone.querySelectorAll('.annotation-marker, .katex-mathml, .fig-tools'), function (n) { n.parentNode.removeChild(n); });
    Array.prototype.forEach.call(clone.querySelectorAll('.annotation-hl, .annotation-flash'), function (n) {
      n.classList.remove('annotation-hl', 'annotation-flash', 'is-multi', 'has-issue', 'is-resolved', 'is-new');
    });
    clone.removeAttribute('style');
    clone.setAttribute('width', size.w);
    clone.setAttribute('height', size.h);
    return katexFaces(fonts).then(function (css) {
      if (css) {
        var st = document.createElementNS(SVGNS, 'style');
        st.textContent = css;
        clone.insertBefore(st, clone.firstChild);
      }
      return { str: serialize(clone), w: size.w, h: size.h };
    });
  }

  function pureMermaidSvg(fig) {
    return wechat().then(function (wx) {
      if (!wx || !wx.mermaidToSvg) throw new Error('no re-render');
      return wx.mermaidToSvg(fig);
    }).then(function (svg) {
      var w = parseFloat(svg.getAttribute('width')) || 800, h = parseFloat(svg.getAttribute('height')) || 400;
      return { str: serialize(svg), w: w, h: h };
    });
  }

  function mermaidPng(fig) {
    return liveMermaidSvg(fig).then(function (s) {
      return rasterize(svgDataUrl(s.str), s.w, s.h, fitScale(s.w, s.h, SCALE));
    }).catch(function () {
      return pureMermaidSvg(fig).then(function (s) { return rasterize(svgDataUrl(s.str), s.w, s.h, fitScale(s.w, s.h, SCALE)); });
    });
  }

  // ------------------------------------------------------------- <img>
  function imgPng(img) {
    var src = img.currentSrc || img.src, svg = isSvgUrl(src);
    var cross = new URL(src, location.href).origin !== location.origin;
    return loadImage(src, cross).then(function (probe) {
      var w = probe.naturalWidth || img.naturalWidth || 800, h = probe.naturalHeight || img.naturalHeight || 600;
      return rasterize(src, w, h, fitScale(w, h, svg ? SCALE : 1), cross);
    });
  }
  function fetchBlob(src, type) {
    return fetch(src).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.blob(); })
      .then(function (b) { return type && b.type !== type ? new Blob([b], { type: type }) : b; });
  }

  // ------------------------------------------------------------- public
  function png(fig) {
    if (fig.matches('.mermaid')) return mermaidPng(fig);
    var img = imgOf(fig);
    return img ? imgPng(img) : Promise.reject(new Error('no figure'));
  }
  function svg(fig) {
    if (fig.matches('.mermaid')) {
      return pureMermaidSvg(fig).then(function (s) {
        return new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + s.str], { type: 'image/svg+xml' });
      });
    }
    var img = imgOf(fig), src = img && (img.currentSrc || img.src);
    return isSvgUrl(src) ? fetchBlob(src, 'image/svg+xml') : Promise.resolve(null);
  }
  function original(fig) {
    var img = imgOf(fig), src = img && (img.currentSrc || img.src);
    return src && !isSvgUrl(src) ? fetchBlob(src) : Promise.resolve(null);
  }
  function save(blob, name) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
  }
  function extOf(fig) {
    var img = imgOf(fig), m = img && /\.([a-z0-9]+)(?:[?#]|$)/i.exec(img.currentSrc || img.src);
    return m ? m[1].toLowerCase() : 'img';
  }

  // ------------------------------------------------------------- menu
  function isAuthor() { return !!(window.BlogFigures && window.BlogFigures.isAuthor && window.BlogFigures.isAuthor()); }
  var openState = null;
  function closeMenu() {
    if (!openState) return;
    openState.menu.remove();
    openState.btn.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', openState.outside, true);
    document.removeEventListener('keydown', openState.key);
    window.removeEventListener('scroll', closeMenu);
    openState = null;
  }
  function openMenu(btn, fig) {
    if (!isAuthor()) return;
    var again = openState && openState.btn === btn;
    closeMenu();
    if (again) return;
    var img = imgOf(fig), src = img && (img.currentSrc || img.src);
    var vector = fig.matches('.mermaid') || isSvgUrl(src);
    var items = [{ k: 'png', t: vector ? '下载 PNG（完整图，2×）' : '下载 PNG' }];
    if (vector) items.push({ k: 'svg', t: '下载 SVG' });
    else if (img) items.push({ k: 'orig', t: '下载原图（.' + extOf(fig) + '）' });
    var menu = document.createElement('div');
    menu.className = 'table-copy-menu fig-download-menu';
    menu.setAttribute('role', 'menu');
    menu.innerHTML = items.map(function (it) { return '<button type="button" role="menuitem" data-kind="' + it.k + '">' + it.t + '</button>'; }).join('');
    (btn.closest('.post-container') || document.body).appendChild(menu);
    var r = btn.getBoundingClientRect();
    menu.style.top = (r.bottom + 6) + 'px';
    menu.style.right = Math.max(8, window.innerWidth - r.right) + 'px';
    btn.setAttribute('aria-expanded', 'true');
    menu.addEventListener('click', function (e) {
      var item = e.target.closest('[data-kind]');
      if (!item) return;
      e.preventDefault(); e.stopPropagation();
      var kind = item.getAttribute('data-kind');
      closeMenu();
      run(btn, fig, kind);
    });
    openState = {
      btn: btn, menu: menu,
      outside: function (e) { if (!menu.contains(e.target) && !btn.contains(e.target)) closeMenu(); },
      key: function (e) { if (e.key === 'Escape') closeMenu(); }
    };
    document.addEventListener('click', openState.outside, true);
    document.addEventListener('keydown', openState.key);
    window.addEventListener('scroll', closeMenu, { passive: true });
  }
  function run(btn, fig, kind) {
    if (!isAuthor() || btn.disabled) return;
    var title = btn.title;
    btn.disabled = true; btn.title = '正在生成…';
    var job = kind === 'svg' ? svg(fig).then(function (b) { return b && [b, filename(fig, 'svg')]; })
      : kind === 'orig' ? original(fig).then(function (b) { return b && [b, filename(fig, extOf(fig))]; })
      : png(fig).then(function (b) { return [b, filename(fig, 'png')]; });
    job.then(function (r) {
      if (!r) throw new Error('这张图没有该格式');
      save(r[0], r[1]);
      btn.title = '已下载 ' + r[1];
    }).catch(function (e) {
      btn.title = '下载失败：' + (e && e.message || e);
    }).then(function () {
      btn.disabled = false;
      setTimeout(function () { btn.title = title; }, 2500);
    });
  }

  window.FigureExport = { png: png, svg: svg, original: original, filename: filename, openMenu: openMenu, closeMenu: closeMenu };
})();
