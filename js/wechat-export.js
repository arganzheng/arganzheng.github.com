/*!
 * wechat-export.js — 「复制为公众号格式」 (author only; lazy-loaded by js/share.js).
 *
 * Turns the rendered article (.post-container) into self-contained HTML with
 * inline styles and puts it on the clipboard as text/html (+ text/plain), so it
 * can be pasted straight into the WeChat Official Account editor or Zhihu.
 * What has to change on the way (WeChat strips classes, external links, SVG and
 * <pre> styling; it re-uploads images it can fetch):
 *   - non-body chrome removed (series nav/TOC, comments, highlights, copy buttons…)
 *   - every element gets style="" from STYLES; Rouge tokens keep their colour
 *     via getComputedStyle on the live DOM; all class/id attributes dropped
 *   - external links -> text + [n], listed under 「参考链接」 at the end;
 *     Markdown footnotes / inline tips -> the same numbered list
 *   - KaTeX -> <img> of the TeX (codecogs for WeChat, zhihu.com/equation for Zhihu)
 *   - Mermaid SVG -> PNG and .webp images -> JPEG data URLs via canvas (WeChat
 *     takes neither SVG nor WebP); other images -> absolute URLs
 * window.WechatExport.copy({url, title, target: 'wechat'|'zhihu'}) -> Promise<{images, refs}>
 */
(function () {
  'use strict';

  var ACCENT = '#0085a1';
  var FONT = '-apple-system, BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif';
  var MONO = 'Menlo, Consolas, "Courier New", monospace';

  // Tag -> inline style. Keep it plain: the WeChat editor drops anything fancy.
  var STYLES = {
    p: 'margin:0 0 1.2em;font-size:16px;line-height:1.8;color:#333;letter-spacing:0.3px;text-align:justify;',
    h1: 'margin:1.6em 0 0.8em;font-size:22px;font-weight:700;line-height:1.4;color:#222;',
    h2: 'margin:1.6em 0 0.8em;padding-left:10px;border-left:4px solid ' + ACCENT + ';font-size:20px;font-weight:700;line-height:1.4;color:#222;',
    h3: 'margin:1.4em 0 0.7em;font-size:18px;font-weight:700;line-height:1.4;color:#222;',
    h4: 'margin:1.2em 0 0.6em;font-size:16px;font-weight:700;color:#333;',
    h5: 'margin:1em 0 0.5em;font-size:16px;font-weight:700;color:#333;',
    h6: 'margin:1em 0 0.5em;font-size:15px;font-weight:700;color:#555;',
    blockquote: 'margin:0 0 1.2em;padding:0.6em 1em;border-left:4px solid #d0d7de;background:#f6f8fa;color:#57606a;font-size:15px;line-height:1.75;',
    ul: 'margin:0 0 1.2em;padding-left:1.6em;list-style:disc;',
    ol: 'margin:0 0 1.2em;padding-left:1.6em;list-style:decimal;',
    li: 'margin:0.3em 0;font-size:16px;line-height:1.8;color:#333;',
    a: 'color:' + ACCENT + ';text-decoration:none;',
    strong: 'font-weight:700;color:#222;',
    em: 'font-style:italic;',
    del: 'text-decoration:line-through;color:#999;',
    hr: 'margin:1.6em 0;border:0;border-top:1px solid #e1e4e8;',
    img: 'display:block;max-width:100%;margin:0.6em auto 1.2em;border-radius:4px;',
    table: 'display:table;width:100%;margin:0 0 1.2em;border-collapse:collapse;font-size:14px;line-height:1.6;',
    th: 'padding:8px 10px;border:1px solid #d0d7de;background:#f6f8fa;font-weight:700;text-align:left;',
    td: 'padding:8px 10px;border:1px solid #d0d7de;text-align:left;vertical-align:top;',
    figure: 'margin:0 0 1.2em;',
    figcaption: 'margin-top:-0.6em;font-size:13px;color:#888;text-align:center;',
    sup: 'font-size:11px;color:' + ACCENT + ';vertical-align:super;line-height:0;',
    sub: 'font-size:11px;vertical-align:sub;line-height:0;',
    kbd: 'padding:1px 5px;border:1px solid #ccc;border-radius:3px;background:#f7f7f7;font-family:' + MONO + ';font-size:13px;'
  };
  var INLINE_CODE = 'padding:2px 5px;border-radius:3px;background:rgba(27,31,35,0.06);color:#c7254e;font-family:' + MONO + ';font-size:14px;';
  var PRE_WRAP = 'margin:0 0 1.2em;padding:12px 14px;border-radius:6px;background:#f6f8fa;border:1px solid #e1e4e8;overflow-x:auto;';
  var PRE_CODE = 'display:block;white-space:pre;word-wrap:normal;font-family:' + MONO + ';font-size:13px;line-height:1.6;color:#24292f;-webkit-text-size-adjust:none;';
  var REMOVE = '.series-nav, .series-context, .series-toc, .pager, .related-posts, .comment, .annotation-comments, .annotation-panel, .annotation-marker, .annotation-toolbar, .code-copy, .inline-popover-card, .post-share, .catalog-toggle, script, style, noscript, .reversefootnote, .mermaidTooltip, hr[style*="hidden"]';

  function el(tag, style, html) {
    var e = document.createElement(tag);
    if (style) e.setAttribute('style', style);
    if (html != null) e.innerHTML = html;
    return e;
  }
  var base = location.href;
  function abs(u) { try { return new URL(u, base).href; } catch (e) { return u; } }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // ---------------------------------------------------------------- images
  function svgToPng(svg) {
    return new Promise(function (resolve, reject) {
      var box = svg.getBoundingClientRect();
      var w = Math.max(1, Math.round(parseFloat(svg.getAttribute('width')) || box.width || 800));
      var h = Math.max(1, Math.round(parseFloat(svg.getAttribute('height')) || box.height || 400));
      var clone = svg.cloneNode(true);
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      clone.setAttribute('width', w); clone.setAttribute('height', h);
      if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      // Mermaid puts its CSS in a <style> inside the svg, so the standalone copy keeps its look.
      var blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml;charset=utf-8' });
      var src = URL.createObjectURL(blob);
      var img = new Image();
      img.onload = function () {
        // 2x for crisp text, but cap the bitmap at ~1600 px wide so a diagram-heavy
        // article stays a few MB on the clipboard.
        var scale = Math.min(2, 1600 / w), canvas = document.createElement('canvas');
        canvas.width = w * scale; canvas.height = h * scale;
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(src);
        try { resolve(canvas.toDataURL('image/png')); } catch (e) { reject(e); }
      };
      img.onerror = function () { URL.revokeObjectURL(src); reject(new Error('svg render failed')); };
      img.src = src;
    });
  }
  // Mermaid draws labels as <foreignObject> (htmlLabels), which taints a canvas
  // and makes toDataURL throw. Re-render the source with pure-SVG labels for the
  // export, then restore the page's config.
  var mermaidSeq = 0;
  function mermaidToPng(container) {
    var src = container.getAttribute('data-mermaid-source');
    var live = container.querySelector('svg');
    if (!src || !window.mermaid || !mermaid.render) return live ? svgToPng(live) : Promise.reject(new Error('no mermaid'));
    var cfg = mermaid.mermaidAPI && mermaid.mermaidAPI.getConfig ? JSON.parse(JSON.stringify(mermaid.mermaidAPI.getConfig())) : null;
    mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose', htmlLabels: false, flowchart: { htmlLabels: false, useMaxWidth: false, curve: 'basis' }, sequence: { useMaxWidth: false }, gantt: { useMaxWidth: false }, mindmap: { useMaxWidth: false } });
    var host = document.createElement('div');
    host.style.cssText = 'position:fixed;left:-10000px;top:0;';
    document.body.appendChild(host);
    return Promise.resolve(mermaid.render('wx-export-' + (++mermaidSeq), src)).then(function (r) {
      host.innerHTML = typeof r === 'string' ? r : r.svg;
      var svg = host.querySelector('svg');
      var live2 = live ? live.getBoundingClientRect() : null;
      // useMaxWidth:false gives the natural size; keep the aspect ratio but never below the on-page width
      var w = parseFloat(svg.getAttribute('width')) || (live2 && live2.width) || 800;
      var h = parseFloat(svg.getAttribute('height')) || (live2 && live2.height) || 400;
      if (live2 && w < live2.width) { h = h * live2.width / w; w = live2.width; }
      svg.setAttribute('width', w); svg.setAttribute('height', h);
      svg.style.maxWidth = 'none';
      return svgToPng(svg);
    }).then(function (png) {
      host.remove(); if (cfg) mermaid.initialize(cfg); return png;
    }, function (e) {
      host.remove(); if (cfg) mermaid.initialize(cfg); throw e;
    });
  }
  // WebP -> JPEG (q 0.9, <= 1280 px wide, white behind any transparency): PNG
  // would be ~1 MB per screenshot and a long post has dozens of them.
  function imgToJpeg(url) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = function () {
        var scale = Math.min(1, 1280 / img.naturalWidth);
        var canvas = document.createElement('canvas');
        canvas.width = Math.round(img.naturalWidth * scale); canvas.height = Math.round(img.naturalHeight * scale);
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        try { resolve(canvas.toDataURL('image/jpeg', 0.9)); } catch (e) { reject(e); }
      };
      img.onerror = function () { reject(new Error('image load failed: ' + url)); };
      img.src = url;
    });
  }

  // ------------------------------------------------------------- transform
  function build(opts) {
    var target = opts.target || 'wechat';
    // Relative links resolve against the canonical URL, not localhost when previewing.
    base = opts.url ? new URL(location.pathname + location.search, opts.url).href : location.href;
    var container = document.querySelector('.post-container');
    if (!container) throw new Error('no .post-container');

    // Rouge tokens: remember the live colours before cloning (same tree order in the clone).
    var origTokens = container.querySelectorAll('pre code span');
    var tokenStyles = [];
    for (var i = 0; i < origTokens.length; i++) {
      var cs = getComputedStyle(origTokens[i]);
      var st = 'color:' + cs.color + ';';
      if (cs.fontWeight === 'bold' || +cs.fontWeight >= 600) st += 'font-weight:700;';
      if (cs.fontStyle === 'italic') st += 'font-style:italic;';
      tokenStyles.push(st);
    }
    // Mermaid / webp / KaTeX are replaced by placeholders now and resolved async below.
    var jobs = [];
    var refs = []; // [{text, href}] -> 「参考链接」 list
    var notes = {}; // footnote id -> number in refs

    var root = container.cloneNode(true);
    var cloneTokens = root.querySelectorAll('pre code span');
    for (i = 0; i < cloneTokens.length; i++) cloneTokens[i].setAttribute('style', tokenStyles[i] || '');

    // Footnote bodies stay in the tree during the transforms (so their links /
    // formulas get converted too) and are moved into the tail at the end.
    var fnBodies = {}, fnContainer = root.querySelector('.footnotes');
    root.querySelectorAll('.footnotes li[id]').forEach(function (li) { fnBodies[li.id] = li; });
    root.querySelectorAll(REMOVE).forEach(function (n) { n.remove(); });
    // Highlights from reader comments: keep the text, drop the mark.
    root.querySelectorAll('mark.annotation-hl').forEach(function (m) { m.replaceWith.apply(m, Array.prototype.slice.call(m.childNodes)); });

    function addRef(text, href) {
      for (var k = 0; k < refs.length; k++) if (refs[k].href === href && href) return k + 1;
      refs.push({ text: text, href: href });
      return refs.length;
    }
    function supNum(n) { return el('sup', STYLES.sup, '[' + n + ']'); }

    // Mermaid diagrams -> PNG.
    var liveMermaid = container.querySelectorAll('.mermaid');
    var mermaidChain = Promise.resolve(); // one at a time: mermaid.render shares global config
    root.querySelectorAll('.mermaid').forEach(function (m, idx) {
      var ph = el('img', STYLES.img); ph.alt = 'diagram';
      m.replaceWith(ph);
      mermaidChain = mermaidChain.then(function () {
        return liveMermaid[idx] ? mermaidToPng(liveMermaid[idx]) : Promise.reject(new Error('no diagram'));
      }).then(function (d) { ph.src = d; }, function () { ph.replaceWith(el('p', STYLES.p, '（图表：见原文）')); });
    });
    jobs.push(mermaidChain);

    // KaTeX -> image of the TeX source.
    root.querySelectorAll('.katex-display, .katex').forEach(function (k) {
      if (!root.contains(k)) return; // nested .katex inside an already-handled .katex-display
      var ann = k.querySelector('annotation[encoding="application/x-tex"]');
      var tex = ann ? ann.textContent.trim() : k.textContent.trim();
      var display = k.classList.contains('katex-display');
      var img = document.createElement('img');
      img.alt = tex;
      if (target === 'zhihu') {
        img.setAttribute('eeimg', '1');
        img.src = 'https://www.zhihu.com/equation?tex=' + encodeURIComponent(tex + (display ? '\\\\' : ''));
      } else {
        img.src = 'https://latex.codecogs.com/png.image?' + encodeURIComponent('\\dpi{300} ' + tex);
      }
      img.setAttribute('style', display ? 'display:block;max-width:100%;height:auto;margin:0.8em auto;' : 'display:inline;height:1.2em;vertical-align:middle;margin:0 2px;');
      var wrap = display ? el('p', 'text-align:center;margin:0 0 1.2em;') : null;
      if (wrap) { wrap.appendChild(img); k.replaceWith(wrap); } else k.replaceWith(img);
    });

    // Inline tips -> footnote.
    root.querySelectorAll('.inline-tip').forEach(function (t) {
      var n = addRef(t.textContent.trim() + '：' + (t.getAttribute('data-tip') || t.getAttribute('title') || '').replace(/^tip:\s*/i, ''), '');
      var span = el('span', 'border-bottom:1px dashed #999;', esc(t.textContent));
      t.replaceWith(span, supNum(n));
    });
    // Markdown footnotes -> footnote.
    root.querySelectorAll('sup[id^="fnref:"]').forEach(function (s) {
      var a = s.querySelector('a[href^="#fn:"]');
      var id = a ? a.getAttribute('href').slice(1) : null;
      var body = id && fnBodies[id];
      var n;
      if (body) {
        if (!notes[id]) { refs.push({ node: body }); notes[id] = refs.length; }
        n = notes[id];
      } else n = addRef(s.textContent.trim(), '');
      s.replaceWith(supNum(n));
    });
    // Links: WeChat keeps only mp.weixin.qq.com links, so everything else becomes text + [n].
    root.querySelectorAll('a[href]').forEach(function (a) {
      var href = a.getAttribute('href') || '';
      if (/^#/.test(href)) { a.replaceWith.apply(a, Array.prototype.slice.call(a.childNodes)); return; } // in-page anchors
      var full = abs(href);
      var text = a.textContent.trim();
      if (a.querySelector('img')) { a.replaceWith.apply(a, Array.prototype.slice.call(a.childNodes)); return; } // linked images: keep the image
      if (target === 'zhihu' || /^https?:\/\/mp\.weixin\.qq\.com\//.test(full)) {
        a.setAttribute('href', full); a.setAttribute('style', STYLES.a);
        a.removeAttribute('target'); a.removeAttribute('rel');
        return;
      }
      var n = addRef(text, full);
      var span = el('span', 'color:' + ACCENT + ';', a.innerHTML);
      a.replaceWith(span, supNum(n));
    });

    // Code blocks: Rouge's div.highlighter-rouge > div.highlight > pre > code.
    root.querySelectorAll('div.highlighter-rouge, pre').forEach(function (block) {
      if (!block.parentNode || block.closest('section[data-code]')) return;
      var code = block.querySelector('code') || block;
      var section = el('section', PRE_WRAP);
      section.setAttribute('data-code', '1');
      var c = el('code', PRE_CODE);
      c.innerHTML = code.innerHTML;
      section.appendChild(c);
      block.replaceWith(section);
    });
    root.querySelectorAll('code').forEach(function (c) {
      if (c.closest('section[data-code]')) return;
      c.setAttribute('style', INLINE_CODE);
    });

    // Images: absolute URLs; webp -> JPEG data URL (WeChat's uploader rejects webp).
    var imgCount = 0;
    root.querySelectorAll('img').forEach(function (img) {
      imgCount++;
      if (!img.getAttribute('src')) return; // Mermaid placeholder, filled in by its job
      var attr = img.getAttribute('src');
      var src = abs(attr); // canonical (https://arganzheng.life/...) for the pasted HTML
      var local = new URL(attr, location.href); // what this page can actually fetch (localhost when previewing)
      img.removeAttribute('loading'); img.removeAttribute('decoding'); img.removeAttribute('srcset'); img.removeAttribute('sizes');
      if (/\.webp(\?|$)/i.test(src) && local.origin === location.origin) {
        jobs.push(imgToJpeg(local.href).then(function (d) { img.src = d; }, function () { img.src = src; }));
      } else img.src = src;
      if (!img.getAttribute('style')) img.setAttribute('style', STYLES.img);
    });

    // Generic styles, then strip classes/ids/data-* everywhere.
    root.querySelectorAll('*').forEach(function (n) {
      var tag = n.tagName.toLowerCase();
      if (STYLES[tag]) {
        var own = n.getAttribute('style') || '';
        // Kramdown's cells carry style="text-align: …"; keep that, add ours. Nodes we styled already (data-code, katex img) are left alone.
        if (!own) n.setAttribute('style', STYLES[tag]);
        else if (/^\s*text-align:[^;]*;?\s*$/.test(own)) n.setAttribute('style', STYLES[tag].replace(/text-align:[^;]*;/, '') + own.trim().replace(/;?$/, ';'));
      }
      if (tag === 'th' || tag === 'td') { n.removeAttribute('align'); }
      Array.prototype.slice.call(n.attributes).forEach(function (at) {
        if (at.name === 'class' || at.name === 'id' || /^data-/.test(at.name) || /^aria-/.test(at.name) || at.name === 'role' || at.name === 'title' || at.name === 'target' || at.name === 'rel') n.removeAttribute(at.name);
      });
    });
    // Tables must scroll on phones.
    root.querySelectorAll('table').forEach(function (t) {
      var wrap = el('section', 'overflow-x:auto;margin:0 0 1.2em;');
      t.replaceWith(wrap); wrap.appendChild(t);
      t.setAttribute('style', STYLES.table.replace('margin:0 0 1.2em;', 'margin:0;'));
    });

    // Tail: 参考链接 / 脚注, then the canonical link.
    var tail = el('section', '');
    if (refs.length) {
      tail.appendChild(el('h3', STYLES.h3, '参考与脚注'));
      var ol = el('ol', STYLES.ol.replace('1.6em', '1.8em'));
      refs.forEach(function (r) {
        var li = el('li', STYLES.li.replace('16px', '14px') + 'color:#555;word-break:break-all;');
        if (r.node) {
          r.node.querySelectorAll('p').forEach(function (n) { n.setAttribute('style', 'margin:0;font-size:14px;line-height:1.7;color:#555;'); });
          li.innerHTML = r.node.innerHTML;
        } else li.innerHTML = esc(r.text) + (r.href ? (r.text ? '：' : '') + '<span style="color:#888;">' + esc(r.href) + '</span>' : '');
        ol.appendChild(li);
      });
      tail.appendChild(ol);
    }
    tail.appendChild(el('p', STYLES.p.replace('#333', '#888') + 'font-size:14px;', '原文：<span style="color:' + ACCENT + ';">' + esc(opts.url) + '</span>（原文有可点击的目录、脚注浮窗和划线评论）'));
    if (fnContainer) fnContainer.remove();
    root.appendChild(tail);

    return Promise.all(jobs).then(function () {
      var wrapper = el('section', 'font-family:' + FONT + ';font-size:16px;color:#333;');
      wrapper.innerHTML = root.innerHTML.replace(/\n{3,}/g, '\n\n');
      return { html: wrapper.outerHTML, text: wrapper.innerText || wrapper.textContent, images: imgCount, refs: refs.length };
    });
  }

  // -------------------------------------------------------------- clipboard
  function writeClipboard(html, text) {
    if (navigator.clipboard && window.ClipboardItem && window.isSecureContext) {
      var item = new ClipboardItem({
        'text/html': new Blob([html], { type: 'text/html' }),
        'text/plain': new Blob([text], { type: 'text/plain' })
      });
      return navigator.clipboard.write([item]).catch(function () { return legacy(html); });
    }
    return legacy(html);
  }
  function legacy(html) {
    var host = el('div', 'position:fixed;left:-9999px;top:0;opacity:0;', html);
    host.setAttribute('contenteditable', 'true');
    document.body.appendChild(host);
    var range = document.createRange(); range.selectNodeContents(host);
    var sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
    var ok = document.execCommand('copy');
    sel.removeAllRanges(); document.body.removeChild(host);
    return ok ? Promise.resolve() : Promise.reject(new Error('copy failed'));
  }

  window.WechatExport = {
    build: build,
    copy: function (opts) {
      return build(opts || {}).then(function (out) {
        return writeClipboard(out.html, out.text).then(function () { return out; });
      });
    }
  };
})();
