/*!
 * share.js — the action bar above a post's comments, `.post-stats` strips
 * (阅读 · 点赞 · 评论 · 分享) in post headers and lists, and author-only controls
 * revealed when `js/annotations.js` identifies the signed-in viewer.
 *   - 点赞 (heart): anonymous per-post counter in the worker's D1 (POST /votes,
 *     dir 'up' | null), one per browser (localStorage), no login.
 *   - 分享 popover: system share sheet (Web Share API), Weibo / X / LinkedIn
 *     intent links, WeChat QR (js/vendor/qrcode.min.js, loaded on first use),
 *     copy link. One popover element, re-anchored to whichever button opened it.
 *     Every completed share is one POST /shares; Moments and passage shares
 *     can additionally provide a PNG card generator and a long-press / download
 *     overlay.
 *   - author-only 「导出」 menu: PDF, WeChat copy, offline HTML, plain Markdown,
 *     rich HTML with embedded images/formula images, and download Markdown.
 *   - stats: list pages get everything from one GET /stats?paths=…; on the post
 *     page 点赞 / 分享 come from GET /votes and views / comment count arrive from
 *     js/annotations.js as `blog:stats` events ({views} / {comments}).
 *     WeChat and HTML modules are lazy-loaded on first use.
 */
(function () {
  'use strict';

  function absolutizeMarkdown(source, site) {
    source = String(source == null ? '' : source);
    var base = String(site || '').replace(/\/+$/, '');
    var imageCount = 0;
    var urlCount = 0;
    var protectedValues = [];
    var marker = '\u0000md-export-';
    while (source.indexOf(marker) !== -1) marker += 'x';

    function protect(value) {
      var token = marker + protectedValues.length + '\u0000';
      protectedValues.push(value);
      return token;
    }
    function refKey(label) {
      return String(label).replace(/\s+/g, ' ').replace(/^\s+|\s+$/g, '').toLowerCase();
    }
    function isRootRelative(url) {
      return /^\/(?!\/)/.test(url) && !/\{\{|\{%/.test(url);
    }
    function protectInlineCode(value) {
      var out = '';
      var i = 0;
      while (i < value.length) {
        if (value.charAt(i) !== '`') { out += value.charAt(i); i += 1; continue; }
        var start = i;
        while (value.charAt(i) === '`') i += 1;
        var length = i - start;
        var slashes = 0;
        for (var back = start - 1; back >= 0 && value.charAt(back) === '\\'; back -= 1) slashes += 1;
        if (slashes % 2) { out += value.slice(start, i); continue; }
        var search = i;
        var end = -1;
        while (search < value.length) {
          var next = value.indexOf('`', search);
          if (next < 0) break;
          var runEnd = next + 1;
          while (value.charAt(runEnd) === '`') runEnd += 1;
          if (runEnd - next === length) { end = runEnd; break; }
          search = runEnd;
        }
        if (end < 0) out += value.slice(start, i);
        else { out += protect(value.slice(start, end)); i = end; }
      }
      return out;
    }
    function protectMarkdown(value) {
      value = value.replace(/\{%[ \t]*comment[ \t]*%\}[\s\S]*?\{%[ \t]*endcomment[ \t]*%\}/gi, protect);
      value = value.replace(/<!--[\s\S]*?-->/g, protect);
      value = value.replace(/\{\{[\s\S]*?\}\}|\{%[\s\S]*?%\}/g, protect);
      return protectInlineCode(value);
    }
    function closesFence(line, fence) {
      var close = /^ {0,3}(`+|~+)[ \t]*$/.exec(line);
      return !!(close && close[1].charAt(0) === fence.char && close[1].length >= fence.length);
    }
    function rewriteMarkdownChunk(value, imageRefs) {
      value = value.replace(/(!?)\[([^\]]*)\]([ \t]*\([ \t]*)(\/(?!\/)[^\s)]*)/g, function (match, image, alt, open, url) {
        if (!isRootRelative(url)) return match;
        urlCount += 1;
        if (image) imageCount += 1;
        return image + '[' + alt + ']' + open + base + url;
      });
      value = value.replace(/^([ \t]{0,3}\[([^\]]+)\]:[ \t]*)(\/(?!\/)\S+)/gm, function (match, prefix, label, url) {
        if (!isRootRelative(url)) return match;
        urlCount += 1;
        if (imageRefs[refKey(label)]) imageCount += 1;
        return prefix + base + url;
      });
      return value.replace(/<[A-Za-z][^>]*>/g, function (tag) {
        var tagMatch = /^<\s*([a-z][\w:-]*)/i.exec(tag);
        var tagName = tagMatch ? tagMatch[1].toLowerCase() : '';
        var isImageSource = tagName === 'img' || tagName === 'source';
        tag = tag.replace(/(\s)(src|href|poster)(\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/gi, function (match, space, name, equals, doubleValue, singleValue, bareValue) {
          var url = doubleValue !== undefined ? doubleValue : (singleValue !== undefined ? singleValue : bareValue);
          if (!isRootRelative(url)) return match;
          urlCount += 1;
          var lowerName = name.toLowerCase();
          if (lowerName === 'poster' || (lowerName === 'src' && isImageSource) || (lowerName === 'href' && tagName === 'image')) imageCount += 1;
          var absolute = base + url;
          if (doubleValue !== undefined) return space + name + equals + '"' + absolute + '"';
          if (singleValue !== undefined) return space + name + equals + "'" + absolute + "'";
          return space + name + equals + absolute;
        });
        return tag.replace(/(\s)(srcset)(\s*=\s*)(?:"([^"]*)"|'([^']*)')/gi, function (match, space, name, equals, doubleValue, singleValue) {
          var value = doubleValue !== undefined ? doubleValue : singleValue;
          var updated = value.replace(/(^[ \t]*|,[ \t]*)(\/(?!\/)[^\s,]+)/g, function (candidate, separator, url) {
            if (!isRootRelative(url)) return candidate;
            urlCount += 1;
            imageCount += 1;
            return separator + base + url;
          });
          var quote = doubleValue !== undefined ? '"' : "'";
          return space + name + equals + quote + updated + quote;
        });
      });
    }

    var chunks = [];
    var pieces = source.split(/(\r\n|\n|\r)/);
    var markdown = '';
    var fence = null;
    function flushMarkdown() {
      if (markdown) chunks.push({ markdown: protectMarkdown(markdown) });
      markdown = '';
    }
    for (var i = 0; i < pieces.length; i += 2) {
      var line = pieces[i];
      var newline = pieces[i + 1] || '';
      if (fence) {
        chunks.push({ protected: line + newline });
        if (closesFence(line, fence)) fence = null;
      } else {
        var open = /^ {0,3}(`{3,}|~{3,})/.exec(line);
        if (open) {
          flushMarkdown();
          chunks.push({ protected: line + newline });
          fence = { char: open[1].charAt(0), length: open[1].length };
        } else markdown += line + newline;
      }
    }
    flushMarkdown();

    var allMarkdown = chunks.map(function (chunk) { return chunk.markdown || ''; }).join('');
    var imageRefs = Object.create(null);
    var imageReference = /!\[([^\]]*)\](?:[ \t]*\[([^\]]*)\])?/g;
    var referenceMatch;
    while ((referenceMatch = imageReference.exec(allMarkdown))) {
      if (/^[ \t]*\(/.test(allMarkdown.slice(imageReference.lastIndex))) continue;
      imageRefs[refKey(referenceMatch[2] || referenceMatch[1])] = true;
    }

    var output = chunks.map(function (chunk) {
      return chunk.markdown != null ? rewriteMarkdownChunk(chunk.markdown, imageRefs) : chunk.protected;
    }).join('');
    for (var tokenIndex = 0; tokenIndex < protectedValues.length; tokenIndex += 1) {
      output = output.split(marker + tokenIndex + '\u0000').join(protectedValues[tokenIndex]);
    }
    return { text: output, imageCount: imageCount, urlCount: urlCount };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { absolutizeMarkdown: absolutizeMarkdown };

  var bars = Array.prototype.slice.call(document.querySelectorAll('.post-actions'));
  var strips = Array.prototype.slice.call(document.querySelectorAll('.post-stats'));
  var historySpans = Array.prototype.slice.call(document.querySelectorAll('.post-history'));
  function updatePrivatePost(v) {
    var root = document.documentElement;
    var gate = document.querySelector('.private-post-gate');
    var nav = document.querySelector('.nav-private[data-author]');
    var author = (nav && nav.getAttribute('data-author')) || (gate && gate.getAttribute('data-author'));
    var isAuthor = !!(v && author && v.login === author);
    if (isAuthor) {
      root.classList.add('author-ok');
      try { localStorage.setItem('blogAuthor', v.login); } catch (e) {}
    } else {
      root.classList.remove('author-ok');
      try { localStorage.removeItem('blogAuthor'); } catch (e) {}
    }
  }
  document.addEventListener('blog:viewer', function (e) { updatePrivatePost(e.detail); });
  if (window.BlogAnnotations && window.BlogAnnotations.viewer) {
    var initialViewer = window.BlogAnnotations.viewer();
    var initialSession = window.BlogAnnotations.core && window.BlogAnnotations.core.getSession && window.BlogAnnotations.core.getSession();
    if (initialViewer || !initialSession) updatePrivatePost(initialViewer);
  }
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
  function splitMarkdownFrontMatter(source) {
    var match = /^(---[ \t]*\r?\n[\s\S]*?\r?\n---[ \t]*(?:\r?\n|$))/.exec(source);
    return match ? { frontMatter: match[1], body: source.slice(match[1].length) } : { frontMatter: '', body: source };
  }
  function markdownSite(bar) {
    return bar.getAttribute('data-site-url') || location.origin;
  }
  function copyMarkdownPayload(source, bar) {
    var parts = splitMarkdownFrontMatter(source);
    var transformed = absolutizeMarkdown(parts.body, markdownSite(bar));
    var title = bar.getAttribute('data-title') || '';
    var subtitle = bar.getAttribute('data-subtitle') || '';
    var text = '# ' + title + '\n\n';
    if (subtitle) text += subtitle.split(/\r?\n/).map(function (line) { return '> ' + line; }).join('\n') + '\n\n';
    text += transformed.text.replace(/^(?:\r?\n)+/, '');
    return { text: text, imageCount: transformed.imageCount };
  }
  function richPlainText(html) {
    var doc = new DOMParser().parseFromString(html, 'text/html');
    function text(node) {
      if (node.nodeType === 3) return node.nodeValue;
      if (node.nodeType !== 1) return '';
      var tag = node.tagName.toLowerCase();
      if (tag === 'br') return '\n';
      if (tag === 'img') return node.getAttribute('alt') || '';
      var value = Array.prototype.map.call(node.childNodes, text).join('');
      if (tag === 'td' || tag === 'th') return value + '\t';
      if (/^(address|article|aside|blockquote|dd|div|dl|dt|figcaption|figure|footer|h[1-6]|header|hr|li|main|nav|ol|p|pre|section|table|tr|ul)$/.test(tag)) {
        return '\n' + value + '\n';
      }
      return value;
    }
    return text(doc.body).replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  }

  // ---------------------------------------------------------- share popover
  var pop = null, popFor = null, exportPop = null, exportPopBtn = null;
  var menuDismissalBound = false, printPdf = null, copyWechat = null, downloadHtml = null, copyMarkdown = null, copyRich = null, downloadMarkdown = null;
  var cardOverlay = null, cardObjectUrl = null;
  function bindMenuDismissal() {
    if (menuDismissalBound) return;
    menuDismissalBound = true;
    document.addEventListener('click', function (e) {
      if (pop && !pop.hidden && !pop.contains(e.target) && !(popFor && popFor.btn.contains(e.target))) closePop();
      if (exportPop && !exportPop.hidden && !exportPop.contains(e.target) && !(exportPopBtn && exportPopBtn.contains(e.target))) closeExportPop();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { closeMenus(); closeCard(); }
    });
    window.addEventListener('resize', closeMenus);
  }
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
    bindMenuDismissal();
    return pop;
  }
  function dataOf(bar) {
    return {
      url: bar.getAttribute('data-url') || location.href.replace(/[#?].*$/, ''),
      title: bar.getAttribute('data-title') || document.title,
      text: bar.getAttribute('data-text') || ''
    };
  }
  function placePop(p, btn) {
    var r = btn.getBoundingClientRect(), pw = p.offsetWidth, ph = p.offsetHeight;
    var left = Math.min(Math.max(8, r.left + window.scrollX), window.scrollX + document.documentElement.clientWidth - pw - 8);
    var below = r.bottom + 8 + ph < window.innerHeight || r.top < ph + 8;
    p.style.left = left + 'px';
    p.style.top = (below ? r.bottom + window.scrollY + 8 : r.top + window.scrollY - ph - 8) + 'px';
    p.classList.toggle('is-above', !below);
  }
  // Open the popover anchored to `btn`; `ctx.card` optionally returns a PNG Blob.
  // js/annotations.js reuses it for passage links (BlogShare.open).
  function openPop(ctx, btn) {
    var p = ensurePop(), d = ctx;
    closeExportPop();
    if (popFor) closePop();
    popFor = ctx; ctx.btn = btn;
    ctx.onShared = ctx.onShared || function () {}; ctx.toast = ctx.toast || function () {};
    p.querySelector('[data-k="weibo"]').href = 'https://service.weibo.com/share/share.php?url=' + enc(d.url) + '&title=' + enc(d.title + (d.text ? ' — ' + d.text : ''));
    p.querySelector('[data-k="x"]').href = 'https://twitter.com/intent/tweet?url=' + enc(d.url) + '&text=' + enc(d.title + (d.text ? ' — ' + d.text : ''));
    p.querySelector('[data-k="linkedin"]').href = 'https://www.linkedin.com/sharing/share-offsite/?url=' + enc(d.url);
    p.querySelector('[data-k="image"]').hidden = typeof d.card !== 'function';
    var qr = p.querySelector('.pa-sp-qr'); qr.hidden = true; qr.querySelector('.pa-sp-qr-img').innerHTML = '';
    p.hidden = false; p.style.visibility = 'hidden';
    placePop(p, btn);
    p.style.visibility = '';
    btn.setAttribute('aria-expanded', 'true');
  }
  function closePop() {
    if (!pop || pop.hidden) return;
    pop.hidden = true;
    if (popFor && popFor.btn) popFor.btn.setAttribute('aria-expanded', 'false');
    popFor = null;
  }
  function ensureExportPop() {
    if (exportPop) return exportPop;
    exportPop = document.createElement('div');
    exportPop.className = 'pa-share-pop pa-export-pop';
    exportPop.setAttribute('role', 'menu');
    exportPop.hidden = true;
    exportPop.innerHTML =
      '<button type="button" class="pa-sp-item" role="menuitem" data-k="pdf"><i class="fa fa-file-pdf"></i>存为 PDF</button>' +
      '<button type="button" class="pa-sp-item" role="menuitem" data-k="wechat"><i class="fa fa-brands fa-weixin"></i>复制为公众号格式</button>' +
      '<button type="button" class="pa-sp-item" role="menuitem" data-k="html"><i class="fa fa-file-code"></i>HTML</button>' +
      '<button type="button" class="pa-sp-item" role="menuitem" data-k="md-copy" title="复制 Markdown（图片链接改为完整地址）"><i class="fa fa-copy"></i>复制 Markdown</button>' +
      '<button type="button" class="pa-sp-item" role="menuitem" data-k="rich" title="复制富文本（带图片）"><i class="fa fa-copy"></i>复制富文本（带图片）</button>' +
      '<button type="button" class="pa-sp-item" role="menuitem" data-k="md" title="下载 Markdown 文件"><i class="fa fa-file-lines"></i>下载 Markdown</button>';
    document.body.appendChild(exportPop);
    exportPop.addEventListener('click', function (e) {
      var item = e.target.closest('.pa-sp-item');
      if (!item || !exportPop.contains(item)) return;
      e.preventDefault();
      var kind = item.getAttribute('data-k');
      closeExportPop();
      if (kind === 'pdf' && printPdf) printPdf(item);
      else if (kind === 'wechat' && copyWechat) copyWechat(item);
      else if (kind === 'html' && downloadHtml) downloadHtml(item);
      else if (kind === 'md-copy' && copyMarkdown) copyMarkdown(item);
      else if (kind === 'rich' && copyRich) copyRich(item);
      else if (kind === 'md' && downloadMarkdown) downloadMarkdown(item);
    });
    bindMenuDismissal();
    return exportPop;
  }
  function isExportOpenFor(btn) { return !!(exportPop && !exportPop.hidden && exportPopBtn === btn); }
  function openExportPop(btn) {
    var p = ensureExportPop();
    closePop();
    exportPopBtn = btn;
    p.hidden = false; p.style.visibility = 'hidden';
    placePop(p, btn);
    p.style.visibility = '';
    btn.setAttribute('aria-expanded', 'true');
  }
  function closeExportPop() {
    if (!exportPop || exportPop.hidden) return;
    exportPop.hidden = true;
    if (exportPopBtn) exportPopBtn.setAttribute('aria-expanded', 'false');
    exportPopBtn = null;
  }
  function closeMenus() { closePop(); closeExportPop(); }
  function showCard(ctx) {
    if (typeof ctx.card !== 'function') return;
    ctx.card().then(function (blob) {
      if (!blob) return;
      closeCard();
      cardObjectUrl = URL.createObjectURL(blob);
      var touch = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
      cardOverlay = document.createElement('div');
      cardOverlay.className = 'pa-share-image';
      cardOverlay.setAttribute('role', 'dialog');
      cardOverlay.setAttribute('aria-modal', 'true');
      cardOverlay.innerHTML =
        '<div class="pa-share-image-panel">' +
          '<button type="button" class="pa-share-image-close" aria-label="关闭">×</button>' +
          '<img class="pa-share-image-preview" alt="分享图片">' +
          (touch ? '<p>长按图片保存或分享</p>' :
            '<p>右键图片可复制，或</p><div class="pa-share-image-actions">' +
            '<a class="pa-share-image-download" download="' + (ctx.filename || 'moment-' + (ctx.id || 'share') + '.png') + '">下载</a></div>') +
          '</div>';
      cardOverlay.querySelector('img').src = cardObjectUrl;
      var download = cardOverlay.querySelector('.pa-share-image-download');
      if (download) download.href = cardObjectUrl;
      cardOverlay.addEventListener('click', function (e) {
        if (e.target === cardOverlay || e.target.closest('.pa-share-image-close')) closeCard();
      });
      document.body.appendChild(cardOverlay);
      ctx.onShared('image');
    }).catch(function (err) {
      if (err && (err.name === 'AbortError' || err.name === 'NotAllowedError')) return;
      ctx.toast('图片生成失败');
    });
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
    copyText: copyText,
    showImage: function (card, filename) {
      var ctx = shareCtx(document.querySelector('.post-actions'));
      ctx.card = card;
      ctx.filename = filename;
      showCard(ctx);
    }
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
  var downloadBtn = document.querySelector('.post-actions .pa-download');
  var editBtn = document.querySelector('.post-actions .pa-edit');
  if (downloadBtn || editBtn || historySpans.length) {
    var bar = document.querySelector('.post-actions[data-author]');
    var author = (bar && bar.getAttribute('data-author')) ||
      (historySpans[0] && historySpans[0].getAttribute('data-author')) || '';
    var isAuthor = false;
    var onViewer = function (v) {
      isAuthor = !!(v && author && v.login === author);
      if (downloadBtn) downloadBtn.hidden = !isAuthor;
      if (editBtn) editBtn.hidden = !isAuthor;
      historySpans.forEach(function (span) { span.hidden = !isAuthor; });
      if (!isAuthor) closeExportPop();
    };
    document.addEventListener('blog:viewer', function (e) { onViewer(e.detail); });
    if (window.BlogAnnotations && window.BlogAnnotations.viewer) onViewer(window.BlogAnnotations.viewer());
    copyWechat = function (item) {
      if (!isAuthor || item.disabled) return;
      item.disabled = true;
      var d = dataOf(bar);
      var ready = window.WechatExport ? Promise.resolve() : loadScript('/js/wechat-export.js');
      ready.then(function () { return window.WechatExport.copy({ url: d.url, title: d.title }); }).then(function (info) {
        toast(bar, '已复制（' + info.images + ' 图 · ' + info.refs + ' 条参考链接），到公众号 / 知乎编辑器里粘贴', 4000);
      }, function (err) {
        toast(bar, '失败：' + (err && err.message || err), 4000);
      }).then(function () { item.disabled = false; });
    };
    downloadHtml = function (item) {
      if (!isAuthor || item.disabled) return;
      item.disabled = true;
      var d = dataOf(bar);
      var ready = window.HtmlExport ? Promise.resolve() : loadScript('/js/html-export.js');
      ready.then(function () {
        return window.HtmlExport.download({
          title: d.title,
          subtitle: bar.getAttribute('data-subtitle') || '',
          url: d.url,
          date: bar.getAttribute('data-date') || ''
        });
      }).then(function (filename) {
        toast(bar, '已下载 ' + filename, 4000);
      }, function (err) {
        toast(bar, '失败：' + (err && err.message || err), 4000);
      }).then(function () { item.disabled = false; });
    };
    printPdf = function (item) {
      if (!isAuthor || item.disabled) return;
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
        item.disabled = false;
      }
      item.disabled = true;
      window.addEventListener('afterprint', restorePrintState);
      root.setAttribute('data-theme', 'light');
      root.classList.add('print-export');
      try { window.print(); }
      catch (err) {
        restorePrintState();
        toast(bar, '打印失败：' + (err && err.message || err), 4000);
      }
    };
    copyMarkdown = function (item) {
      if (!isAuthor || item.disabled) return;
      item.disabled = true;
      toast(bar, '正在准备 Markdown…', 60000);
      fetchMarkdownSource(bar).then(function (source) {
        return copyText(copyMarkdownPayload(source, bar).text);
      }).then(function () {
        toast(bar, '已复制 Markdown（图片为完整链接）', 4000);
      }, function (err) {
        toast(bar, '复制失败：' + (err && err.message || err), 4000);
      }).then(function () { item.disabled = false; });
    };
    copyRich = function (item) {
      if (!isAuthor || item.disabled) return;
      item.disabled = true;
      toast(bar, '正在准备图片和公式…', 60000);
      if (!navigator.clipboard || !navigator.clipboard.write || !window.ClipboardItem) {
        toast(bar, '复制失败：浏览器不支持富文本剪贴板', 4000);
        item.disabled = false;
        return;
      }
      var richPromise = (window.RichExport ? Promise.resolve() : loadScript('/js/rich-export.js'))
        .then(function () {
          return window.RichExport.articleHtml({
            url: bar.getAttribute('data-url') || location.href,
            title: bar.getAttribute('data-title') || '',
            subtitle: bar.getAttribute('data-subtitle') || ''
          });
        });
      var htmlBlobPromise = richPromise.then(function (result) {
        return new Blob([result.html], { type: 'text/html;charset=utf-8' });
      });
      var plainBlobPromise = richPromise.then(function (result) {
        return new Blob([richPlainText(result.html)], { type: 'text/plain;charset=utf-8' });
      });
      htmlBlobPromise.catch(function () {});
      plainBlobPromise.catch(function () {});
      var writePromise;
      try {
        writePromise = navigator.clipboard.write([new window.ClipboardItem({
          'text/html': htmlBlobPromise,
          'text/plain': plainBlobPromise
        })]);
      } catch (err) {
        writePromise = Promise.reject(err);
      }
      Promise.all([writePromise, richPromise]).then(function (values) {
        var result = values[1];
        var message = '已复制富文本（' + result.images + ' 张图片、' +
          result.formulas + ' 个公式已嵌入，粘贴时编辑器会自动上传）';
        if (result.failed > 0) message += '，' + result.failed + ' 张没带上';
        toast(bar, message, 5000);
      }, function () {
        toast(bar, '复制失败：浏览器不支持富文本剪贴板', 4000);
      }).then(function () { item.disabled = false; });
    };
    downloadMarkdown = function (item) {
      if (!isAuthor || item.disabled) return;
      item.disabled = true;
      fetchMarkdownSource(bar).then(function (source) {
        var sourcePath = bar.getAttribute('data-source') || '';
        var filename = sourcePath.split(/[\\/]/).pop() || 'article.md';
        var parts = splitMarkdownFrontMatter(source);
        var transformed = absolutizeMarkdown(parts.body, markdownSite(bar));
        var blob = new Blob([parts.frontMatter, transformed.text], { type: 'text/markdown;charset=utf-8' });
        var objectUrl = URL.createObjectURL(blob);
        var link = document.createElement('a');
        link.href = objectUrl;
        link.download = filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        try {
          link.click();
          toast(bar, '已下载 ' + filename, 4000);
        } finally {
          if (link.parentNode) link.parentNode.removeChild(link);
          setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 1000);
        }
      }).catch(function (err) {
        toast(bar, '失败：' + (err && err.message || err), 4000);
      }).then(function () { item.disabled = false; });
    };
    if (downloadBtn) {
      downloadBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        if (!isAuthor) return;
        if (isExportOpenFor(downloadBtn)) closeExportPop();
        else openExportPop(downloadBtn);
      });
    }
  }

})();
