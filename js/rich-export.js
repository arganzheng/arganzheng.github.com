/*!
 * rich-export.js — rich clipboard HTML for author-only Markdown copy.
 * window.RichExport.articleHtml({url, title, subtitle}) ->
 *   Promise<{html, images, failed}>
 */
(function () {
  'use strict';

  var scriptVersion = (document.currentScript && (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';
  var wechatPromise = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement('script');
      script.src = src + (scriptVersion ? '?v=' + scriptVersion : '');
      script.onload = resolve;
      script.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(script);
    });
  }

  function ensureWechatExport() {
    if (window.WechatExport && window.WechatExport.REMOVE &&
        window.WechatExport.svgToPng && window.WechatExport.mermaidToPng &&
        window.WechatExport.imgToJpeg) {
      return Promise.resolve(window.WechatExport);
    }
    if (!wechatPromise) {
      wechatPromise = loadScript('/js/wechat-export.js').then(function () {
        if (!window.WechatExport || !window.WechatExport.REMOVE ||
            !window.WechatExport.svgToPng || !window.WechatExport.mermaidToPng ||
            !window.WechatExport.imgToJpeg) {
          throw new Error('WechatExport helpers unavailable');
        }
        return window.WechatExport;
      }).catch(function (error) {
        wechatPromise = null;
        throw error;
      });
    }
    return wechatPromise;
  }

  function unwrap(node) {
    if (!node.parentNode) return;
    while (node.firstChild) node.parentNode.insertBefore(node.firstChild, node);
    node.parentNode.removeChild(node);
  }

  function toDataUrl(blob) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = function () { reject(reader.error || new Error('image read failed')); };
      reader.readAsDataURL(blob);
    });
  }

  function svgUrlToPng(url) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.onload = function () {
        var width = image.naturalWidth || image.width || 800;
        var height = image.naturalHeight || image.height || 400;
        var scale = Math.min(2, 1600 / width);
        var canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        var context = canvas.getContext('2d');
        if (!context) { reject(new Error('canvas unavailable')); return; }
        context.fillStyle = '#fff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        try { resolve(canvas.toDataURL('image/png')); }
        catch (error) { reject(error); }
      };
      image.onerror = function () { reject(new Error('svg image load failed: ' + url)); };
      image.src = url;
    });
  }

  function fetchImage(url, crossOrigin) {
    return fetch(url, crossOrigin ? { mode: 'cors' } : undefined).then(function (response) {
      if (!response.ok) throw new Error('image HTTP ' + response.status);
      return response.blob();
    }).then(toDataUrl);
  }

  function replaceWithText(node, text) {
    node.replaceWith(document.createTextNode(text));
  }

  function stripAndStyle(root) {
    var nodes = [root].concat(Array.prototype.slice.call(root.querySelectorAll('*')));
    nodes.forEach(function (node) {
      var tag = node.tagName.toLowerCase();
      Array.prototype.slice.call(node.attributes).forEach(function (attribute) {
        if (attribute.name === 'class' || attribute.name === 'id' || attribute.name === 'style' ||
            /^data-/.test(attribute.name)) {
          node.removeAttribute(attribute.name);
        }
      });
      if (tag === 'img') {
        node.removeAttribute('srcset');
        node.removeAttribute('sizes');
        node.removeAttribute('loading');
        node.removeAttribute('decoding');
      } else {
        node.removeAttribute('width');
        node.removeAttribute('height');
      }
      if (tag === 'th' || tag === 'td') node.removeAttribute('align');
    });

    nodes.forEach(function (node) {
      var tag = node.tagName.toLowerCase();
      if (tag === 'pre') node.setAttribute('style', 'font-family:monospace;white-space:pre;overflow-x:auto;');
      else if (tag === 'code') node.setAttribute('style', 'font-family:monospace;');
      else if (tag === 'table') node.setAttribute('style', 'border-collapse:collapse;border:1px solid #bbb;');
      else if (tag === 'th' || tag === 'td') node.setAttribute('style', 'border:1px solid #bbb;padding:6px 8px;');
      else if (tag === 'img') node.setAttribute('style', 'max-width:100%;height:auto;');
    });
  }

  function articleHtml(options) {
    options = options || {};
    return ensureWechatExport().then(function (wechat) {
      var article = document.querySelector('.post-container');
      if (!article) throw new Error('找不到文章正文');
      var liveImages = article.querySelectorAll('img');
      var liveMermaid = article.querySelectorAll('.mermaid');
      var root = article.cloneNode(true);
      var clonedImages = root.querySelectorAll('img');
      var clonedMermaid = root.querySelectorAll('.mermaid');
      Array.prototype.forEach.call(clonedImages, function (image, index) {
        var source = liveImages[index];
        image._richExportSource = source ? (source.currentSrc || source.src || source.getAttribute('src') || source.getAttribute('data-src')) : image.getAttribute('src');
      });
      Array.prototype.forEach.call(clonedMermaid, function (diagram, index) {
        diagram._richExportSource = liveMermaid[index] || null;
      });

      root.querySelectorAll('[data-fold-owner]').forEach(function (node) {
        node.removeAttribute('hidden');
        node.removeAttribute('data-fold-owner');
      });
      root.querySelectorAll('.reversefootnote').forEach(unwrap);
      root.querySelectorAll('mark.annotation-hl').forEach(unwrap);
      root.querySelectorAll(wechat.REMOVE).forEach(function (node) { node.remove(); });
      root.querySelectorAll('.inline-tip').forEach(function (node) {
        replaceWithText(node, node.textContent);
      });
      root.querySelectorAll('a.code-ref').forEach(function (node) {
        replaceWithText(node, node.textContent);
      });

      var baseUrl = options.url || location.href;
      root.querySelectorAll('a[href]').forEach(function (link) {
        var href = link.getAttribute('href') || '';
        if (href.charAt(0) === '#') { unwrap(link); return; }
        try {
          var absolute = new URL(href, baseUrl);
          if (absolute.protocol === 'javascript:' || absolute.protocol === 'data:') link.removeAttribute('href');
          else link.setAttribute('href', absolute.href);
        } catch (error) {}
      });
      root.querySelectorAll('picture source').forEach(function (source) { source.remove(); });

      var wrapper = document.createElement('div');
      var heading = document.createElement('h1');
      heading.textContent = options.title || '';
      wrapper.appendChild(heading);
      if (options.subtitle) {
        var quote = document.createElement('blockquote');
        var paragraph = document.createElement('p');
        paragraph.textContent = options.subtitle;
        quote.appendChild(paragraph);
        wrapper.appendChild(quote);
      }
      wrapper.appendChild(root);

      var images = 0;
      var failed = 0;
      var imageJobs = [];
      root.querySelectorAll('img').forEach(function (image) {
        var source = image._richExportSource || image.getAttribute('src') || '';
        image.removeAttribute('srcset');
        image.removeAttribute('sizes');
        image.removeAttribute('loading');
        image.removeAttribute('decoding');
        if (!source) return;
        var url;
        try { url = new URL(source, location.href); }
        catch (error) { image.setAttribute('src', source); failed++; return; }
        if (url.protocol === 'data:') { image.setAttribute('src', url.href); images++; return; }
        var sameOrigin = url.origin === location.origin;
        var conversion;
        if (sameOrigin && /\.svg$/i.test(url.pathname)) conversion = svgUrlToPng(url.href);
        else if (sameOrigin && /\.webp$/i.test(url.pathname)) conversion = wechat.imgToJpeg(url.href);
        else conversion = fetchImage(url.href, !sameOrigin);
        imageJobs.push(Promise.resolve(conversion).then(function (dataUrl) {
          image.setAttribute('src', dataUrl);
          images++;
        }, function () {
          image.setAttribute('src', url.href);
          failed++;
        }));
      });

      var mermaidChain = Promise.resolve();
      Array.prototype.forEach.call(root.querySelectorAll('.mermaid'), function (diagram) {
        var liveDiagram = diagram._richExportSource;
        mermaidChain = mermaidChain.then(function () {
          return liveDiagram ? wechat.mermaidToPng(liveDiagram) : Promise.reject(new Error('diagram unavailable'));
        }).then(function (dataUrl) {
          var image = document.createElement('img');
          image.setAttribute('src', dataUrl);
          image.setAttribute('alt', '图表');
          diagram.replaceWith(image);
          images++;
        }, function () {
          var fallback = document.createElement('p');
          fallback.textContent = '（图表：见原文）';
          diagram.replaceWith(fallback);
          failed++;
        });
      });

      Array.prototype.forEach.call(root.querySelectorAll('.katex-display, .katex'), function (formula) {
        if (!root.contains(formula)) return;
        if (!formula.classList.contains('katex-display') && formula.closest('.katex-display')) return;
        var annotation = formula.querySelector('annotation[encoding="application/x-tex"]');
        var tex = annotation ? annotation.textContent.trim() : formula.textContent.trim();
        if (formula.classList.contains('katex-display')) {
          var paragraph = document.createElement('p');
          paragraph.appendChild(document.createTextNode('\\[' + tex + '\\]'));
          formula.replaceWith(paragraph);
        } else replaceWithText(formula, '\\(' + tex + '\\)');
      });

      return Promise.all(imageJobs.concat([mermaidChain])).then(function () {
        stripAndStyle(wrapper);
        return { html: wrapper.outerHTML, images: images, failed: failed };
      });
    });
  }

  window.RichExport = { articleHtml: articleHtml, svgUrlToPng: svgUrlToPng };
})();
