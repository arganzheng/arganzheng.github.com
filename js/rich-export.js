/*!
 * rich-export.js — rich clipboard HTML with embedded images and formula images.
 * window.RichExport.articleHtml({url, title, subtitle}) ->
 *   Promise<{html, images, formulas, failed}>
 */
(function () {
  'use strict';

  var MATHJAX_URL = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js';
  var scriptVersion = (document.currentScript && (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';
  var wechatPromise = null;
  var mathJaxPromise = null;

  function loadScript(src, versioned) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement('script');
      script.src = src + (versioned !== false && scriptVersion ? '?v=' + scriptVersion : '');
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

  function ensureMathJax() {
    if (window.MathJax && typeof window.MathJax.tex2svg === 'function') {
      return Promise.resolve(window.MathJax);
    }
    if (mathJaxPromise) return mathJaxPromise;
    if (!window.MathJax) {
      window.MathJax = { startup: { typeset: false }, svg: { fontCache: 'none' } };
    } else {
      window.MathJax.startup = window.MathJax.startup || {};
      if (window.MathJax.startup.typeset == null) window.MathJax.startup.typeset = false;
      window.MathJax.svg = window.MathJax.svg || {};
      if (!window.MathJax.svg.fontCache) window.MathJax.svg.fontCache = 'none';
    }
    mathJaxPromise = loadScript(MATHJAX_URL, false).then(function () {
      var mathJax = window.MathJax;
      return Promise.resolve(mathJax && mathJax.startup && mathJax.startup.promise).then(function () {
        if (!window.MathJax || typeof window.MathJax.tex2svg !== 'function') {
          throw new Error('MathJax tex2svg unavailable');
        }
        return window.MathJax;
      });
    }).catch(function (error) {
      mathJaxPromise = null;
      throw error;
    });
    return mathJaxPromise;
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

  function svgLength(value) {
    var match = /^\s*([\d.]+)\s*(ex|em|px|pt)?/i.exec(String(value || ''));
    if (!match) return 0;
    var number = parseFloat(match[1]);
    var unit = (match[2] || 'px').toLowerCase();
    if (unit === 'ex') return number * 8;
    if (unit === 'em') return number * 16;
    if (unit === 'pt') return number * 96 / 72;
    return number;
  }

  function rasterizeSvg(svg, width, height) {
    return new Promise(function (resolve, reject) {
      var clone = svg.cloneNode(true);
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      clone.setAttribute('width', width);
      clone.setAttribute('height', height);
      if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      var blob = new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml;charset=utf-8' });
      var objectUrl = URL.createObjectURL(blob);
      var image = new Image();
      image.onload = function () {
        var canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(width * 2));
        canvas.height = Math.max(1, Math.round(height * 2));
        var context = canvas.getContext('2d');
        if (!context) {
          URL.revokeObjectURL(objectUrl);
          reject(new Error('canvas unavailable'));
          return;
        }
        context.fillStyle = '#fff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(objectUrl);
        try { resolve(canvas.toDataURL('image/png')); }
        catch (error) { reject(error); }
      };
      image.onerror = function () {
        URL.revokeObjectURL(objectUrl);
        reject(new Error('formula svg render failed'));
      };
      image.src = objectUrl;
    });
  }

  function mathSvgToPng(svg) {
    var width = svgLength(svg.getAttribute('width'));
    var height = svgLength(svg.getAttribute('height'));
    var viewBox = (svg.getAttribute('viewBox') || '').trim().split(/[ ,]+/).map(Number);
    if ((!width || !height) && viewBox.length === 4) {
      width = width || viewBox[2];
      height = height || viewBox[3];
    }
    width = Math.max(1, Math.round(width || 100));
    height = Math.max(1, Math.round(height || 32));
    return rasterizeSvg(svg, width, height).then(function (dataUrl) {
      return { src: dataUrl, width: width, height: height };
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

  function stripAndStyle(root, formulaRecords) {
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
    formulaRecords.forEach(function (record) {
      record.image.setAttribute('style', 'max-width:100%;height:auto;width:' + record.width + 'px;vertical-align:middle;');
      if (record.display) record.wrapper.setAttribute('style', 'text-align:center;margin:1em 0;');
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
      var formulas = 0;
      var failed = 0;
      var formulaFallbacks = 0;
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

      var formulaRecords = [];
      var mathPromise = null;
      Array.prototype.forEach.call(root.querySelectorAll('.katex-display, .katex'), function (formula) {
        if (!root.contains(formula)) return;
        if (!formula.classList.contains('katex-display') && formula.closest('.katex-display')) return;
        var annotation = formula.querySelector('annotation[encoding="application/x-tex"]');
        var tex = annotation ? annotation.textContent.trim() : formula.textContent.trim();
        var display = formula.classList.contains('katex-display');
        if (!mathPromise) mathPromise = ensureMathJax();
        var job = mathPromise.then(function (mathJax) {
          var output = mathJax.tex2svg(tex, { display: display });
          var svg = output.querySelector && output.querySelector('svg');
          if (!svg) throw new Error('MathJax produced no SVG');
          return mathSvgToPng(svg);
        }).then(function (rendered) {
          var image = document.createElement('img');
          image.setAttribute('src', rendered.src);
          image.setAttribute('alt', tex);
          image.setAttribute('width', rendered.width);
          image.setAttribute('height', rendered.height);
          var wrapperNode = null;
          if (display) {
            wrapperNode = document.createElement('p');
            wrapperNode.appendChild(image);
            formula.replaceWith(wrapperNode);
          } else formula.replaceWith(image);
          formulaRecords.push({
            image: image,
            wrapper: wrapperNode,
            display: display,
            width: rendered.width,
            height: rendered.height
          });
          formulas++;
        }, function () {
          var text = display ? '\\[' + tex + '\\]' : '\\(' + tex + '\\)';
          if (display) {
            var paragraph = document.createElement('p');
            paragraph.appendChild(document.createTextNode(text));
            formula.replaceWith(paragraph);
          } else replaceWithText(formula, text);
          formulaFallbacks++;
        });
        imageJobs.push(job);
      });

      return Promise.all(imageJobs.concat([mermaidChain])).then(function () {
        stripAndStyle(wrapper, formulaRecords);
        if (formulaFallbacks && window.console && window.console.warn) {
          window.console.warn('RichExport kept ' + formulaFallbacks + ' formula(s) as TeX text.');
        }
        return { html: wrapper.outerHTML, images: images, formulas: formulas, failed: failed };
      });
    });
  }

  window.RichExport = { articleHtml: articleHtml };
})();
