/*!
 * html-export.js — self-contained article export for offline reading.
 */
(function () {
  'use strict';

  var SITE_CSS = '/css/argan-blog.min.css';
  var KATEX_CSS = 'https://cdn.jsdelivr.net/npm/katex@0.18.7/dist/katex.min.css';
  var KATEX_FONTS = 'https://cdn.jsdelivr.net/npm/katex@0.18.7/dist/fonts/';
  var REMOVE = [
    '.series-nav', '.series-context', '.series-toc', '.pager', '.related-posts',
    '.comment', '.annotation-comments', '.annotation-panel', '.annotation-marker',
    '.annotation-toolbar', '.sec-react', '.heading-fold', '.code-copy',
    '.fig-feedback', '.fig-tools', '.table-tools', '.table-copy-menu',
    '.inline-popover-card', '.post-share', '.post-actions', '.post-license',
    '.post-stale', '.catalog-toggle', '.heading-anchor', 'script', 'style',
    'noscript', '.reversefootnote', '.mermaidTooltip', 'hr[style*="hidden"]'
  ];

  function fetchText(url) {
    return fetch(url).then(function (response) {
      if (!response.ok) throw new Error('HTTP ' + response.status + '：' + url);
      return response.text();
    });
  }

  function toDataUrl(blob) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve(reader.result); };
      reader.onerror = function () { reject(reader.error || new Error('图片读取失败')); };
      reader.readAsDataURL(blob);
    });
  }

  function inlineImages(root) {
    var images = Array.prototype.slice.call(root.querySelectorAll('img'));
    var origin = window.location.origin;
    var sources = images.map(function (image) {
      var src = image.getAttribute('data-export-src') || image.currentSrc || image.getAttribute('src');
      image.removeAttribute('data-export-src');
      return { image: image, src: src };
    });
    sources.forEach(function (entry) {
      var image = entry.image;
      image.removeAttribute('srcset');
      image.removeAttribute('sizes');
      var picture = image.closest('picture');
      if (picture) Array.prototype.slice.call(picture.querySelectorAll('source')).forEach(function (source) {
        source.remove();
      });
    });
    return Promise.all(sources.map(function (entry) {
      var image = entry.image, src = entry.src;
      if (!src) return Promise.resolve();
      var url = new URL(src, window.location.href);
      if (url.origin !== origin || url.protocol === 'data:') {
        image.setAttribute('src', url.href);
        return Promise.resolve();
      }
      return fetch(url.href).then(function (response) {
        if (!response.ok) throw new Error('图片下载失败（HTTP ' + response.status + '）：' + url.href);
        return response.blob();
      }).then(toDataUrl).then(function (dataUrl) {
        image.setAttribute('src', dataUrl);
      });
    }));
  }

  function absoluteLinks(root, baseUrl) {
    Array.prototype.slice.call(root.querySelectorAll('a[href]')).forEach(function (link) {
      link.href = new URL(link.getAttribute('href'), baseUrl || window.location.href).href;
    });
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, function (char) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char];
    });
  }

  function filenameFor(url) {
    var path = new URL(url, window.location.href).pathname;
    return decodeURIComponent(path.split('/').filter(Boolean).pop() || 'article.html');
  }

  function downloadHtml(html, filename) {
    var objectUrl = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }));
    var link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    link.style.display = 'none';
    document.body.appendChild(link);
    try {
      link.click();
    } finally {
      link.remove();
      setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 1000);
    }
  }

  function download(options) {
    var article = document.querySelector('.post-container');
    if (!article) return Promise.reject(new Error('找不到文章正文'));
    var content = article.cloneNode(true);
    var sourceImages = article.querySelectorAll('img');
    var clonedImages = content.querySelectorAll('img');
    Array.prototype.forEach.call(clonedImages, function (image, index) {
      var source = sourceImages[index];
      if (source) image.setAttribute('data-export-src', source.currentSrc || source.src);
    });
    REMOVE.forEach(function (selector) {
      Array.prototype.slice.call(content.querySelectorAll(selector)).forEach(function (node) {
        if (selector === 'style' && node.closest('svg')) return;
        node.remove();
      });
    });
    absoluteLinks(content, options.url);

    return Promise.all([fetchText(SITE_CSS), fetchText(KATEX_CSS)]).then(function (styles) {
      return inlineImages(content).then(function () {
        var katex = styles[1].replace(/url\((['"]?)(fonts\/[^)'"]+)\1\)/g, 'url("' + KATEX_FONTS + '$2")');
        var css = (styles[0] + '\n' + katex +
          '\nhtml,body{color-scheme:light;background:#fff;color:#24292f}' +
          '.export-header{max-width:900px;margin:2rem auto;padding:0 1rem 1.5rem;border-bottom:1px solid #ddd}' +
          '.export-header h1{margin-top:0}.export-subtitle,.export-date,.export-original{color:#666}')
          .replace(/<\/style/gi, '<\\/style');
        var title = escapeHtml(options.title);
        var subtitle = options.subtitle ? '<p class="export-subtitle">' + escapeHtml(options.subtitle) + '</p>' : '';
        var date = options.date ? '<p class="export-date">' + escapeHtml(options.date) + '</p>' : '';
        var original = '<p class="export-original"><a href="' + escapeHtml(options.url) + '">' +
          escapeHtml(options.url) + '</a></p>';
        var html = '<!doctype html><html lang="zh-CN" data-theme="light"><head>' +
          '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' +
          '<title>' + title + '</title><style>' + css + '</style></head><body data-theme="light">' +
          '<header class="export-header"><h1>' + title + '</h1>' + subtitle + date + original + '</header>' +
          content.outerHTML + '</body></html>';
        var filename = filenameFor(options.url);
        downloadHtml(html, filename);
        return filename;
      });
    });
  }

  window.HtmlExport = { download: download };
})();
