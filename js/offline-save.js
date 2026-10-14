(function () {
  'use strict';

  var CACHE = 'blog-saved-v1';
  var MANIFEST_URL = location.origin + '/__offline-saved__.json';
  var CDN = [
    'https://cdn.jsdelivr.net/npm/katex@',
    'https://cdn.jsdelivr.net/npm/mermaid@'
  ];
  var supported = 'caches' in window && !!navigator.serviceWorker;
  var manifestKey = new Request(MANIFEST_URL);
  var mutationQueue = Promise.resolve();

  function serializeMutation(operation) {
    var result = mutationQueue.then(operation, operation);
    mutationQueue = result.catch(function () {});
    return result;
  }

  function changed() {
    document.dispatchEvent(new CustomEvent('blog:offline-saved'));
  }

  function pageUrl(path) {
    var url = new URL(path || location.pathname, location.href);
    url.search = ''; url.hash = '';
    return url.href;
  }

  function getCache(create) {
    if (!supported) return Promise.resolve(null);
    if (create) return caches.open(CACHE);
    return caches.keys().then(function (names) {
      return names.indexOf(CACHE) === -1 ? null : caches.open(CACHE);
    });
  }

  function readManifest(cache) {
    if (!cache) return Promise.resolve({ version: 1, pages: {} });
    return cache.match(manifestKey).then(function (response) {
      return response ? response.json() : { version: 1, pages: {} };
    }).catch(function () { return { version: 1, pages: {} }; });
  }

  function writeManifest(cache, manifest) {
    return cache.put(manifestKey, new Response(JSON.stringify(manifest), {
      headers: { 'content-type': 'application/json; charset=utf-8' }
    }));
  }

  function prune(cache, manifest) {
    var pageKeys = Object.keys(manifest.pages);
    var keep = new Set([MANIFEST_URL].concat(pageKeys));
    pageKeys.forEach(function (url) {
      (manifest.pages[url].assets || []).forEach(function (asset) { keep.add(asset); });
    });
    return cache.keys().then(function (requests) {
      return Promise.all(requests.filter(function (request) {
        return !keep.has(request.url);
      }).map(function (request) { return cache.delete(request); }));
    });
  }

  function list() {
    return getCache(false).then(readManifest).then(function (manifest) {
      return Object.keys(manifest.pages).map(function (url) {
        return Object.assign({ url: url }, manifest.pages[url]);
      }).sort(function (a, b) { return b.savedAt - a.savedAt; });
    }).catch(function () { return []; });
  }

  function has(path) {
    var url = pageUrl(path);
    return getCache(false).then(readManifest).then(function (manifest) {
      return !!manifest.pages[url];
    }).catch(function () { return false; });
  }

  function titleFrom(html) {
    var doc = new DOMParser().parseFromString(html, 'text/html');
    return (doc.title || '未命名文章').replace(/\s+-\s+阿甘的博客\s*\|.*$/, '').trim();
  }

  function cacheResponse(cache, key, response, html) {
    var headers = new Headers(response.headers);
    headers.delete('content-length'); headers.delete('content-encoding');
    headers.set('sw-cached-at', String(Date.now()));
    headers.set('sw-title', encodeURIComponent(titleFrom(html)));
    return cache.put(key, new Response(html, {
      status: response.status, statusText: response.statusText, headers: headers
    }));
  }

  function collectAssets() {
    var urls = new Set(), externalImages = new Set();
    function add(value, isImage) {
      if (!value) return;
      try {
        var url = new URL(value, location.href);
        url.hash = '';
        if (url.origin !== location.origin &&
            !CDN.some(function (prefix) { return url.href.indexOf(prefix) === 0; })) {
          if (isImage) externalImages.add(url.href);
          return;
        }
        if (url.origin === location.origin &&
            !/\.(?:css|js|woff2?|ttf|otf|webp|png|jpe?g|gif|svg|avif|ico)$/i.test(url.pathname)) return;
        if (/^\/(?:admin|private|moments)(?:\/|$)/.test(url.pathname) || url.pathname === '/sw.js') return;
        urls.add(url.href);
      } catch (_) {}
    }
    try {
      performance.getEntriesByType('resource').forEach(function (entry) { add(entry.name, false); });
    } catch (_) {}
    document.querySelectorAll('.post-container img').forEach(function (image) {
      add(image.currentSrc, true); add(image.getAttribute('src'), true);
    });
    document.querySelectorAll('.intro-header, .header-img, #header-img').forEach(function (element) {
      var style = getComputedStyle(element).backgroundImage, match;
      var re = /url\(["']?([^"')]+)["']?\)/g;
      while ((match = re.exec(style))) add(match[1], false);
      if (element.tagName === 'IMG') { add(element.currentSrc, false); add(element.getAttribute('src'), false); }
    });
    document.querySelectorAll('link[rel="stylesheet"][href],script[src]').forEach(function (element) {
      add(element.href || element.src, false);
    });
    return { urls: Array.from(urls), skipped: externalImages.size };
  }

  function fetchAssets(cache, urls, onlyMissing) {
    var stored = [], bytes = 0, index = 0, workers = [];
    function next() {
      if (index >= urls.length) return Promise.resolve();
      var url = urls[index++];
      return (onlyMissing ? cache.match(url) : Promise.resolve(null)).then(function (existing) {
        if (existing) {
          stored.push(url);
          return existing.blob().then(function (blob) { bytes += blob.size; });
        }
        var options = CDN.some(function (prefix) { return url.indexOf(prefix) === 0; }) ? { mode: 'cors' } : undefined;
        return fetch(url, options).then(function (response) {
          if (!response.ok) return;
          return response.clone().blob().then(function (blob) {
            return cache.put(url, response.clone()).then(function () {
              stored.push(url); bytes += blob.size;
            });
          });
        }).catch(function () {});
      }).catch(function () {}).then(next);
    }
    for (var i = 0; i < Math.min(6, urls.length); i++) workers.push(next());
    return Promise.all(workers).then(function () { return { assets: stored, bytes: bytes }; });
  }

  function persistOnce(manifest) {
    if (Object.keys(manifest.pages).length > 0) return;
    try {
      if (localStorage.getItem('blogOfflinePersistRequested')) return;
      localStorage.setItem('blogOfflinePersistRequested', '1');
    } catch (_) {}
    try {
      if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(function () {});
    } catch (_) {}
  }

  function saveNow(path, refreshing) {
    if (!supported) return Promise.reject(new Error('浏览器不支持离线保存'));
    var key = pageUrl(path);
    if (key !== pageUrl(location.pathname)) return Promise.reject(new Error('只能保存当前文章'));
    return fetch(location.pathname, { cache: 'no-cache', credentials: 'same-origin' }).then(function (response) {
      if (!response.ok) throw new Error('无法读取文章');
      if (!/text\/html/i.test(response.headers.get('content-type') || '')) throw new Error('返回内容不是 HTML');
      return response.text().then(function (html) {
        if (html.indexOf('name="robots" content="noindex') !== -1) throw new Error('这篇文章不能离线保存');
        return getCache(true).then(function (cache) {
          return readManifest(cache).then(function (manifest) {
            if (!refreshing) persistOnce(manifest);
            var previous = manifest.pages[key];
            return cacheResponse(cache, key, response, html).then(function () {
              var found = collectAssets();
              return fetchAssets(cache, found.urls, !!refreshing).then(function (assets) {
                var now = Date.now(), entry = {
                  path: new URL(key).pathname,
                  title: titleFrom(html),
                  savedAt: previous ? previous.savedAt : now,
                  updatedAt: now,
                  bytes: new Blob([html]).size + assets.bytes,
                  assets: assets.assets,
                  skipped: found.skipped
                };
                manifest.pages[key] = entry;
                return writeManifest(cache, manifest).then(function () {
                  return prune(cache, manifest).then(function () { changed(); return entry; });
                });
              });
            });
          });
        });
      });
    });
  }

  function save(path, refreshing) {
    return serializeMutation(function () { return saveNow(path, refreshing); });
  }

  function removeNow(path) {
    var key = pageUrl(path);
    return getCache(false).then(function (cache) {
      if (!cache) return false;
      return readManifest(cache).then(function (manifest) {
        if (!manifest.pages[key]) return false;
        delete manifest.pages[key];
        return Promise.all([cache.delete(key), writeManifest(cache, manifest)]).then(function () {
          return prune(cache, manifest).then(function () { changed(); return true; });
        });
      });
    });
  }

  function remove(path) {
    return serializeMutation(function () { return removeNow(path); });
  }

  function clear() {
    if (!supported) return Promise.resolve(false);
    return serializeMutation(function () {
      return caches.delete(CACHE).then(function (deleted) {
        if (deleted) changed();
        return deleted;
      });
    });
  }

  window.BlogOfflineSave = {
    list: list, has: has, save: save, remove: remove, clear: clear,
    formatSize: formatSize,
    supported: supported
  };

  function formatSize(bytes) {
    return bytes >= 1024 * 1024 ? (bytes / (1024 * 1024)).toFixed(1) + ' MB' :
      Math.max(1, Math.round(bytes / 1024)) + ' KB';
  }
  function showToast(bar, message) {
    var toast = bar.querySelector('.pa-toast');
    if (!toast) return;
    toast.textContent = message; toast.hidden = false;
    clearTimeout(toast._offlineTimer);
    toast._offlineTimer = setTimeout(function () { toast.hidden = true; }, 2500);
  }
  function renderButton(button, entry) {
    var label = button.querySelector('.pa-offline-label'), size = button.querySelector('.pa-offline-size');
    button.classList.toggle('is-active', !!entry);
    button.setAttribute('aria-pressed', entry ? 'true' : 'false');
    label.textContent = entry ? '已离线保存' : '离线保存';
    size.textContent = entry ? '· ' + formatSize(entry.bytes) : '';
    button.title = entry ? '已保存 ' + formatSize(entry.bytes) + '，点击删除离线副本' :
      '离线保存这篇文章（连同站内图片），断网也能打开';
    button._offlineEntry = entry || null;
  }

  var button = document.querySelector('.post-actions .pa-offline');
  if (supported && button && !/^\/(?:admin|private|moments)(?:\/|$)/.test(location.pathname)) {
    button.hidden = false;
    list().then(function (entries) {
      renderButton(button, entries.find(function (entry) { return entry.url === pageUrl(button.dataset.url); }));
    });
    button.addEventListener('click', function () {
      if (button.disabled) return;
      var bar = button.closest('.post-actions'), existing = button._offlineEntry;
      button.disabled = true;
      if (existing) {
        remove(button.dataset.url).then(function () {
          renderButton(button, null); showToast(bar, '已删除离线副本');
        }).catch(function (error) { showToast(bar, '离线保存失败：' + error.message); })
          .finally(function () { button.disabled = false; });
      } else {
        button.querySelector('.pa-offline-label').textContent = '正在保存…';
        save(button.dataset.url).then(function (entry) {
          renderButton(button, entry);
          showToast(bar, '已离线保存（' + formatSize(entry.bytes) + '），断网也能打开' +
            (entry.skipped ? '；' + entry.skipped + ' 张站外图片未保存' : ''));
        }).catch(function (error) {
          renderButton(button, null);
          var reason = /quota/i.test(error.name + ' ' + error.message) ? '浏览器存储空间不足' : error.message;
          showToast(bar, '离线保存失败：' + reason);
        }).finally(function () { button.disabled = false; });
      }
    });
  }

  document.addEventListener('blog:offline-saved', function () {
    if (!button || !supported) return;
    list().then(function (entries) {
      renderButton(button, entries.find(function (entry) { return entry.url === pageUrl(button.dataset.url); }));
    });
  });
  function refreshSavedPage() {
    if (!supported || document.documentElement.dataset.swOffline || !navigator.onLine) return;
    has(location.pathname).then(function (saved) { if (saved) save(location.pathname, true).catch(function () {}); });
  }
  window.addEventListener('load', function () {
    var run = window.requestIdleCallback || function (callback) { return setTimeout(callback, 500); };
    run(refreshSavedPage);
  }, { once: true });
})();
