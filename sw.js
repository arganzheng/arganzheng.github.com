---
layout: null
permalink: /sw.js
---
'use strict';

// build {{ site.time | date: "%Y%m%d%H%M%S" }}
const CACHE_PAGES = 'blog-pages-v1';
const CACHE_STATIC = 'blog-static-v1';
const CACHE_IMAGES = 'blog-img-v1';
const CACHE_CDN = 'blog-cdn-v1';
const CACHE_SAVED = 'blog-saved-v1';
const SAVED_MANIFEST = self.location.origin + '/__offline-saved__.json';
const CURRENT_CACHES = new Set([CACHE_PAGES, CACHE_STATIC, CACHE_IMAGES, CACHE_CDN, CACHE_SAVED]);
const CDN_PREFIXES = [
  'https://cdn.jsdelivr.net/npm/katex@',
  'https://cdn.jsdelivr.net/npm/mermaid@'
];
const NAVIGATION_TIMEOUT = 8000;
const PAGE_LIMIT = 60;
const IMAGE_LIMIT = 300;

self.addEventListener('install', event => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    await self.clients.claim();
    const names = await caches.keys();
    await Promise.all(names
      .filter(name => (name.startsWith('blog-') && !CURRENT_CACHES.has(name)) ||
        name === 'precache-v1' || name === 'runtime')
      .map(name => caches.delete(name)));
  })());
});

function cacheUrl(url) {
  const key = new URL(url);
  key.search = '';
  key.hash = '';
  return key.href;
}

function pageTitle(html) {
  const match = html.match(/<title\b[^>]*>([\s\S]*?)<\/title\s*>/i);
  return match ? match[1].replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim().replace(/\s+-\s+阿甘的博客\s*\|.*$/, '') : '未命名文章';
}

async function trimPages(cache) {
  const requests = await cache.keys();
  const pages = await Promise.all(requests.map(async request => {
    const response = await cache.match(request);
    return { request, time: Number(response && response.headers.get('sw-cached-at')) || 0 };
  }));
  pages.sort((a, b) => b.time - a.time);
  await Promise.all(pages.slice(PAGE_LIMIT).map(page => cache.delete(page.request)));
}

function cacheNavigationResponse(request, network) {
  return network.then(async response => {
    if (!response.ok || !/text\/html/i.test(response.headers.get('content-type') || '')) return;

    const html = await response.clone().text();
    const cache = await caches.open(CACHE_PAGES);
    const key = cacheUrl(request.url);
    if (html.includes('name="robots" content="noindex')) {
      await cache.delete(key);
      return;
    }

    const headers = new Headers(response.headers);
    headers.delete('content-length');
    headers.delete('content-encoding');
    headers.set('sw-cached-at', String(Date.now()));
    headers.set('sw-title', encodeURIComponent(pageTitle(html)));
    await cache.put(key, new Response(html, {
      status: response.status,
      statusText: response.statusText,
      headers
    }));
    await trimPages(cache);
  }).catch(() => {});
}

function offlinePage(cache) {
  return (async () => {
    let saved = [];
    try {
      const savedCache = await caches.open(CACHE_SAVED);
      const manifestResponse = await savedCache.match(SAVED_MANIFEST);
      const manifest = manifestResponse ? await manifestResponse.json() : { pages: {} };
      saved = Object.keys(manifest.pages || {}).map(url => {
        const page = manifest.pages[url];
        return { url, title: page.title || page.path || url, time: Number(page.savedAt) || 0 };
      }).sort((a, b) => b.time - a.time);
    } catch (_) {}
    let links = [];
    if (cache) {
      const requests = await cache.keys();
      links = await Promise.all(requests.map(async request => {
        const response = await cache.match(request);
        if (!response) return null;
        let title = response.headers.get('sw-title') || '';
        try { title = decodeURIComponent(title); } catch (_) {}
        return {
          url: request.url,
          title: title || new URL(request.url).pathname,
          time: Number(response.headers.get('sw-cached-at')) || 0
        };
      }));
    }
    links = links.filter(Boolean).filter(page => !saved.some(item => item.url === page.url))
      .sort((a, b) => b.time - a.time);
    const renderItem = page => {
      const title = page.title.replace(/[&<>"']/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      })[character]);
      const url = page.url.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
      const date = new Date(page.time);
      const savedDate = date.getFullYear() + '-' +
        String(date.getMonth() + 1).padStart(2, '0') + '-' +
        String(date.getDate()).padStart(2, '0');
      return `<li><a href="${url}">${title}</a>${page.time ? `<small>（保存于 ${savedDate}）</small>` : ''}</li>`;
    };
    const savedSection = saved.length ? `<h2>离线保存的文章</h2><ul>${saved.map(renderItem).join('')}</ul>` : '';
    const recentSection = links.length ? `<h2>最近读过</h2><ul>${links.map(renderItem).join('')}</ul>` : '';
    const html = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>当前离线</title>
<style>
:root{color-scheme:light dark;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
body{max-width:720px;margin:12vh auto;padding:0 24px;background:#fff;color:#24292f;line-height:1.65}
h1{font-size:1.6rem}
li{margin:.5em 0}
a{color:#0969da}
@media(prefers-color-scheme:dark){body{background:#17181b;color:#d3d5d9}a{color:#4cb7cb}}
</style>
</head>
<body>
<h1>当前离线，这一页还没有缓存</h1>
${savedSection}${recentSection}${saved.length || links.length ? '' : '<p>目前没有已缓存的文章</p>'}
</body>
</html>`;
    return new Response(html, {
      status: 200,
      headers: { 'Content-Type': 'text/html; charset=utf-8' }
    });
  })();
}

async function offlineCopy(response) {
  const html = await response.clone().text();
  const cachedAt = response.headers.get('sw-cached-at') || '';
  const headers = new Headers(response.headers);
  headers.delete('content-length');
  headers.delete('content-encoding');
  const offlineHtml = html.replace(/<html(?=[\s>])/i, `<html data-sw-offline="${cachedAt}"`);
  return new Response(offlineHtml, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

function navigationResponse(request, event) {
  const network = fetch(request, { cache: 'no-cache' });
  event.waitUntil(cacheNavigationResponse(request, network));
  return (async () => {
    let cache = null;
    let cached = null;
    try {
      cache = await caches.open(CACHE_PAGES);
      cached = await cache.match(cacheUrl(request.url));
    } catch (_) {}

    if (!cached) {
      try { cached = await (await caches.open(CACHE_SAVED)).match(cacheUrl(request.url)); } catch (_) {}
    }

    if (!cached) {
      try {
        return await network;
      } catch (_) {
        return offlinePage(cache);
      }
    }

    const timeout = {};
    try {
      const result = await Promise.race([
        network,
        new Promise(resolve => setTimeout(() => resolve(timeout), NAVIGATION_TIMEOUT))
      ]);
      return result === timeout ? offlineCopy(cached) : result;
    } catch (_) {
      return offlineCopy(cached);
    }
  })();
}

function cacheFirstCdn(request) {
  return caches.open(CACHE_CDN).then(async cache => {
    const cached = await cache.match(request);
    if (cached) return cached;
    const saved = await (await caches.open(CACHE_SAVED)).match(request);
    if (saved) return saved;

    const response = await fetch(request);
    if (response.ok || response.type === 'opaque') {
      try { await cache.put(request, response.clone()); } catch (_) {}
    }
    return response;
  });
}

async function trimCache(cache, limit) {
  const requests = await cache.keys();
  const excess = requests.length - limit;
  if (excess > 0) {
    await Promise.all(requests.slice(0, excess).map(request => cache.delete(request)));
  }
}

function versionedAsset(request, url) {
  return caches.open(CACHE_STATIC).then(async cache => {
    const cached = await cache.match(request);
    if (cached) return cached;

    try {
      const response = await fetch(request);
      if (response.ok && response.type === 'basic') {
        await cache.put(request, response.clone());
        const older = (await cache.keys()).filter(key => {
          const oldUrl = new URL(key.url);
          return oldUrl.origin === url.origin && oldUrl.pathname === url.pathname &&
            key.url !== request.url;
        });
        await Promise.all(older.map(key => cache.delete(key)));
      }
      return response;
    } catch (error) {
      const savedCache = await caches.open(CACHE_SAVED);
      const saved = await savedCache.match(request);
      if (saved) return saved;
      const older = await cache.match(request, { ignoreSearch: true });
      if (older) return older;
      const savedOlder = await savedCache.match(request, { ignoreSearch: true });
      if (savedOlder) return savedOlder;
      throw error;
    }
  });
}

function networkFirst(request, cacheName) {
  return caches.open(cacheName).then(async cache => {
    try {
      const response = await fetch(request);
      if (response.ok && response.type === 'basic') {
        try {
          await cache.delete(request);
          await cache.put(request, response.clone());
          if (cacheName === CACHE_IMAGES) await trimCache(cache, IMAGE_LIMIT);
        } catch (_) {}
      }
      return response;
    } catch (error) {
      const cached = await cache.match(request);
      if (cached) return cached;
      const saved = await (await caches.open(CACHE_SAVED)).match(request);
      if (saved) return saved;
      throw error;
    }
  });
}

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET' || request.headers.has('range') ||
      request.destination === 'video' || request.destination === 'audio') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    if (CDN_PREFIXES.some(prefix => request.url.startsWith(prefix))) {
      event.respondWith(cacheFirstCdn(request));
    }
    return;
  }

  if (['/moments', '/admin', '/private'].some(path =>
    url.pathname === path || url.pathname.startsWith(`${path}/`))) return;

  if (request.mode === 'navigate') {
    event.respondWith(navigationResponse(request, event));
    return;
  }

  if (url.searchParams.has('v') && /\.(?:css|js)$/i.test(url.pathname)) {
    event.respondWith(versionedAsset(request, url));
    return;
  }

  const cacheName = request.destination === 'image' || request.destination === 'font'
    ? CACHE_IMAGES : CACHE_STATIC;
  event.respondWith(networkFirst(request, cacheName));
});
