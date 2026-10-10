---
layout: null
permalink: /moments/sw.js
---
const CACHE = 'moments-post-{{ site.time | date: "%Y%m%d%H%M%S" }}';
const APP_PATH = new URL('/moments/post.html', self.location.origin).pathname;

self.addEventListener('install', (event) => event.waitUntil(self.skipWaiting()));
self.addEventListener('activate', (event) => event.waitUntil((async () => {
  const names = await caches.keys();
  await Promise.all(names.filter((name) => name.startsWith('moments-post-') && name !== CACHE).map((name) => caches.delete(name)));
  await self.clients.claim();
})()));

function openShareDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('moments-share', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('share');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveShare(request) {
  const form = await request.formData();
  const files = form.getAll('images').filter((value) => value instanceof Blob && value.size).slice(0, 9);
  const data = {
    title: String(form.get('title') || ''),
    text: String(form.get('text') || ''),
    url: String(form.get('url') || ''),
    files: files.map((blob) => ({ blob, name: blob.name || 'shared-image', type: blob.type || '' }))
  };
  const db = await openShareDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('share', 'readwrite');
    tx.objectStore('share').put(data, 'latest');
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
  return Response.redirect(new URL('/moments/post.html?shared=1', self.location.origin), 303);
}

self.addEventListener('fetch', (event) => {
  const request = event.request, url = new URL(request.url);
  if (request.method === 'POST' && url.origin === self.location.origin && url.pathname === APP_PATH && url.searchParams.has('share')) {
    event.respondWith(saveShare(request));
    return;
  }
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  const appAsset = url.pathname === APP_PATH || ['script', 'style', 'image', 'font'].includes(request.destination);
  if (!appAsset) return;
  const refresh = fetch(request).then((response) => caches.open(CACHE).then((cache) => {
    if (response.ok) cache.put(request, response.clone());
    return response;
  }));
  event.waitUntil(refresh.catch(() => {}));
  event.respondWith(caches.open(CACHE).then((cache) => cache.match(request, { ignoreSearch: true }))
    .then((cached) => cached || refresh));
});
