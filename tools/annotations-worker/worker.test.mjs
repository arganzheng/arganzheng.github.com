import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import test from 'node:test';
import worker, { entryFields, momentBody, momentInput, momentVideoKey, putMomentVideo, readPinnedIds, renderEntry, writePinnedIds } from './worker.js';

const env = {
  MOMENT_KEY: 'k',
  REPO: 'arganzheng/arganzheng.github.com',
  ALLOWED_ORIGINS: 'https://arganzheng.life',
};
const ctx = { waitUntil() {} };

function request(method, path = '/moments', authorization, origin) {
  const headers = new Headers();
  if (authorization) headers.set('Authorization', authorization);
  if (origin) headers.set('Origin', origin);
  return new Request(`https://worker.test${path}`, { method, headers });
}

test('POST /reactions rejects legacy doubt and reason kinds', async () => {
  const DB = { exec: async () => {} };
  for (const kind of ['doubt', 'reason']) {
    const response = await worker.fetch(new Request('https://worker.test/reactions', {
      method: 'POST',
      headers: { Origin: 'https://arganzheng.life', 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: '/post.html', hash: 'deadbeef', quote: 'quoted text', kind }),
    }), { ...env, DB }, ctx);
    assert.equal(response.status, 400);
  }
});

test('POST /reactions/resolve is no longer a route', async () => {
  const response = await worker.fetch(request('POST', '/reactions/resolve', undefined, 'https://arganzheng.life'), env, ctx);
  assert.equal(response.status, 404);
});

function pinRequest(authorization, body) {
  const headers = new Headers({ Origin: 'https://arganzheng.life', 'Content-Type': 'application/json' });
  if (authorization) headers.set('Authorization', authorization);
  return new Request('https://worker.test/reactions/pin', { method: 'POST', headers, body: JSON.stringify(body) });
}

function reactionDb(seed = []) {
  const rows = new Map(seed.map((row) => [`${row.path}:${row.hash}`, { pinned: 0, ...row }]));
  return {
    exec: async () => {},
    prepare(sql) {
      return {
        bind(...values) {
          if (sql.startsWith('SELECT hash, quote, section, up, share, pinned')) {
            return {
              all: async () => ({
                results: Array.from(rows.values())
                  .filter((row) => row.path === values[0] && (row.up > 0 || row.share > 0 || row.pinned > 0))
                  .map(({ hash, quote, section, up, share, pinned }) => ({ hash, quote, section, up, share, pinned })),
              }),
            };
          }
          if (sql.startsWith('INSERT INTO passage_reactions')) {
            return {
              first: async () => {
                const [path, hash, quote, section, pinned, updated_at] = values;
                const key = `${path}:${hash}`;
                const row = rows.get(key) || { path, hash, quote, section, up: 0, share: 0, pinned: 0 };
                row.quote = row.quote || quote;
                row.section = row.section || section;
                row.pinned = pinned;
                row.updated_at = updated_at;
                rows.set(key, row);
                return { up: row.up, share: row.share, pinned: row.pinned };
              },
            };
          }
          throw new Error(`unexpected D1 query: ${sql}`);
        },
      };
    },
  };
}

test('POST /reactions/pin requires authentication before validating the body', async () => {
  const response = await worker.fetch(pinRequest(undefined, {}), env, ctx);
  assert.equal(response.status, 401);
});

test('POST /reactions/pin rejects non-owners', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    if ((typeof input === 'string' ? input : input.url) === 'https://api.github.com/user') return Response.json({ login: 'reader' });
    throw new Error('unexpected fetch');
  };
  try {
    const response = await worker.fetch(pinRequest('Bearer reader-token', {
      path: '/post.html', hash: 'deadbeef', quote: 'a quoted passage', on: true,
    }), { ...env, DB: reactionDb() }, ctx);
    assert.equal(response.status, 403);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('owner pinning preserves counters and GET drops an unpinned zero-count row', async () => {
  const DB = reactionDb([{ path: '/post.html', hash: '01234567', quote: 'liked passage', section: 'Section', up: 2, share: 3 }]);
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input) => {
    if ((typeof input === 'string' ? input : input.url) === 'https://api.github.com/user') return Response.json({ login: 'arganzheng' });
    throw new Error('unexpected fetch');
  };
  const getItems = async () => {
    const response = await worker.fetch(new Request('https://worker.test/reactions?path=%2Fpost.html', {
      headers: { Origin: 'https://arganzheng.life' },
    }), { ...env, DB }, ctx);
    assert.equal(response.status, 200);
    return (await response.json()).items;
  };
  try {
    const pinned = await worker.fetch(pinRequest('Bearer owner-token', {
      path: '/post.html', hash: '01234567', quote: 'liked passage', section: 'Section', on: true,
    }), { ...env, DB }, ctx);
    assert.equal(pinned.status, 200);
    assert.deepEqual(await pinned.json(), { up: 2, share: 3, pinned: 1 });
    assert.deepEqual(await getItems(), [{
      hash: '01234567', quote: 'liked passage', section: 'Section', up: 2, share: 3, pinned: 1,
    }]);

    const newlyPinned = await worker.fetch(pinRequest('Bearer owner-token', {
      path: '/post.html', hash: 'deadbeef', quote: 'author marked passage', section: 'Section', on: true,
    }), { ...env, DB }, ctx);
    assert.deepEqual(await newlyPinned.json(), { up: 0, share: 0, pinned: 1 });
    assert.ok((await getItems()).some((row) => row.hash === 'deadbeef' && row.pinned === 1));

    const unpinned = await worker.fetch(pinRequest('Bearer owner-token', {
      path: '/post.html', hash: 'deadbeef', quote: 'author marked passage', section: 'Section', on: false,
    }), { ...env, DB }, ctx);
    assert.deepEqual(await unpinned.json(), { up: 0, share: 0, pinned: 0 });
    assert.ok(!(await getItems()).some((row) => row.hash === 'deadbeef'));
    assert.deepEqual((await getItems()).find((row) => row.hash === '01234567'), {
      hash: '01234567', quote: 'liked passage', section: 'Section', up: 2, share: 3, pinned: 1,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('pin data is excluded from reader ranking and feedback responses', async () => {
  const queries = [];
  const DB = {
    exec: async () => {},
    prepare(sql) {
      queries.push(sql);
      const statement = {
        bind() { return statement; },
        all: async () => ({ results: [] }),
        first: async () => null,
      };
      return statement;
    },
  };
  const top = await worker.fetch(new Request('https://worker.test/reactions/top?kind=up', {
    headers: { Origin: 'https://arganzheng.life' },
  }), { ...env, DB }, ctx);
  assert.deepEqual(await top.json(), { rows: [] });
  const topQuery = queries.find((sql) => sql.includes('FROM passage_reactions') && sql.includes('LIMIT'));
  assert.match(topQuery, /SELECT path, hash, quote, section, up, share, updated_at/);
  assert.match(topQuery, /WHERE \(up > 0 OR share > 0\)/);
  assert.doesNotMatch(topQuery, /\bpinned\b/);

  queries.length = 0;
  const allFeedback = await worker.fetch(new Request('https://worker.test/feedback', {
    headers: { Origin: 'https://arganzheng.life' },
  }), { ...env, DB }, ctx);
  assert.deepEqual(await allFeedback.json(), { posts: {} });
  const allFeedbackQuery = queries.find((sql) => sql.includes('FROM passage_reactions'));
  assert.match(allFeedbackQuery, /SELECT path, hash, quote, section, up, share, updated_at/);
  assert.match(allFeedbackQuery, /WHERE up > 0 OR share > 0/);
  assert.doesNotMatch(allFeedbackQuery, /\bpinned\b/);

  queries.length = 0;
  const postFeedback = await worker.fetch(new Request('https://worker.test/feedback?path=%2Fpost.html', {
    headers: { Origin: 'https://arganzheng.life' },
  }), { ...env, DB }, ctx);
  assert.deepEqual((await postFeedback.json()).reactions, []);
  const postFeedbackQuery = queries.find((sql) => sql.includes('FROM passage_reactions'));
  assert.match(postFeedbackQuery, /SELECT hash, quote, section, up, share, updated_at/);
  assert.match(postFeedbackQuery, /up > 0 OR share > 0/);
  assert.doesNotMatch(postFeedbackQuery, /\bpinned\b/);
});

test('no-Origin POST /moments accepts the configured API key and reaches publish handler', async () => {
  const response = await worker.fetch(request('POST', '/moments', 'Bearer k'), env, ctx);
  assert.equal(response.status, 501);
});

test('no-Origin POST /moments rejects a wrong API key at the Origin gate', async () => {
  const response = await worker.fetch(request('POST', '/moments', 'Bearer wrong'), env, ctx);
  assert.equal(response.status, 403);
});

test('no-Origin POST /moments does not accept a GitHub token as the API key', async () => {
  const response = await worker.fetch(request('POST', '/moments', 'Bearer github-token'), env, ctx);
  assert.equal(response.status, 403);
});

test('no-Origin GET /moments remains forbidden even with the API key', async () => {
  const response = await worker.fetch(request('GET', '/moments', 'Bearer k'), env, ctx);
  assert.equal(response.status, 403);
});

test('allowed Origin without authentication reaches the route authorization check', async () => {
  const response = await worker.fetch(request('POST', '/moments', undefined, 'https://arganzheng.life'), env, ctx);
  assert.equal(response.status, 401);
});

test('readPinnedIds parses block lists without dropping the first id', () => {
  const ids = ['20261006-0900', '20261007-0900'];
  assert.deepEqual(readPinnedIds('pinned:\n  - 20261006-0900\n'), ids.slice(0, 1));
  assert.deepEqual(readPinnedIds(`pinned:\n${ids.map((id) => `  - ${id}`).join('\n')}\n`), ids);
});

test('readPinnedIds parses inline and quoted ids', () => {
  const ids = ['20261006-0900', '20261007-0900'];
  assert.deepEqual(readPinnedIds(`pinned: [${ids.join(', ')}]\n`), ids);
  assert.deepEqual(readPinnedIds(`pinned: ['${ids[0]}', "${ids[1]}"]\n`), ids);
});

test('readPinnedIds handles empty and missing keys', () => {
  assert.deepEqual(readPinnedIds('pinned: []\n'), []);
  assert.deepEqual(readPinnedIds('other: value\n'), []);
});

test('readPinnedIds round-trips writePinnedIds output', () => {
  const ids = ['20261006-0900', '20261007-0900'];
  for (const values of [[], ids.slice(0, 1), ids]) {
    assert.deepEqual(readPinnedIds(writePinnedIds(values)), values);
  }
});

test('multipart POST without Origin and with the API key reaches publish handler', async () => {
  const form = new FormData();
  form.set('text', '一条快捷指令随笔');
  const response = await worker.fetch(new Request('https://worker.test/moments', {
    method: 'POST',
    headers: { Authorization: 'Bearer k' },
    body: form,
  }), env, ctx);
  assert.equal(response.status, 501);
});

function multipart(fields) {
  const form = new FormData();
  for (const [key, value] of fields) form.append(key, value);
  return new Request('https://worker.test/moments', { method: 'POST', body: form });
}

function decodeBase64(data) {
  return Uint8Array.from(atob(data), (char) => char.charCodeAt(0));
}

test('momentBody parses multipart images, preserves names, and sniffs PNG bytes', async () => {
  const jpeg = Uint8Array.of(0xff, 0xd8, 0xff, 0x11, 0x22);
  const png = Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a);
  const response = await momentBody(multipart([
    ['text', '海边散步'],
    ['image', new File([jpeg], 'sea.jpg', { type: 'image/jpeg' })],
    ['image', new File([png], 'sky.png', { type: '' })],
  ]), {});
  const body = response.body;
  assert.equal(body.text, '海边散步');
  assert.equal(body.images.length, 2);
  assert.equal(body.images[0].name, 'sea.jpg');
  assert.equal(body.images[0].type, 'image/jpeg');
  assert.deepEqual(decodeBase64(body.images[0].data), jpeg);
  assert.equal(body.images[1].name, 'sky.png');
  assert.equal(body.images[1].type, 'image/png');
  assert.deepEqual(decodeBase64(body.images[1].data), png);
});

test('momentBody parses comma-separated tags and gives form text entries an empty image list', async () => {
  const { body } = await momentBody(multipart([
    ['text', '散步'],
    ['tags', '散步, 海'],
  ]), {});
  assert.equal(body.text, '散步');
  assert.deepEqual(body.tags, ['散步', '海']);
  assert.deepEqual(body.images, []);
});

test('momentBody rejects oversized files before base64 encoding', async () => {
  const file = new File([new Uint8Array(3 * 1024 * 1024 + 1)], 'large.jpg', { type: 'image/jpeg' });
  await assert.rejects(momentBody(multipart([['image', file]]), {}), (error) => {
    assert.equal(error.status, 413);
    assert.equal(error.message, '第 1 张图太大（> 3 MB）');
    return true;
  });
});

test('momentBody leaves an unknown file type empty when its bytes are not recognized', async () => {
  const form = new FormData();
  form.append('image', new File([Uint8Array.of(0x00, 0x01, 0x02, 0x03)], 'unknown.bin', { type: '' }));
  const { body } = await momentBody({
    headers: new Headers({ 'Content-Type': 'multipart/form-data' }),
    formData: async () => form,
  }, {});
  assert.equal(body.images[0].type, '');
});

test('momentBody preserves text/plain and JSON request bodies', async () => {
  const text = await momentBody(new Request('https://worker.test/moments', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: '今天走了很多路',
  }), {});
  assert.deepEqual(text, { body: { text: '今天走了很多路' }, uploadedVideo: null });

  const json = { text: '散步', tags: ['散步'], images: [] };
  const parsed = await momentBody(new Request('https://worker.test/moments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(json),
  }), {});
  assert.deepEqual(parsed, { body: json, uploadedVideo: null });
});

function fakeMedia(objects = new Map()) {
  const puts = [], deletes = [], gets = [], heads = [];
  return {
    puts,
    deletes,
    gets,
    heads,
    objects,
    async put(...args) { puts.push(args); },
    async delete(...args) { deletes.push(args); },
    async get(key, { range, onlyIf } = {}) {
      gets.push({ key, range, onlyIf });
      const file = objects.get(key);
      if (!file) return null;
      const metadata = {
        size: file.bytes.length,
        httpEtag: file.etag,
        writeHttpMetadata(headers) {
          headers.set('Content-Type', file.type);
          if (file.cacheControl) headers.set('Cache-Control', file.cacheControl);
        },
      };
      const ifNoneMatch = onlyIf && onlyIf.get('If-None-Match');
      if (ifNoneMatch && (ifNoneMatch.trim() === '*' || ifNoneMatch.split(',').map((value) => value.trim()).includes(file.etag))) {
        return { ...metadata, body: null };
      }

      let bytes = file.bytes;
      let resolvedRange;
      const rangeHeader = range && range.get('Range');
      if (rangeHeader) {
        const match = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader);
        if (!match) throw new RangeError('unsatisfiable range');
        if (match[1] === '') {
          const suffix = Number(match[2]);
          if (!suffix || !bytes.length) throw new RangeError('unsatisfiable range');
          const start = Math.max(0, bytes.length - suffix);
          resolvedRange = { suffix };
          bytes = bytes.slice(start);
        } else {
          const start = Number(match[1]);
          const requestedEnd = match[2] === '' ? bytes.length - 1 : Number(match[2]);
          if (start >= bytes.length || requestedEnd < start) throw new RangeError('unsatisfiable range');
          const end = Math.min(bytes.length - 1, requestedEnd);
          resolvedRange = { offset: start, length: end - start + 1 };
          bytes = bytes.slice(start, end + 1);
        }
      }
      return { ...metadata, range: resolvedRange, body: new Response(bytes).body };
    },
    async head(key) {
      heads.push(key);
      const file = objects.get(key);
      return file ? { size: file.bytes.length } : null;
    },
  };
}

const proxyMediaKey = 'moments/2026/10/abcdef1234.mov';
const proxyMediaBytes = new TextEncoder().encode('0123456789');
function proxyMedia() {
  return fakeMedia(new Map([[
    proxyMediaKey,
    {
      bytes: proxyMediaBytes,
      type: 'video/quicktime',
      etag: '"video-etag"',
      cacheControl: 'public, max-age=31536000, immutable',
    },
  ]]));
}

function proxyRequest(path, options = {}) {
  return new Request(`https://worker.test${path}`, options);
}

test('GET /media serves full video bytes without Origin or CORS gating', async () => {
  const MEDIA = proxyMedia();
  const request = proxyRequest(`/media/${proxyMediaKey}`);
  const response = await worker.fetch(request, { MEDIA }, ctx);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Content-Type'), 'video/quicktime');
  assert.equal(response.headers.get('Accept-Ranges'), 'bytes');
  assert.equal(response.headers.get('ETag'), '"video-etag"');
  assert.equal(response.headers.get('Cache-Control'), 'public, max-age=31536000, immutable');
  assert.equal(response.headers.get('Content-Length'), String(proxyMediaBytes.length));
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), proxyMediaBytes);
  assert.equal(MEDIA.gets[0].key, proxyMediaKey);
  assert.equal(MEDIA.gets[0].range, request.headers);
  assert.equal(MEDIA.gets[0].onlyIf, request.headers);

  const fallbackMedia = proxyMedia();
  fallbackMedia.objects.get(proxyMediaKey).cacheControl = '';
  const fallback = await worker.fetch(proxyRequest(`/media/${proxyMediaKey}`), { MEDIA: fallbackMedia }, ctx);
  assert.equal(fallback.headers.get('Cache-Control'), 'public, max-age=31536000, immutable');

  const foreignOrigin = await worker.fetch(
    proxyRequest(`/media/${proxyMediaKey}`, { headers: { Origin: 'https://untrusted.example' } }),
    { MEDIA },
    ctx,
  );
  assert.equal(foreignOrigin.status, 200);
});

test('GET /media supports byte ranges and HEAD', async () => {
  const MEDIA = proxyMedia();
  const firstBytes = await worker.fetch(
    proxyRequest(`/media/${proxyMediaKey}`, { headers: { Range: 'bytes=0-1' } }),
    { MEDIA },
    ctx,
  );
  assert.equal(firstBytes.status, 206);
  assert.equal(firstBytes.headers.get('Content-Range'), `bytes 0-1/${proxyMediaBytes.length}`);
  assert.equal(firstBytes.headers.get('Content-Length'), '2');
  assert.equal(Buffer.from(await firstBytes.arrayBuffer()).toString(), '01');

  const lastBytes = await worker.fetch(
    proxyRequest(`/media/${proxyMediaKey}`, { headers: { Range: 'bytes=-3' } }),
    { MEDIA },
    ctx,
  );
  assert.equal(lastBytes.status, 206);
  assert.equal(lastBytes.headers.get('Content-Range'), `bytes 7-9/${proxyMediaBytes.length}`);
  assert.equal(Buffer.from(await lastBytes.arrayBuffer()).toString(), '789');

  const head = await worker.fetch(proxyRequest(`/media/${proxyMediaKey}`, { method: 'HEAD' }), { MEDIA }, ctx);
  assert.equal(head.status, 200);
  assert.equal(head.body, null);
  assert.equal(head.headers.get('Content-Length'), String(proxyMediaBytes.length));
});

test('GET /media handles conditional requests, missing objects, and unsatisfiable ranges', async () => {
  const MEDIA = proxyMedia();
  const notModified = await worker.fetch(
    proxyRequest(`/media/${proxyMediaKey}`, { headers: { 'If-None-Match': '"video-etag"' } }),
    { MEDIA },
    ctx,
  );
  assert.equal(notModified.status, 304);
  assert.equal(notModified.body, null);
  assert.equal(notModified.headers.get('ETag'), '"video-etag"');

  const missing = await worker.fetch(proxyRequest('/media/moments/2026/10/ffffffff.mp4'), { MEDIA }, ctx);
  assert.equal(missing.status, 404);

  const unsatisfiable = await worker.fetch(
    proxyRequest(`/media/${proxyMediaKey}`, { headers: { Range: 'bytes=100-' } }),
    { MEDIA },
    ctx,
  );
  assert.equal(unsatisfiable.status, 416);
  assert.equal(unsatisfiable.headers.get('Content-Range'), `bytes */${proxyMediaBytes.length}`);
  assert.deepEqual(MEDIA.heads, [proxyMediaKey]);
});

test('GET /media rejects invalid keys before accessing R2 and returns 404 without a binding', async () => {
  const MEDIA = proxyMedia();
  for (const path of [
    '/media/../x.mp4',
    '/media/other/2026/10/ab.mp4',
    '/media/moments/2026/10/ab.exe',
  ]) {
    const request = { url: `https://worker.test${path}`, method: 'GET', headers: new Headers() };
    const response = await worker.fetch(request, { MEDIA }, ctx);
    assert.equal(response.status, 404, path);
  }
  const normalizedDotDot = await worker.fetch(proxyRequest('/media/../x.mp4'), { MEDIA }, ctx);
  assert.equal(normalizedDotDot.status, 404);
  assert.equal(MEDIA.gets.length, 0);
  assert.equal((await worker.fetch(proxyRequest(`/media/${proxyMediaKey}`), {}, ctx)).status, 404);
});

test('putMomentVideo and momentVideoKey use the worker /media base', async () => {
  const MEDIA = fakeMedia();
  const env = {
    MEDIA,
    MEDIA_BASE: 'https://blog-annotations.arganzheng.workers.dev/media',
  };
  const src = await putMomentVideo(env, new Uint8Array([1, 2, 3]), 'video/quicktime', 3);
  assert.match(src, /^https:\/\/blog-annotations\.arganzheng\.workers\.dev\/media\/moments\/\d{4}\/\d{2}\/[0-9a-f]{10}\.mov$/);
  assert.equal(momentVideoKey(env, src), MEDIA.puts[0][0]);
});

function mediaRequest(type = 'video/mp4', length = 3, authorization = 'Bearer k') {
  return new Request('https://worker.test/moments/media', {
    method: 'POST',
    headers: {
      Authorization: authorization,
      'Content-Type': type,
      'Content-Length': String(length),
    },
    body: new Uint8Array([1, 2, 3]),
  });
}

test('POST /moments/media streams a video to R2 and returns its public URL', async () => {
  const MEDIA = fakeMedia();
  const response = await worker.fetch(mediaRequest(), { ...env, MEDIA, MEDIA_BASE: 'https://media.example' }, ctx);
  assert.equal(response.status, 201);
  assert.equal(MEDIA.puts.length, 1);
  const [key, body, options] = MEDIA.puts[0];
  assert.match(key, /^moments\/\d{4}\/\d{2}\/[0-9a-f]{10}\.mp4$/);
  assert.ok(body instanceof ReadableStream);
  assert.deepEqual(options.httpMetadata, {
    contentType: 'video/mp4',
    cacheControl: 'public, max-age=31536000, immutable',
  });
  assert.deepEqual(await response.json(), { src: `https://media.example/${key}` });
});

test('POST /moments/media returns 501 when R2 is unconfigured', async () => {
  const response = await worker.fetch(mediaRequest(), env, ctx);
  assert.equal(response.status, 501);
  assert.deepEqual(await response.json(), { error: '还没配置视频存储（R2）' });
});

test('POST /moments/media rejects oversized and unsupported videos', async () => {
  const MEDIA = fakeMedia();
  const configured = { ...env, MEDIA, MEDIA_BASE: 'https://media.example' };
  const large = await worker.fetch(mediaRequest('video/mp4', 50 * 1024 * 1024 + 1), configured, ctx);
  assert.equal(large.status, 413);
  assert.deepEqual(await large.json(), { error: '视频太大（> 50 MB）' });
  const unsupported = await worker.fetch(mediaRequest('video/avi'), configured, ctx);
  assert.equal(unsupported.status, 400);
  assert.deepEqual(await unsupported.json(), { error: '视频格式不支持：video/avi' });
  assert.equal(MEDIA.puts.length, 0);
});

test('POST /moments/media requires authentication', async () => {
  const response = await worker.fetch(
    new Request('https://worker.test/moments/media', {
      method: 'POST',
      headers: {
        Origin: 'https://arganzheng.life',
        'Content-Type': 'video/mp4',
        'Content-Length': '3',
      },
      body: new Uint8Array([1, 2, 3]),
    }),
    { ...env, MEDIA: fakeMedia(), MEDIA_BASE: 'https://media.example' },
    ctx,
  );
  assert.equal(response.status, 401);
});

const mediaEnv = { MEDIA: fakeMedia(), MEDIA_BASE: 'https://media.example' };
const validVideo = { src: 'https://media.example/moments/2026/10/0123456789.mp4' };

test('momentInput enforces video exclusivity and source ownership', () => {
  assert.throws(
    () => momentInput({ time: '2026-10-10 12:00', video: validVideo, images: [{ url: '/img/moments/2026/10/a.jpg' }] }, mediaEnv),
    { message: '视频和图片不能同时发' },
  );
  assert.throws(
    () => momentInput({ time: '2026-10-10 12:00', video: { src: 'https://elsewhere.example/moments/a.mp4' } }, mediaEnv),
    { message: '视频地址不合法' },
  );
  const fields = momentInput({ time: '2026-10-10 12:00', video: validVideo }, mediaEnv);
  assert.deepEqual(fields.video, { src: validVideo.src, poster: null });
});

test('renderEntry emits video markdown with and without a poster', () => {
  const fields = momentInput({ text: '海边', time: '2026-10-10 12:00', video: validVideo }, mediaEnv);
  assert.equal(renderEntry(fields, []).body, `海边\n\n![视频](${validVideo.src})`);
  assert.equal(
    renderEntry(fields, [], '/img/moments/2026/10/20261010-1200-poster.jpg').body,
    `海边\n\n![视频](${validVideo.src} "/img/moments/2026/10/20261010-1200-poster.jpg")`,
  );
});

test('entry fields keep video and poster out of images', () => {
  const fields = entryFields({
    body: `海边\n\n![视频](${validVideo.src} "/img/moments/2026/10/poster.jpg")`,
  });
  assert.deepEqual(fields.images, []);
  assert.deepEqual(fields.video, {
    src: validVideo.src,
    poster: '/img/moments/2026/10/poster.jpg',
  });
  assert.equal(fields.raw, false);
});

const privateKey = generateKeyPairSync('rsa', { modulusLength: 2048 }).privateKey
  .export({ type: 'pkcs8', format: 'pem' });

function encoded(content) {
  return Buffer.from(content).toString('base64');
}

function githubFetch({ monthContent = null, imageFiles = [], capture = {} } = {}) {
  return async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    const method = init.method || (typeof input === 'string' ? 'GET' : input.method) || 'GET';
    if (url === 'https://api.github.com/user') {
      return Response.json({ login: 'arganzheng', name: 'Argan', email: null, id: 1 });
    }
    if (url.endsWith('/app/installations/1/access_tokens') && method === 'POST') {
      return Response.json({ token: 'installation-token', expires_at: '2099-01-01T00:00:00Z' });
    }
    if (url.endsWith('/git/ref/heads/master')) return Response.json({ object: { sha: 'head' } });
    if (url.endsWith('/git/commits/head')) return Response.json({ tree: { sha: 'tree-head' } });
    if (url.includes('/contents/moments/2026-10.md?ref=')) {
      return monthContent == null
        ? new Response('', { status: 404 })
        : Response.json({ content: encoded(monthContent) });
    }
    if (url.includes('/contents/img/moments/2026/10?ref=')) return Response.json(imageFiles);
    if (url.endsWith('/git/blobs') && method === 'POST') return Response.json({ sha: 'poster-blob' });
    if (url.endsWith('/git/trees') && method === 'POST') {
      capture.tree = JSON.parse(init.body).tree;
      return Response.json({ sha: 'tree-new' });
    }
    if (url.endsWith('/git/commits') && method === 'POST') return Response.json({ sha: 'commit-new' });
    if (url.endsWith('/git/refs/heads/master') && method === 'PATCH') return Response.json({});
    throw new Error(`unexpected GitHub request: ${method} ${url}`);
  };
}

const githubEnv = {
  ...env,
  MEDIA_BASE: 'https://media.example',
  GITHUB_APP_ID: '1',
  GITHUB_APP_PRIVATE_KEY: privateKey,
  GITHUB_INSTALLATION_ID: '1',
};

test('multipart video uploads to R2 and publishes its markdown line', async () => {
  const MEDIA = fakeMedia();
  const form = new FormData();
  form.set('text', '海边');
  form.set('time', '2026-10-10 12:00');
  form.set('video', new File([new Uint8Array([1, 2, 3])], 'clip.mov', { type: 'application/octet-stream' }));
  form.set('poster', new File([new Uint8Array([0xff, 0xd8, 0xff, 0xd9])], 'poster.jpg', { type: 'image/jpeg' }));
  const req = new Request('https://worker.test/moments', {
    method: 'POST',
    headers: { Authorization: 'Bearer k' },
    body: form,
  });
  const capture = {};
  const originalFetch = globalThis.fetch;
  globalThis.fetch = githubFetch({ capture });
  try {
    const response = await worker.fetch(req, { ...githubEnv, MEDIA }, ctx);
    assert.equal(response.status, 201);
    assert.equal(MEDIA.puts.length, 1);
    assert.equal(MEDIA.puts[0][2].httpMetadata.contentType, 'video/quicktime');
    const month = capture.tree.find((item) => item.path === 'moments/2026-10.md').content;
    assert.match(month, /!\[视频\]\(https:\/\/media\.example\/moments\/2026\/\d{2}\/[0-9a-f]{10}\.mov "\/img\/moments\/2026\/10\/20261010-1200-poster\.jpg"\)/);
    assert.ok(capture.tree.some((item) => item.path === 'img/moments/2026/10/20261010-1200-poster.jpg'));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('GET edit response separates video from images', async () => {
  const content = `---\nlayout: moments\n---\n\n## 2026-10-10 12:00\n海边\n\n![视频](${validVideo.src} "/img/moments/2026/10/poster.jpg")\n`;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = githubFetch({ monthContent: content });
  try {
    const response = await worker.fetch(
      request('GET', '/moments?month=2026-10&id=20261010-1200', 'Bearer github-token', 'https://arganzheng.life'),
      { ...githubEnv, MEDIA: fakeMedia() },
      ctx,
    );
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.deepEqual(body.images, []);
    assert.deepEqual(body.video, { src: validVideo.src, poster: '/img/moments/2026/10/poster.jpg' });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('DELETE removes the committed video from R2', async () => {
  const MEDIA = fakeMedia();
  MEDIA.delete = async (key) => {
    MEDIA.deletes.push([key]);
    throw new Error('R2 unavailable');
  };
  const content = `---\nlayout: moments\n---\n\n## 2026-10-10 12:00\n![视频](${validVideo.src} "/img/moments/2026/10/20261010-1200-poster.jpg")\n`;
  const capture = {};
  const originalFetch = globalThis.fetch;
  globalThis.fetch = githubFetch({
    monthContent: content,
    imageFiles: [{ name: '20261010-1200-poster.jpg' }],
    capture,
  });
  try {
    const response = await worker.fetch(
      new Request('https://worker.test/moments', {
        method: 'DELETE',
        headers: {
          Origin: 'https://arganzheng.life',
          Authorization: 'Bearer github-token',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ month: '2026-10', id: '20261010-1200' }),
      }),
      { ...githubEnv, MEDIA },
      ctx,
    );
    assert.equal(response.status, 200);
    assert.deepEqual(MEDIA.deletes, [['moments/2026/10/0123456789.mp4']]);
    assert.ok(capture.tree.some((item) => item.path === 'img/moments/2026/10/20261010-1200-poster.jpg' && item.sha === null));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('PUT without a video removes the old video from R2 after the GitHub commit', async () => {
  const MEDIA = fakeMedia();
  const content = `---\nlayout: moments\n---\n\n## 2026-10-10 12:00\n![视频](${validVideo.src})\n`;
  const capture = {};
  const originalFetch = globalThis.fetch;
  globalThis.fetch = githubFetch({ monthContent: content, capture });
  try {
    const response = await worker.fetch(
      new Request('https://worker.test/moments', {
        method: 'PUT',
        headers: {
          Origin: 'https://arganzheng.life',
          Authorization: 'Bearer github-token',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          month: '2026-10',
          id: '20261010-1200',
          time: '2026-10-10 12:00',
          text: '更新',
          images: [],
        }),
      }),
      { ...githubEnv, MEDIA },
      ctx,
    );
    assert.equal(response.status, 200);
    assert.deepEqual(MEDIA.deletes, [['moments/2026/10/0123456789.mp4']]);
    const month = capture.tree.find((item) => item.path === 'moments/2026-10.md').content;
    assert.doesNotMatch(month, /media\.example\/moments/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
