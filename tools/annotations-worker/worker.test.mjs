import assert from 'node:assert/strict';
import { generateKeyPairSync } from 'node:crypto';
import test from 'node:test';
import worker, { entryFields, momentBody, momentInput, renderEntry } from './worker.js';

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

function fakeMedia() {
  const puts = [], deletes = [];
  return {
    puts,
    deletes,
    async put(...args) { puts.push(args); },
    async delete(...args) { deletes.push(args); },
  };
}

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
