import assert from 'node:assert/strict';
import test from 'node:test';
import worker, { momentBody } from './worker.js';

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
  const request = multipart([
    ['text', '海边散步'],
    ['image', new File([jpeg], 'sea.jpg', { type: 'image/jpeg' })],
    ['image', new File([png], 'sky.png', { type: '' })],
  ]);
  const body = await momentBody(request);
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
  const body = await momentBody(multipart([
    ['text', '散步'],
    ['tags', '散步, 海'],
  ]));
  assert.equal(body.text, '散步');
  assert.deepEqual(body.tags, ['散步', '海']);
  assert.deepEqual(body.images, []);
});

test('momentBody rejects oversized files before base64 encoding', async () => {
  const file = new File([new Uint8Array(3 * 1024 * 1024 + 1)], 'large.jpg', { type: 'image/jpeg' });
  await assert.rejects(momentBody(multipart([['image', file]])), (error) => {
    assert.equal(error.status, 413);
    assert.equal(error.message, '第 1 张图太大（> 3 MB）');
    return true;
  });
});

test('momentBody leaves an unknown file type empty when its bytes are not recognized', async () => {
  const form = new FormData();
  form.append('image', new File([Uint8Array.of(0x00, 0x01, 0x02, 0x03)], 'unknown.bin', { type: '' }));
  const body = await momentBody({
    headers: new Headers({ 'Content-Type': 'multipart/form-data' }),
    formData: async () => form,
  });
  assert.equal(body.images[0].type, '');
});

test('momentBody preserves text/plain and JSON request bodies', async () => {
  const text = await momentBody(new Request('https://worker.test/moments', {
    method: 'POST',
    headers: { 'Content-Type': 'text/plain' },
    body: '今天走了很多路',
  }));
  assert.deepEqual(text, { text: '今天走了很多路' });

  const json = { text: '散步', tags: ['散步'], images: [] };
  const parsed = await momentBody(new Request('https://worker.test/moments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(json),
  }));
  assert.deepEqual(parsed, json);
});
