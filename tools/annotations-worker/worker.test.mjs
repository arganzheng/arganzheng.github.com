import assert from 'node:assert/strict';
import test from 'node:test';
import worker from './worker.js';

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
