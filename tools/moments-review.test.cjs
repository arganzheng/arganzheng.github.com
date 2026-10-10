'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { pickReview } = require('../js/moments-review.js');
const { buildMessage, pickFallback, serverChanEndpoint } = require('./moments-review.cjs');

test('pickReview prefers same day in earlier years', () => {
  const entries = [
    { id: 'a', date: '2024-11-06' },
    { id: 'b', date: '2020-11-06' },
    { id: 'c', date: '2026-10-06' }
  ];
  assert.deepEqual(pickReview(entries, 2026, 11, 6), [
    { e: entries[0], label: '2 年前的今天' },
    { e: entries[1], label: '6 年前的今天' }
  ]);
});

test('pickReview falls back to the same day of an earlier month', () => {
  const entry = { id: 'a', date: '2026-09-06' };
  assert.deepEqual(pickReview([entry], 2026, 11, 6), [
    { e: entry, label: '2 个月前的今天' }
  ]);
});

test('fallback selection is deterministic for a date', () => {
  const entries = Array.from({ length: 10 }, (_, i) => ({ id: String(i) }));
  const first = pickFallback(entries, '2026-11-06');
  assert.deepEqual(pickFallback(entries, '2026-11-06'), first);
  assert.equal(first.length, 1);
  assert.equal(first[0].label, '随机回顾');
  assert.ok(entries.includes(first[0].e));
});

test('buildMessage includes location, absolute image and link URLs, and truncates text', () => {
  const { title, body } = buildMessage([{
    label: '2 个月前的今天',
    e: {
      date: '2026-09-06',
      time: '08:30',
      place: '杭州',
      text: '字'.repeat(520),
      img: '/img/moments/photo.jpg',
      url: '/moments/20260906-0830.html'
    }
  }]);
  assert.equal(title, '随笔回顾 · 2 个月前的今天');
  assert.match(body, /\*\*2 个月前的今天\*\* · 2026-09-06 08:30 · 杭州/);
  assert.match(body, /!\[\]\(https:\/\/arganzheng\.life\/img\/moments\/photo\.jpg\)/);
  assert.match(body, /\[查看原文\]\(https:\/\/arganzheng\.life\/moments\/20260906-0830\.html\)/);
  assert.equal(body.match(/字/g).length, 499);
  assert.match(body, /…/);
});

test('Server酱 endpoint handles standard and sctp keys', () => {
  assert.equal(serverChanEndpoint('abc123'), 'https://sctapi.ftqq.com/abc123.send');
  assert.equal(serverChanEndpoint('sctp123tTOKEN'), 'https://123.push.ft07.com/send/sctp123tTOKEN.send');
});
