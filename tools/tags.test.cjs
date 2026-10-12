'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const tags = require('../js/tags.js');

const data = {
  tags: ['A/B Testing', 'C++', 'CUDA', 'Debugging', 'Triton', 'AI-Infra', '推理'],
  counts: [1, 1, 2, 1, 1, 1, 1],
  aliases: { Debug: 'Debugging' },
  tree: [{
    n: 'AI', p: 'AI', c: [{
      n: 'AI-Infra', t: 5, p: 'AI/AI-Infra', l: [], c: [{
        n: '推理', p: 'AI/AI-Infra/推理', l: [6], c: []
      }]
    }], l: [2, 4]
  }],
  posts: [
    ['/one.html', 'One', '2026-10-03', [2, 4, 5, 6]],
    ['/two.html', 'Two', '2026-10-02', [2, 5]],
    ['/three.html', 'Three', '2026-10-01', [0, 1, 3]]
  ]
};

test('hash parser accepts plus and slash tags from old links', () => {
  assert.equal(tags.parseHash('#C++', data).items[0].id, 1);
  assert.equal(tags.parseHash('#A/B%20Testing', data).items[0].id, 0);
});

test('aliases resolve to their canonical tag and format canonically', () => {
  const parsed = tags.parseHash('#Debug', data);
  assert.equal(parsed.items[0].id, 3);
  assert.equal(parsed.redirected, true);
  assert.equal(tags.formatHash(parsed.items, data), '#Debugging');
});

test('group paths select the node tag and all descendant tags', () => {
  const path = tags.parseHash('#AI/AI-Infra/推理', data);
  assert.deepEqual(tags.tagsForSelection(path.items[0], data), [6]);
  const group = tags.parseHash('#AI/AI-Infra', data);
  assert.deepEqual(tags.tagsForSelection(group.items[0], data), [5, 6]);
  assert.deepEqual(tags.selectedPostIndices(data, group.items), [0, 1]);
  assert.deepEqual(tags.tagsForSelection(tags.parseHash('#AI-Infra', data).items[0], data), [5, 6]);
});

test('multiple tags intersect their post sets', () => {
  const parsed = tags.parseHash('#CUDA,Triton', data);
  assert.deepEqual(tags.selectedPostIndices(data, parsed.items), [0]);
  assert.deepEqual(tags.selectedPostIndices(data, parsed.items.slice(0, 1)), [0, 1]);
});

test('unknown items are reported and discarded', () => {
  const parsed = tags.parseHash('#not-a-tag', data);
  assert.deepEqual(parsed.items, []);
  assert.deepEqual(parsed.unknown, ['not-a-tag']);
});

test('filter matches an alias name as well as its canonical tag', () => {
  assert.equal(tags.filterMatches('Debugging', data.aliases, 'debug'), true);
  assert.equal(tags.filterMatches('Debugging', data.aliases, 'not-there'), false);
});
