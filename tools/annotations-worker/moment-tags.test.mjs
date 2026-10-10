import assert from 'node:assert/strict';
import test from 'node:test';
import { rewriteMomentTags } from './moment-tags.mjs';

test('renames exact tags and nested child tags but not prefixes', () => {
  const source = '## 2026-10-01\n#旧 #旧/子 #旧2 #旧词\n';
  const result = rewriteMomentTags(source, '旧', '新');
  assert.equal(result.content, '## 2026-10-01\n#新 #新/子 #旧2 #旧词\n');
  assert.equal(result.changed, 1);
});

test('leaves inline code, fences, indented code, and URLs untouched', () => {
  const source = '## 2026-10-01\n`#旧` https://example.test/?tag=#旧\n```\n#旧\n```\n    #旧\n#旧\n';
  const result = rewriteMomentTags(source, '旧', '新');
  assert.equal(result.content, '## 2026-10-01\n`#旧` https://example.test/?tag=#旧\n```\n#旧\n```\n    #旧\n#新\n');
});

test('merges into an existing tag without adding a duplicate list item', () => {
  const result = rewriteMomentTags('## 2026-10-01\n#旧 #新\n', '旧', '新');
  assert.equal(result.content, '## 2026-10-01\n#新 #新\n');
  assert.equal(result.changed, 1);
});
