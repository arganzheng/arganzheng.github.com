'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { isAllowedBlogImageUrl, splitMarkdown } = require('../split-markdown.js');

test('splits allowed Markdown and HTML images while preserving every source character', () => {
  const markdown = [
    'before ![inline image](https://arganzheng.life/one.svg "title") after',
    '<img width="24" alt="HTML image" src="https://arganzheng.life/two.webp">',
    '```markdown',
    '![fenced](https://arganzheng.life/fenced.svg)',
    '```',
    'Inline `![code](https://arganzheng.life/inline.svg)` stays literal.',
    'Foreign ![external](https://example.com/external.png) stays literal.'
  ].join('\n');
  const tokens = splitMarkdown(markdown, isAllowedBlogImageUrl);
  const images = tokens.filter((token) => token.type === 'image');

  assert.equal(images.length, 2);
  assert.equal(images[0].url, 'https://arganzheng.life/one.svg');
  assert.equal(images[0].alt, 'inline image');
  assert.match(images[0].raw, /"title"/);
  assert.equal(images[1].url, 'https://arganzheng.life/two.webp');
  assert.equal(images[1].alt, 'HTML image');
  assert.match(images[1].raw, /^<img\b/i);
  assert.ok(tokens.some((token) => token.type === 'text' && token.text.includes('https://example.com/external.png')));
  assert.ok(tokens.some((token) => token.type === 'text' && token.text.includes('https://arganzheng.life/fenced.svg')));
  assert.ok(tokens.some((token) => token.type === 'text' && token.text.includes('https://arganzheng.life/inline.svg')));
  assert.equal(tokens.map((token) => token.type === 'image' ? token.raw : token.text).join(''), markdown);
});

test('allows only the exact blog hostname over HTTP or HTTPS', () => {
  assert.equal(isAllowedBlogImageUrl('https://arganzheng.life/image.svg'), true);
  assert.equal(isAllowedBlogImageUrl('http://arganzheng.life/image.png'), true);
  assert.equal(isAllowedBlogImageUrl('https://arganzheng.life.example.com/image.png'), false);
  assert.equal(isAllowedBlogImageUrl('https://example.com/image.png'), false);
  assert.equal(isAllowedBlogImageUrl('data:image/png;base64,AA=='), false);
});
