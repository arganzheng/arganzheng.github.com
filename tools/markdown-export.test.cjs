// Run with: node tools/markdown-export.test.cjs
'use strict';

const assert = require('node:assert/strict');
const childProcess = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'js/share.js'), 'utf8');
const shareModule = { exports: {} };
vm.runInNewContext(source, {
  module: shareModule,
  document: {
    querySelectorAll: () => [],
    addEventListener: () => {},
    querySelector: () => null
  },
  window: {}
}, { filename: 'js/share.js' });

const absolutizeMarkdown = shareModule.exports.absolutizeMarkdown;
const rewriteMarkdownUrls = shareModule.exports.rewriteMarkdownUrls;
const buildMarkdownPayload = shareModule.exports.buildMarkdownPayload;
const { zipImageName, zipStore } = require('../js/md-zip-export.js');
const site = 'https://arganzheng.life';
const realPosts = [
  ['SVG figures', '_posts/2026-07-17-parallelism-strategies-which-state-to-shard.md'],
  ['HTML image', '_posts/2026-09-10-blog-is-back.md'],
  ['glossary include', '_posts/2026-08-11-deep-dive-into-vllm.md']
];
const syntheticMarkdown = [
  '![inline](/img/a.svg "figure title")',
  '[link](/guide/start "link title")',
  '[figure]: /img/reference.webp "reference title"',
  '![reference][figure]',
  '<img src="/img/photo.webp" srcset="/img/photo-1.webp 1x, /img/photo-2.webp 2x" alt="photo">',
  "<a href='/about.html'>about</a><video poster=\"/img/poster.webp\"></video>",
  '![cdn](//cdn.example/image.png)',
  '[relative](img/relative.png)',
  '[anchor](#x)',
  '[mail](mailto:reader@example.com)',
  '`![inline-code](/img/inline.png)`',
  '````md',
  '![fenced](/img/fenced.png)',
  '```',
  '````',
  '~~~html',
  '<img src="/img/tilde.png">',
  '~~~',
  '{% include glossary-table.html %}'
].join('\n');

function readPost(file) {
  return fs.readFileSync(path.join(root, file), 'utf8');
}

function summarize(name, file) {
  const before = readPost(file);
  const result = absolutizeMarkdown(before, site);
  const beforeLines = before.split(/\r?\n/);
  const afterLines = result.text.split(/\r?\n/);
  const changed = [];
  for (let i = 0; i < Math.max(beforeLines.length, afterLines.length); i += 1) {
    if (beforeLines[i] !== afterLines[i]) {
      changed.push({ line: i + 1, before: beforeLines[i] || '', after: afterLines[i] || '' });
    }
  }
  return {
    name,
    file,
    bytesBefore: Buffer.byteLength(before),
    bytesAfter: Buffer.byteLength(result.text),
    rewrittenUrls: result.urlCount,
    rewrittenImages: result.imageCount,
    changedLines: changed.length,
    examples: changed.slice(0, 5)
  };
}

test('rewrites Markdown and HTML root URLs while preserving code and special URLs', () => {
  const result = absolutizeMarkdown(syntheticMarkdown, site);

  assert.match(result.text, /!\[inline\]\(https:\/\/arganzheng\.life\/img\/a\.svg "figure title"\)/);
  assert.match(result.text, /\[link\]\(https:\/\/arganzheng\.life\/guide\/start "link title"\)/);
  assert.match(result.text, /\[figure\]: https:\/\/arganzheng\.life\/img\/reference\.webp "reference title"/);
  assert.match(result.text, /src="https:\/\/arganzheng\.life\/img\/photo\.webp"/);
  assert.match(result.text, /srcset="https:\/\/arganzheng\.life\/img\/photo-1\.webp 1x, https:\/\/arganzheng\.life\/img\/photo-2\.webp 2x"/);
  assert.match(result.text, /href='https:\/\/arganzheng\.life\/about\.html'/);
  assert.match(result.text, /poster="https:\/\/arganzheng\.life\/img\/poster\.webp"/);
  assert.match(result.text, /!\[cdn\]\(\/\/cdn\.example\/image\.png\)/);
  assert.match(result.text, /\[relative\]\(img\/relative\.png\)/);
  assert.match(result.text, /\[anchor\]\(#x\)/);
  assert.match(result.text, /\[mail\]\(mailto:reader@example\.com\)/);
  assert.match(result.text, /`!\[inline-code\]\(\/img\/inline\.png\)`/);
  assert.match(result.text, /!\[fenced\]\(\/img\/fenced\.png\)/);
  assert.match(result.text, /<img src="\/img\/tilde\.png">/);
  assert.match(result.text, /\{% include glossary-table\.html %\}/);
  assert.equal(result.imageCount, 6);
  assert.equal(result.urlCount, 8);
});

test('normalizes a site value with a trailing base-path slash', () => {
  const result = absolutizeMarkdown('![diagram](/img/diagram.svg)', 'https://example.test/blog/');
  assert.equal(result.text, '![diagram](https://example.test/blog/img/diagram.svg)');
  assert.equal(result.imageCount, 1);
});

test('shares the copy Markdown front-matter and title header', () => {
  const result = buildMarkdownPayload(
    '---\ntitle: Source title\n---\n\n![figure](/img/figure.svg)',
    site,
    'Page title',
    'First line\nSecond line'
  );
  assert.equal(result.text, [
    '# Page title',
    '',
    '> First line',
    '> Second line',
    '',
    '![figure](https://arganzheng.life/img/figure.svg)'
  ].join('\n'));
});

test('absolutizes an SVG-figure post', () => {
  const result = absolutizeMarkdown(readPost(realPosts[0][1]), site);
  assert.ok(result.imageCount > 0);
  assert.match(result.text, /https:\/\/arganzheng\.life\/img\/in-post\/parallelism-states-grid\.svg/);
});

test('absolutizes an HTML img src in a real post', () => {
  const result = absolutizeMarkdown(readPost(realPosts[1][1]), site);
  assert.ok(result.imageCount > 0);
  assert.match(result.text, /<img src="https:\/\/arganzheng\.life\/img\/in-post\/blog-2026-mobile\.webp"/);
});

test('leaves a real Liquid glossary include unchanged', () => {
  const result = absolutizeMarkdown(readPost(realPosts[2][1]), site);
  assert.match(result.text, /\{% include glossary-table\.html series="deep-dive-into-vllm" %\}/);
});

test('reports image and link kinds for Markdown references and HTML URLs', () => {
  const markdown = [
    '![image](/img/markdown.png)',
    '[link](/guide/start)',
    '[image-ref]: /img/reference.png',
    '![reference][image-ref]',
    '[link-ref]: /guide/reference',
    '[reference][link-ref]',
    '<img src="/img/html.png">',
    '<img srcset="/img/one.png 1x, /img/two.png 2x">',
    '<video poster="/img/poster.png"></video>',
    '<a href="/about.html">about</a>',
    '<svg><image href="/img/embedded.svg"></image></svg>'
  ].join('\n');
  const seen = [];
  const result = rewriteMarkdownUrls(markdown, (url, kind) => {
    seen.push({ url, kind });
    return '/rewritten' + url;
  });

  assert.equal(result.imageCount, 7);
  assert.equal(result.urlCount, 10);
  assert.equal(seen.filter((entry) => entry.kind === 'image').length, 7);
  assert.equal(seen.filter((entry) => entry.kind === 'link').length, 3);
  assert.deepEqual(seen.map((entry) => `${entry.kind}:${entry.url}`).sort(), [
    'image:/img/embedded.svg',
    'image:/img/html.png',
    'image:/img/markdown.png',
    'image:/img/one.png',
    'image:/img/poster.png',
    'image:/img/reference.png',
    'image:/img/two.png',
    'link:/about.html',
    'link:/guide/reference',
    'link:/guide/start'
  ]);
  assert.match(result.text, /!\[image\]\(\/rewritten\/img\/markdown\.png\)/);
  assert.match(result.text, /\[image-ref\]: \/rewritten\/img\/reference\.png/);
  assert.match(result.text, /poster="\/rewritten\/img\/poster\.png"/);
});

test('leaves protected Markdown regions unchanged during URL rewriting', () => {
  const protectedMarkdown = [
    '`![inline](/img/inline.png)`',
    '````md',
    '![fenced](/img/fenced.png)',
    '```',
    '````',
    '{% include glossary-table.html %}',
    '<!-- <img src="/img/comment.png"> -->'
  ].join('\n');
  const seen = [];
  const result = rewriteMarkdownUrls(protectedMarkdown, (url, kind) => {
    seen.push([url, kind]);
    return '/rewritten' + url;
  });

  assert.deepEqual(seen, []);
  assert.equal(result.text, protectedMarkdown);
  assert.equal(result.imageCount, 0);
  assert.equal(result.urlCount, 0);
});

test('maps image names, conversions, collisions, deduplication, and encoded names', () => {
  const used = new Map();
  assert.equal(zipImageName('/img/plot.svg?width=800', used), 'plot.png');
  assert.equal(zipImageName('/img/photo.webp#preview', used), 'photo.jpg');
  assert.equal(zipImageName('/other/plot.svg?width=400', used), 'plot-2.png');
  assert.equal(zipImageName('/third/plot.svg', used), 'plot-3.png');
  assert.equal(zipImageName('/img/plot.svg?width=800', used), 'plot.png');
  assert.equal(zipImageName('/img/%E4%B8%AD%E6%96%87.webp', used), '中文.jpg');
});

test('writes a valid UTF-8 STORE ZIP archive', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'markdown-zip-test-'));
  const archivePath = path.join(directory, 'test.zip');
  const archive = zipStore([
    { name: 'article.md', data: new TextEncoder().encode('# ZIP test\n') },
    { name: 'images/photo.png', data: new Uint8Array([0, 1, 2, 255]) },
    { name: '中文.md', data: new TextEncoder().encode('你好\n') }
  ]);
  fs.writeFileSync(archivePath, Buffer.from(archive));
  const python = [
    'import sys, zipfile',
    'z = zipfile.ZipFile(sys.argv[1])',
    'assert z.testzip() is None',
    'assert z.namelist() == ["article.md", "images/photo.png", "中文.md"]',
    'assert z.read("article.md") == b"# ZIP test\\n"',
    'assert z.read("images/photo.png") == bytes([0, 1, 2, 255])',
    'assert z.read("中文.md") == "你好\\n".encode()',
    'print("ZIP round-trip ok")'
  ].join('; ');
  try {
    const result = childProcess.spawnSync('python3', ['-c', python, archivePath], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.match(result.stdout, /ZIP round-trip ok/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

if (process.env.MARKDOWN_EXPORT_REPORT) {
  const synthetic = absolutizeMarkdown(syntheticMarkdown, site);
  const report = {
    site,
    synthetic: {
      before: syntheticMarkdown,
      after: synthetic.text,
      rewrittenUrls: synthetic.urlCount,
      rewrittenImages: synthetic.imageCount
    },
    realPosts: realPosts.map(([name, file]) => summarize(name, file))
  };
  const output = path.resolve(process.env.MARKDOWN_EXPORT_REPORT);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
}
