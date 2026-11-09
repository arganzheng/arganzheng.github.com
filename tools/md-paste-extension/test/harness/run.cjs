'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');

const extensionRoot = path.resolve(__dirname, '../..');
const shotDir = process.env.MD_PASTE_SHOTS || '/Users/devin/repos/agh-paste-ext-shots';
const prefix = '/__md-paste-harness/';
const sample = [
  '# Extension paste test',
  '',
  'text-before-svg',
  '![SVG diagram](https://arganzheng.life/img/in-post/parallelism-states-grid.svg)',
  'text-between-svg-and-webp',
  '![WebP diagram](https://arganzheng.life/img/in-post/dl-evolution-xor.webp)',
  'text-between-webp-and-svg',
  '![Second SVG](https://arganzheng.life/img/in-post/parallelism-zero-stages.svg)',
  'text-after-images',
  '',
  '```markdown',
  '![Code sample](https://arganzheng.life/img/in-post/parallelism-mesh-8cards.svg)',
  '```'
].join('\n');

function contentType(file) {
  return path.extname(file) === '.html' ? 'text/html; charset=utf-8' : 'text/javascript; charset=utf-8';
}

function withoutFencedCode(value) {
  return value.replace(/^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?^ {0,3}\1[ \t]*$/gm, '');
}

async function run() {
  fs.mkdirSync(shotDir, { recursive: true });
  const server = http.createServer((request, response) => {
    const relative = new URL(request.url, 'http://127.0.0.1').pathname.slice(1) || 'index.html';
    const file = path.resolve(__dirname, relative);
    if (!file.startsWith(__dirname + path.sep) || !fs.existsSync(file)) {
      response.writeHead(404).end('not found');
      return;
    }
    response.writeHead(200, { 'content-type': contentType(file) });
    fs.createReadStream(file).pipe(response);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'md-paste-e2e-'));
  const context = await chromium.launchPersistentContext(userDataDir, {
    headless: false,
    viewport: { width: 1440, height: 960 },
    permissions: ['clipboard-read', 'clipboard-write'],
    args: [
      '--disable-extensions-except=' + extensionRoot,
      '--load-extension=' + extensionRoot
    ]
  });

  try {
    await context.route('https://arganzheng.life/__md-paste-harness/**', async (route) => {
      const pathname = new URL(route.request().url()).pathname.slice(prefix.length);
      const relative = pathname || 'index.html';
      const response = await fetch('http://127.0.0.1:' + port + '/' + relative);
      await route.fulfill({
        status: response.status,
        headers: { 'content-type': response.headers.get('content-type') || 'text/plain' },
        body: await response.text()
      });
    });

    let worker = context.serviceWorkers().find((serviceWorker) => serviceWorker.url().includes('/background.js'));
    if (!worker) {
      worker = await context.waitForEvent('serviceworker', { timeout: 15000 });
    }
    assert.match(worker.url(), /\/background\.js$/);
    assert.equal(await worker.evaluate(() => typeof globalThis.__mdPasteExtension.runOnActiveTab), 'function');

    const editors = ['textarea', 'codemirror6', 'contenteditable'];
    for (const editor of editors) {
      const page = context.pages()[0] || await context.newPage();
      await page.goto('https://arganzheng.life' + prefix + '?editor=' + editor, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.__mdPasteHarnessReady || window.__mdPasteHarnessError, { timeout: 30000 });
      assert.equal(await page.evaluate(() => window.__mdPasteHarnessError || ''), '', editor + ' harness initialization');
      await page.evaluate(() => window.__mdPasteHarness.focus());
      await page.evaluate((text) => navigator.clipboard.writeText(text), sample);

      const runResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
      assert.equal(runResult.status, undefined, editor + ' service-worker dispatch');
      const response = (runResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
      assert.ok(response, editor + ' focused frame response: ' + JSON.stringify(runResult));
      assert.equal(response.status, 'done', editor + ' paste result');
      assert.equal(response.uploaded, 3, editor + ' confirmed uploads');
      assert.equal(response.unconfirmed, 0, editor + ' uncertain uploads');
      assert.equal(response.retained, 0, editor + ' retained links');

      const result = await page.evaluate(() => ({
        source: window.__mdPasteHarness.getSource(),
        files: window.__mdPasteHarness.getUploadedFiles(),
        blobImageNodes: document.querySelectorAll('#contenteditable img[src^="blob:"]').length
      }));
      const outsideCode = withoutFencedCode(result.source);
      assert.equal((outsideCode.match(/!\[[^\]]*\]\(blob:/g) || []).length, 3, editor + ' blob image references');
      assert.equal(outsideCode.includes('https://arganzheng.life/img/in-post/parallelism-states-grid.svg'), false);
      assert.equal(outsideCode.includes('https://arganzheng.life/img/in-post/dl-evolution-xor.webp'), false);
      assert.equal(outsideCode.includes('https://arganzheng.life/img/in-post/parallelism-zero-stages.svg'), false);
      assert.ok(result.source.includes('https://arganzheng.life/img/in-post/parallelism-mesh-8cards.svg'), editor + ' code fence preserved');
      assert.ok(
        result.source.indexOf('text-before-svg') < result.source.indexOf('parallelism-states-grid.png'),
        editor + ' text-before/image order: ' + JSON.stringify(result.source)
      );
      assert.ok(
        result.source.indexOf('parallelism-states-grid.png') < result.source.indexOf('text-between-svg-and-webp'),
        editor + ' first image/text order: ' + JSON.stringify(result.source)
      );
      assert.ok(result.source.indexOf('text-between-svg-and-webp') < result.source.indexOf('dl-evolution-xor.png'));
      assert.ok(result.source.indexOf('dl-evolution-xor.png') < result.source.indexOf('text-between-webp-and-svg'));
      assert.ok(result.source.indexOf('text-between-webp-and-svg') < result.source.indexOf('parallelism-zero-stages.png'));
      assert.ok(result.source.indexOf('parallelism-zero-stages.png') < result.source.indexOf('text-after-images'));
      assert.equal(result.files.length, 3, editor + ' received three Files');
      assert.ok(result.files.every((file) => file.type === 'image/png' && file.name.endsWith('.png')), editor + ' images converted to PNG');

      await page.screenshot({ path: path.join(shotDir, editor + '.png'), fullPage: true });
      const textPath = editor === 'codemirror6' ?
        'text via CodeMirror paste handler' : 'text via insertText fallback';
      console.log(editor + ': ' + textPath + '; images via paste File handler; 3 PNG uploads confirmed');
    }

    const page = context.pages()[0] || await context.newPage();
    await page.goto('https://arganzheng.life' + prefix + '?editor=textarea', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__mdPasteHarnessReady);
    await page.evaluate(() => window.__mdPasteHarness.focus());
    const richHtml = '<p>Rich-copy prose without Markdown images.</p>' +
      '<img src="data:image/png;base64,iVBORw0KGgo=">';
    await page.evaluate(async ({ plain, html }) => {
      const clipboardItem = new ClipboardItem({
        'text/plain': new Blob([plain], { type: 'text/plain' }),
        'text/html': new Blob([html], { type: 'text/html' })
      });
      await navigator.clipboard.write([clipboardItem]);
    }, { plain: 'Rich-copy prose without Markdown images.', html: richHtml });
    const richResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const richReply = (richResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(richReply && richReply.status, 'rich-text');
    assert.equal(await page.locator('#textarea').inputValue(), '');
    const richToast = '剪贴板里是富文本，图片不在 Markdown 里。请在博客导出菜单点「复制 Markdown」后再试。';
    await page.waitForFunction((expected) => {
      const host = document.getElementById('md-paste-extension-toast');
      return host && host.shadowRoot.querySelector('.toast').textContent === expected;
    }, richToast);
    await page.screenshot({
      path: path.join(shotDir, 'rich-text-hint.png'),
      fullPage: true
    });
    console.log('rich-text hint: no insertion; HTML image detected; toast confirmed');

    const plainMarkdown = '# Plain Markdown\n\nNo blog images here.';
    await page.evaluate((text) => navigator.clipboard.writeText(text), plainMarkdown);
    const plainResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const plainReply = (plainResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(plainReply && plainReply.status, 'text-only');
    assert.equal(await page.locator('#textarea').inputValue(), plainMarkdown);
    const plainToast = await page.evaluate(() => {
      const host = document.getElementById('md-paste-extension-toast');
      return host && host.shadowRoot.querySelector('.toast').textContent;
    });
    assert.ok(plainToast.startsWith('已粘贴 '), plainToast);
    console.log('image-less Markdown: text inserted; informational toast confirmed');
  } finally {
    await context.close();
    server.close();
    fs.rmSync(userDataDir, { recursive: true, force: true });
  }
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
