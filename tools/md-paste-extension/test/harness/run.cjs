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
  '```',
  '[下一篇](https://arganzheng.life/foo.html#sec)'
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
  context.setDefaultTimeout(120000);

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
      assert.equal(response.relativeLinks, 1, editor + ' rewritten blog links');

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
      assert.ok(result.source.includes('[下一篇](/foo.html#sec)'), editor + ' blog link rewritten');
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
      const toast = await page.evaluate(() => {
        const host = document.getElementById('md-paste-extension-toast');
        return host && host.shadowRoot.querySelector('.toast').textContent;
      });
      assert.match(toast, /1 个站内链接改成了相对路径/, editor + ' relative-link toast');

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

    const plainMarkdown = '# Plain Markdown\n\n[下一篇](https://arganzheng.life/foo.html#sec)\n\nNo blog images here.';
    await page.evaluate((text) => navigator.clipboard.writeText(text), plainMarkdown);
    const plainResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const plainReply = (plainResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(plainReply && plainReply.status, 'text-only');
    assert.equal(plainReply.relativeLinks, 1);
    assert.equal(
      await page.locator('#textarea').inputValue(),
      '# Plain Markdown\n\n[下一篇](/foo.html#sec)\n\nNo blog images here.'
    );
    const plainToast = await page.evaluate(() => {
      const host = document.getElementById('md-paste-extension-toast');
      return host && host.shadowRoot.querySelector('.toast').textContent;
    });
    assert.ok(plainToast.startsWith('已粘贴 '), plainToast);
    assert.match(plainToast, /1 个站内链接改成了相对路径/);
    console.log('image-less Markdown: text inserted; informational toast confirmed');

    const trustedPage = context.pages()[0] || await context.newPage();
    await trustedPage.goto('https://arganzheng.life' + prefix + '?editor=trusted-only', { waitUntil: 'domcontentloaded' });
    await trustedPage.waitForFunction(() => window.__mdPasteHarnessReady || window.__mdPasteHarnessError, { timeout: 30000 });
    assert.equal(await trustedPage.evaluate(() => window.__mdPasteHarnessError || ''), '');
    await trustedPage.evaluate(() => window.__mdPasteHarness.focus());
    await trustedPage.evaluate((text) => navigator.clipboard.writeText(text), sample);
    const trustedResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const trustedReply = (trustedResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(trustedReply && trustedReply.status, 'done');
    assert.equal(trustedReply.uploaded, 3, 'trusted-only confirmed uploads');
    assert.equal(trustedReply.unconfirmed, 0, 'trusted-only uncertain uploads');
    assert.equal(trustedReply.retained, 0, 'trusted-only retained links');
    const trustedState = await trustedPage.evaluate(() => ({
      source: window.__mdPasteHarness.getSource(),
      files: window.__mdPasteHarness.getUploadedFiles(),
      events: window.__mdPasteHarness.getPasteEvents()
    }));
    const trustedOutsideCode = withoutFencedCode(trustedState.source);
    assert.equal((trustedOutsideCode.match(/!\[[^\]]*\]\(blob:/g) || []).length, 3);
    assert.equal(trustedOutsideCode.includes('https://arganzheng.life/img/in-post/parallelism-states-grid.svg'), false);
    assert.equal(trustedOutsideCode.includes('https://arganzheng.life/img/in-post/dl-evolution-xor.webp'), false);
    assert.equal(trustedOutsideCode.includes('https://arganzheng.life/img/in-post/parallelism-zero-stages.svg'), false);
    assert.ok(trustedState.source.includes('https://arganzheng.life/img/in-post/parallelism-mesh-8cards.svg'));
    assert.ok(trustedState.files.every((file) => file.isTrusted && file.type === 'image/png'));
    const trustedImagePastes = trustedState.events.filter((event) =>
      event.type === 'paste' && event.files > 0 && event.isTrusted
    );
    assert.equal(trustedImagePastes.length, 3, 'trusted fallback paste events');
    assert.ok(trustedState.events.some((event) =>
      event.type === 'paste' && event.files > 0 && !event.isTrusted
    ), 'synthetic paste attempts are ignored by the editor');
    assert.equal(await trustedPage.evaluate(() => navigator.clipboard.readText()), sample, 'Markdown clipboard restored');
    await trustedPage.screenshot({
      path: path.join(shotDir, 'trusted-paste.png'),
      fullPage: true
    });
    console.log('trusted-only editor: 3 trusted paste events (isTrusted=true); all PNG uploads confirmed; Markdown clipboard restored');

    const preventedPage = context.pages()[0] || await context.newPage();
    await preventedPage.goto('https://arganzheng.life' + prefix + '?editor=prevented-no-change', { waitUntil: 'domcontentloaded' });
    await preventedPage.waitForFunction(() => window.__mdPasteHarnessReady || window.__mdPasteHarnessError, { timeout: 30000 });
    assert.equal(await preventedPage.evaluate(() => window.__mdPasteHarnessError || ''), '');
    await preventedPage.evaluate(() => window.__mdPasteHarness.focus());
    const preventedMarkdown = '![Prevented paste](https://arganzheng.life/img/in-post/parallelism-states-grid.svg)';
    await preventedPage.evaluate((text) => navigator.clipboard.writeText(text), preventedMarkdown);
    const preventedResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const preventedReply = (preventedResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(preventedReply && preventedReply.status, 'done');
    assert.equal(preventedReply.uploaded, 0);
    assert.equal(preventedReply.unconfirmed, 1, 'prevented paste without an editor change is unconfirmed');
    assert.equal(preventedReply.retained, 0, 'prevented paste does not fall through to another insertion path');
    const preventedEvents = await preventedPage.evaluate(() => window.__mdPasteHarness.getPasteEvents());
    assert.equal(preventedEvents.length, 1, 'no drop or trusted fallback after prevented paste');
    assert.equal(preventedEvents[0].isTrusted, false);
    assert.equal(await preventedPage.locator('#textarea').inputValue(), '');
    console.log('prevented paste without change: unconfirmed; no duplicate fallback attempted');

    const asyncPage = context.pages()[0] || await context.newPage();
    await asyncPage.goto('https://arganzheng.life' + prefix + '?editor=async-unprevent', { waitUntil: 'domcontentloaded' });
    await asyncPage.waitForFunction(() => window.__mdPasteHarnessReady || window.__mdPasteHarnessError, { timeout: 30000 });
    assert.equal(await asyncPage.evaluate(() => window.__mdPasteHarnessError || ''), '');
    await asyncPage.evaluate(() => window.__mdPasteHarness.focus());
    await asyncPage.evaluate((text) => navigator.clipboard.writeText(text), sample);
    const asyncResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const asyncReply = (asyncResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(asyncReply && asyncReply.status, 'done');
    assert.equal(asyncReply.uploaded, 3, 'async non-prevented confirmed uploads');
    assert.equal(asyncReply.unconfirmed, 0, 'async non-prevented uncertain uploads');
    assert.equal(asyncReply.retained, 0, 'async non-prevented retained links');
    const asyncState = await asyncPage.evaluate(() => ({
      source: window.__mdPasteHarness.getSource(),
      files: window.__mdPasteHarness.getUploadedFiles(),
      events: window.__mdPasteHarness.getPasteEvents()
    }));
    const asyncOutsideCode = withoutFencedCode(asyncState.source);
    assert.equal((asyncOutsideCode.match(/!\[[^\]]*\]\(blob:/g) || []).length, 3);
    assert.equal((asyncOutsideCode.match(/!\[[^\]]*\]\(https?:\/\/arganzheng\.life\//g) || []).length, 0);
    assert.equal(asyncState.files.length, 3);
    assert.ok(asyncState.events.filter((event) => event.type === 'paste').every((event) => !event.isTrusted));
    assert.ok(asyncState.source.indexOf('text-before-svg') < asyncState.source.indexOf('parallelism-states-grid.png'));
    assert.ok(asyncState.source.indexOf('parallelism-states-grid.png') < asyncState.source.indexOf('text-between-svg-and-webp'));
    assert.ok(asyncState.source.indexOf('text-between-svg-and-webp') < asyncState.source.indexOf('dl-evolution-xor.png'));
    assert.ok(asyncState.source.indexOf('dl-evolution-xor.png') < asyncState.source.indexOf('text-between-webp-and-svg'));
    assert.ok(asyncState.source.indexOf('text-between-webp-and-svg') < asyncState.source.indexOf('parallelism-zero-stages.png'));
    assert.ok(asyncState.source.indexOf('parallelism-zero-stages.png') < asyncState.source.indexOf('text-after-images'));
    assert.ok(asyncState.source.includes('https://arganzheng.life/img/in-post/parallelism-mesh-8cards.svg'));
    await asyncPage.screenshot({
      path: path.join(shotDir, 'async-no-prevent.png'),
      fullPage: true
    });
    console.log('async editor without preventDefault: 3 uploads confirmed once each; no raw blog links retained');

    const rasterPage = context.pages()[0] || await context.newPage();
    await rasterPage.goto('https://arganzheng.life' + prefix + '?editor=textarea', { waitUntil: 'domcontentloaded' });
    await rasterPage.waitForFunction(() => window.__mdPasteHarnessReady || window.__mdPasteHarnessError, { timeout: 30000 });
    assert.equal(await rasterPage.evaluate(() => window.__mdPasteHarnessError || ''), '');
    await rasterPage.evaluate(() => window.__mdPasteHarness.focus());
    const rasterMarkdown = [
      'before raster images',
      '![GIF](https://arganzheng.life/img/in-post/zk-dlock-performance-test.gif)',
      'between raster images',
      '![JPEG](https://arganzheng.life/img/in-post/ai-infrastructures-01.jpg)',
      'after raster images'
    ].join('\n');
    await rasterPage.evaluate((text) => navigator.clipboard.writeText(text), rasterMarkdown);
    const rasterResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const rasterReply = (rasterResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(rasterReply && rasterReply.status, 'done');
    assert.equal(rasterReply.uploaded, 2, 'GIF and JPEG confirmed uploads');
    assert.equal(rasterReply.retained, 0);
    const rasterState = await rasterPage.evaluate(() => ({
      source: window.__mdPasteHarness.getSource(),
      files: window.__mdPasteHarness.getUploadedFiles()
    }));
    assert.equal((rasterState.source.match(/!\[[^\]]*\]\(blob:/g) || []).length, 2);
    assert.ok(!rasterState.source.includes('https://arganzheng.life/img/in-post/zk-dlock-performance-test.gif'));
    assert.ok(!rasterState.source.includes('https://arganzheng.life/img/in-post/ai-infrastructures-01.jpg'));
    assert.ok(rasterState.files.every((file) => file.type === 'image/png' && file.name.endsWith('.png')));
    console.log('GIF and JPEG: both converted to PNG before editor paste');

    const failedPage = context.pages()[0] || await context.newPage();
    await failedPage.goto('https://arganzheng.life' + prefix + '?editor=textarea', { waitUntil: 'domcontentloaded' });
    await failedPage.waitForFunction(() => window.__mdPasteHarnessReady || window.__mdPasteHarnessError, { timeout: 30000 });
    assert.equal(await failedPage.evaluate(() => window.__mdPasteHarnessError || ''), '');
    await failedPage.evaluate(() => window.__mdPasteHarness.focus());
    const missingImageMarkdown = [
      'before missing image',
      '![Missing image](https://arganzheng.life/img/in-post/__md-paste-missing-404__.png)',
      'after missing image'
    ].join('\n');
    await failedPage.evaluate((text) => navigator.clipboard.writeText(text), missingImageMarkdown);
    const failedResult = await worker.evaluate(async () => globalThis.__mdPasteExtension.runOnActiveTab());
    const failedReply = (failedResult.replies || []).find((reply) => reply && reply.status !== 'ignored');
    assert.equal(failedReply && failedReply.status, 'done');
    assert.equal(failedReply.uploaded, 0);
    assert.equal(failedReply.unconfirmed, 0);
    assert.equal(failedReply.retained, 1);
    assert.equal(await failedPage.locator('#textarea').inputValue(), missingImageMarkdown);
    const failedToast = await failedPage.evaluate(() => {
      const host = document.getElementById('md-paste-extension-toast');
      return host && host.shadowRoot.querySelector('.toast').textContent;
    });
    assert.match(failedToast, /下载失败 1 张/);
    await failedPage.screenshot({
      path: path.join(shotDir, 'download-failed.png'),
      fullPage: true
    });
    console.log('missing image: download failure counted and shown in retained-link reason');
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
