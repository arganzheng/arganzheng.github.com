'use strict';

importScripts('split-markdown.js');

var MENU_ID = 'paste-blog-markdown';
var OFFSCREEN_URL = chrome.runtime.getURL('offscreen.html');
var offscreenReady = null;

function allowedImageUrl(value) {
  return MdPasteMarkdown.isAllowedBlogImageUrl(value);
}

function byteArrayToBase64(bytes) {
  var binary = '';
  var chunkSize = 0x8000;
  for (var offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function imageName(url, mime) {
  var basename = 'image';
  try {
    var path = new URL(url).pathname.split('/').pop();
    basename = decodeURIComponent(path || 'image');
  } catch (_) {}
  basename = basename.replace(/[?#].*$/, '').replace(/[\\/:*?"<>|]/g, '_');
  var stem = basename.replace(/\.[^.]*$/, '') || 'image';
  var extensions = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif'
  };
  return stem + (extensions[mime] || '.png');
}

async function ensureOffscreenDocument() {
  if (offscreenReady) return offscreenReady;
  offscreenReady = (async function () {
    var exists = false;
    if (chrome.runtime.getContexts) {
      var contexts = await chrome.runtime.getContexts({
        contextTypes: ['OFFSCREEN_DOCUMENT'],
        documentUrls: [OFFSCREEN_URL]
      });
      exists = contexts.length > 0;
    } else if (chrome.offscreen.hasDocument) {
      exists = await chrome.offscreen.hasDocument();
    }
    if (!exists) {
      await chrome.offscreen.createDocument({
        url: 'offscreen.html',
        reasons: ['CLIPBOARD'],
        justification: 'Read the user clipboard after an explicit paste action and convert image files.'
      });
    }
  })();
  try {
    await offscreenReady;
  } catch (error) {
    offscreenReady = null;
    throw error;
  }
}

async function askOffscreen(message) {
  await ensureOffscreenDocument();
  return chrome.runtime.sendMessage(Object.assign({ target: 'md-paste-offscreen' }, message));
}

async function readClipboard() {
  var result = await askOffscreen({ action: 'read-clipboard' });
  if (!result || !result.ok) throw new Error(result && result.error || '无法读取剪贴板');
  return { text: result.text || '', html: result.html || '' };
}

function normalizeFetchUrl(value) {
  var url = new URL(value);
  if (url.protocol === 'http:') url.protocol = 'https:';
  return url.href;
}

async function fetchImage(value) {
  if (!allowedImageUrl(value)) throw new Error('只支持 arganzheng.life 图片');
  var fetchUrl = normalizeFetchUrl(value);
  var response = await fetch(fetchUrl, { credentials: 'omit', redirect: 'follow' });
  if (!response.ok) throw new Error('图片下载失败（HTTP ' + response.status + '）');
  if (!allowedImageUrl(response.url || fetchUrl)) throw new Error('图片跳转到了非博客域名');

  var mime = (response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
  if (!mime) mime = 'application/octet-stream';
  var bytes = new Uint8Array(await response.arrayBuffer());
  if (!bytes.length) throw new Error('图片文件为空');
  var image = {
    base64: byteArrayToBase64(bytes),
    mime: mime,
    name: imageName(value, mime)
  };

  if (mime !== 'image/png') {
    var converted = await askOffscreen({ action: 'convert-image', image: image });
    if (!converted || !converted.ok) throw new Error(converted && converted.error || '图片转换失败');
    image = converted.image;
  }
  return image;
}

async function injectAndRun(tabId, frameId) {
  var target = frameId == null ?
    { tabId: tabId, allFrames: true } :
    { tabId: tabId, frameIds: [frameId] };
  var results = await chrome.scripting.executeScript({
    target: target,
    files: ['split-markdown.js', 'content.js']
  });
  var frames = frameId == null ?
    Array.from(new Set(results.map(function (result) { return result.frameId; }))) :
    [frameId];
  var clipboard = { text: '', html: '' };
  var clipboardError = '';
  try {
    clipboard = await readClipboard();
  } catch (error) {
    clipboardError = String(error && error.message || error);
  }

  var replies = await Promise.all(frames.map(function (currentFrameId) {
    return chrome.tabs.sendMessage(tabId, {
      type: 'MD_PASTE_START',
      text: clipboard.text,
      hasHtmlImages: /<img\b/i.test(clipboard.html),
      clipboardError: clipboardError
    }, { frameId: currentFrameId }).catch(function (error) {
      return { status: 'error', error: String(error && error.message || error) };
    });
  }));
  return { frames: frames, replies: replies };
}

async function runOnActiveTab() {
  var tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  var tab = tabs[0];
  if (!tab || tab.id == null) return { status: 'error', error: '没有活动标签页' };
  try {
    return await injectAndRun(tab.id, null);
  } catch (error) {
    return { status: 'error', error: String(error && error.message || error) };
  }
}

globalThis.__mdPasteExtension = Object.freeze({ runOnActiveTab: runOnActiveTab });

chrome.runtime.onInstalled.addListener(function () {
  chrome.contextMenus.removeAll(function () {
    chrome.contextMenus.create({
      id: MENU_ID,
      title: '粘贴博客 Markdown（上传图片）',
      contexts: ['editable']
    });
  });
});

chrome.contextMenus.onClicked.addListener(function (info, tab) {
  if (info.menuItemId !== MENU_ID || !tab || tab.id == null) return;
  injectAndRun(tab.id, info.frameId).catch(function (error) {
    console.error('Markdown paste failed:', error);
  });
});

chrome.commands.onCommand.addListener(function (command) {
  if (command === 'paste-md') runOnActiveTab();
});

chrome.action.onClicked.addListener(function () {
  runOnActiveTab();
});

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (!message || message.type !== 'MD_PASTE_FETCH_IMAGE') return false;
  fetchImage(message.url).then(function (image) {
    sendResponse({ ok: true, image: image });
  }).catch(function (error) {
    sendResponse({ ok: false, error: String(error && error.message || error) });
  });
  return true;
});
