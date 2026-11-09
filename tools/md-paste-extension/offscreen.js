'use strict';

function fromBase64(value) {
  var binary = atob(value);
  var bytes = new Uint8Array(binary.length);
  for (var index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function toBase64(bytes) {
  var binary = '';
  var chunkSize = 0x8000;
  for (var offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function pngFilename(name) {
  var stem = String(name || 'image').replace(/\.[^.]*$/, '') || 'image';
  return stem + '.png';
}

function readClipboard() {
  var textarea = document.getElementById('clipboard');
  textarea.value = '';
  var pastedText = null;
  var pastedHtml = '';
  function onPaste(event) {
    var clipboard = event.clipboardData;
    pastedText = clipboard ? clipboard.getData('text/plain') : '';
    pastedHtml = clipboard ? clipboard.getData('text/html') || '' : '';
    event.preventDefault();
  }
  textarea.addEventListener('paste', onPaste, { once: true });
  textarea.focus();
  var executed = document.execCommand('paste');
  textarea.removeEventListener('paste', onPaste);
  var text = pastedText == null ? textarea.value : pastedText;

  if (!executed && pastedText == null && navigator.clipboard && navigator.clipboard.readText) {
    return navigator.clipboard.readText().then(function (clipboardText) {
      return { ok: true, text: clipboardText || '', html: '' };
    });
  }
  return Promise.resolve({
    ok: pastedText != null || executed || !!text,
    text: text || '',
    html: pastedHtml || ''
  });
}

function convertImage(image) {
  var mime = String(image.mime || '').toLowerCase();
  if (mime === 'image/png' || mime === 'image/jpeg' || mime === 'image/gif') {
    return Promise.resolve({ base64: image.base64, mime: mime, name: image.name });
  }

  var blob = new Blob([fromBase64(image.base64)], { type: mime || 'application/octet-stream' });
  var source = URL.createObjectURL(blob);
  return new Promise(function (resolve, reject) {
    var bitmap = new Image();
    bitmap.onload = function () {
      try {
        var width = bitmap.naturalWidth || bitmap.width;
        var height = bitmap.naturalHeight || bitmap.height;
        if (!width || !height) throw new Error('无法读取图片尺寸');
        var canvas = document.createElement('canvas');
        canvas.width = width * 2;
        canvas.height = height * 2;
        var context = canvas.getContext('2d');
        if (!context) throw new Error('无法创建图片画布');
        context.fillStyle = '#fff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function (png) {
          URL.revokeObjectURL(source);
          if (!png) {
            reject(new Error('无法编码 PNG'));
            return;
          }
          png.arrayBuffer().then(function (buffer) {
            resolve({
              base64: toBase64(new Uint8Array(buffer)),
              mime: 'image/png',
              name: pngFilename(image.name)
            });
          }, reject);
        }, 'image/png');
      } catch (error) {
        URL.revokeObjectURL(source);
        reject(error);
      }
    };
    bitmap.onerror = function () {
      URL.revokeObjectURL(source);
      reject(new Error('无法解码图片'));
    };
    bitmap.src = source;
  });
}

chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
  if (!message || message.target !== 'md-paste-offscreen') return false;
  if (message.action === 'read-clipboard') {
    readClipboard().then(sendResponse).catch(function (error) {
      sendResponse({ ok: false, error: String(error && error.message || error) });
    });
    return true;
  }
  if (message.action === 'convert-image') {
    convertImage(message.image).then(function (image) {
      sendResponse({ ok: true, image: image });
    }).catch(function (error) {
      sendResponse({ ok: false, error: String(error && error.message || error) });
    });
    return true;
  }
  return false;
});
