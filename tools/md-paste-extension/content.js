'use strict';

(function () {
  if (globalThis.__mdPasteContentInstalled) return;
  globalThis.__mdPasteContentInstalled = true;

  var parser = globalThis.MdPasteMarkdown;
  var toastTimer = null;

  function showToast(message, duration) {
    var host = document.getElementById('md-paste-extension-toast');
    if (!host) {
      host = document.createElement('div');
      host.id = 'md-paste-extension-toast';
      host.style.cssText = 'all:initial;position:fixed;z-index:2147483647;inset:auto 16px 20px auto;';
      document.documentElement.appendChild(host);
      host.attachShadow({ mode: 'open' });
      host.shadowRoot.innerHTML = '<style>' +
        ':host{all:initial} .toast{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;' +
        'color:#fff;background:#252a34;padding:10px 15px;border-radius:8px;box-shadow:0 4px 18px #0004;' +
        'max-width:min(420px,calc(100vw - 32px));white-space:pre-wrap}' +
        '</style><div class="toast" role="status" aria-live="polite"></div>';
    }
    host.shadowRoot.querySelector('.toast').textContent = message;
    if (toastTimer) clearTimeout(toastTimer);
    if (duration) {
      toastTimer = setTimeout(function () {
        host.remove();
        toastTimer = null;
      }, duration);
    }
  }

  function activeEditable() {
    var active = document.activeElement;
    if (!active) return null;
    if (active.matches('textarea, input')) {
      var type = (active.getAttribute('type') || 'text').toLowerCase();
      if (active.disabled || active.readOnly ||
          /^(button|checkbox|color|file|hidden|image|radio|range|reset|submit)$/.test(type)) return null;
      return active;
    }
    if (active.isContentEditable) return active;
    var editable = active.closest('[contenteditable="true"], [contenteditable=""]');
    return editable && editable.isContentEditable ? editable : null;
  }

  function makeDataTransfer(text, file) {
    var transfer = new DataTransfer();
    if (file) transfer.items.add(file);
    else transfer.setData('text/plain', text);
    return transfer;
  }

  function dispatchPaste(target, transfer) {
    var event;
    try {
      event = new ClipboardEvent('paste', {
        clipboardData: transfer,
        bubbles: true,
        cancelable: true,
        composed: true
      });
    } catch (_) {
      event = new Event('paste', { bubbles: true, cancelable: true });
      Object.defineProperty(event, 'clipboardData', { value: transfer });
    }
    if (!event.clipboardData) {
      try { Object.defineProperty(event, 'clipboardData', { value: transfer }); } catch (_) {}
    }
    target.dispatchEvent(event);
    return event.defaultPrevented;
  }

  function insertText(target, text) {
    target.focus({ preventScroll: true });
    var prevented = dispatchPaste(target, makeDataTransfer(text, null));
    if (!prevented) document.execCommand('insertText', false, text);
  }

  function rootFor(target) {
    return target.closest('.CodeMirror, .cm-editor, .monaco-editor, .ProseMirror, [contenteditable="true"]') || target;
  }

  function observeEditor(target) {
    var root = rootFor(target);
    var lastValue = readEditorValue(target, root);
    var changed = false;
    var lastChange = 0;
    var started = Date.now();
    var observer = null;
    var interval = null;
    var onInput = function () {
      changed = true;
      lastChange = Date.now();
    };

    function checkValue() {
      var current = readEditorValue(target, root);
      if (current !== lastValue) {
        lastValue = current;
        onInput();
      }
    }

    function finish(result, resolve) {
      if (interval) clearInterval(interval);
      if (observer) observer.disconnect();
      root.removeEventListener('input', onInput, true);
      resolve(result);
    }

    if (typeof MutationObserver === 'function') {
      observer = new MutationObserver(onInput);
      observer.observe(root, { subtree: true, childList: true, characterData: true, attributes: true });
    }
    root.addEventListener('input', onInput, true);
    interval = setInterval(checkValue, 100);

    return {
      wait: function () {
        return new Promise(function (resolve) {
          function tick() {
            checkValue();
            var elapsed = Date.now() - started;
            if (!changed && elapsed >= 8000) {
              finish('unconfirmed', resolve);
            } else if (elapsed >= 30000) {
              finish('unconfirmed', resolve);
            } else if (changed && Date.now() - lastChange >= 1500) {
              finish('confirmed', resolve);
            } else {
              setTimeout(tick, 100);
            }
          }
          tick();
        });
      },
      cancel: function () {
        if (interval) clearInterval(interval);
        if (observer) observer.disconnect();
        root.removeEventListener('input', onInput, true);
      }
    };
  }

  function readEditorValue(target, root) {
    if (target.matches('textarea, input')) return target.value;
    return root.textContent;
  }

  function fileFromImage(image) {
    var binary = atob(image.base64);
    var bytes = new Uint8Array(binary.length);
    for (var index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return new File([bytes], image.name, { type: image.mime, lastModified: Date.now() });
  }

  function dispatchDrop(target, transfer) {
    var events = ['dragenter', 'dragover', 'drop'].map(function (type) {
      var event = new DragEvent(type, {
        dataTransfer: transfer,
        bubbles: true,
        cancelable: true,
        composed: true
      });
      target.dispatchEvent(event);
      return event;
    });
    return events.some(function (event) { return event.defaultPrevented; });
  }

  function looksLikeMarkdown(text) {
    return /(^|\n)\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|```|~~~)|\[[^\]]+\]\([^)]*\)|`[^`]+`|(?:\*\*|__|~~)|!\[|<img\b[^>]*\bsrc\s*=/i.test(text);
  }

  async function pasteMarkdown(target, text) {
    if (!parser || typeof text !== 'string' || !text.trim() || !looksLikeMarkdown(text)) {
      showToast('剪贴板里没有可粘贴的 Markdown。', 5000);
      return { status: 'empty' };
    }

    var allImageTokens = parser.splitMarkdown(text, parser.isHttpUrl)
      .filter(function (token) { return token.type === 'image'; });
    var externalCount = allImageTokens.filter(function (token) {
      return !parser.isAllowedBlogImageUrl(token.url);
    }).length;
    var tokens = parser.splitMarkdown(text, parser.isAllowedBlogImageUrl);
    var imageCount = tokens.filter(function (token) { return token.type === 'image'; }).length;
    var uploaded = 0;
    var unconfirmed = 0;
    var retained = externalCount;
    var imageIndex = 0;

    for (var index = 0; index < tokens.length; index += 1) {
      var token = tokens[index];
      target.focus({ preventScroll: true });
      if (token.type === 'text') {
        insertText(target, token.text);
        continue;
      }

      imageIndex += 1;
      showToast('上传图片 ' + imageIndex + '/' + imageCount + '…');
      var result;
      try {
        result = await chrome.runtime.sendMessage({ type: 'MD_PASTE_FETCH_IMAGE', url: token.url });
      } catch (error) {
        result = { ok: false, error: String(error && error.message || error) };
      }
      if (!result || !result.ok || !result.image) {
        insertText(target, token.raw);
        retained += 1;
        continue;
      }

      var watcher = observeEditor(target);
      var transfer;
      var handled = false;
      try {
        transfer = makeDataTransfer('', fileFromImage(result.image));
        handled = dispatchPaste(target, transfer);
        if (!handled) handled = dispatchDrop(target, transfer);
      } catch (_) {
        handled = false;
      }

      if (!handled) {
        watcher.cancel();
        insertText(target, token.raw);
        retained += 1;
        continue;
      }

      if (await watcher.wait() === 'confirmed') uploaded += 1;
      else unconfirmed += 1;
    }

    showToast(
      '完成：' + uploaded + ' 张已上传，' + unconfirmed + ' 张未确认，' + retained + ' 张保留外链',
      6000
    );
    return {
      status: 'done',
      uploaded: uploaded,
      unconfirmed: unconfirmed,
      retained: retained,
      images: imageCount
    };
  }

  chrome.runtime.onMessage.addListener(function (message, sender, sendResponse) {
    if (!message || message.type !== 'MD_PASTE_START') return false;
    if (!document.hasFocus()) {
      sendResponse({ status: 'ignored' });
      return false;
    }
    var target = activeEditable();
    if (!target) {
      showToast('请先点击编辑器中的可编辑区域，再运行粘贴。', 5000);
      sendResponse({ status: 'no-editable' });
      return false;
    }
    if (message.clipboardError) {
      showToast('读取剪贴板失败：' + message.clipboardError, 6000);
      sendResponse({ status: 'clipboard-error' });
      return false;
    }
    pasteMarkdown(target, message.text).then(sendResponse).catch(function (error) {
      showToast('粘贴失败：' + String(error && error.message || error), 6000);
      sendResponse({ status: 'error', error: String(error && error.message || error) });
    });
    return true;
  });
})();
