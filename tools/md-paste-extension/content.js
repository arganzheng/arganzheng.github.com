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
    var closed = false;
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

    function cleanup() {
      if (closed) return;
      closed = true;
      if (interval) clearInterval(interval);
      if (observer) observer.disconnect();
      root.removeEventListener('input', onInput, true);
    }

    return {
      waitForChange: function (timeout) {
        return new Promise(function (resolve) {
          var deadline = Date.now() + timeout;
          function tick() {
            if (closed) {
              resolve(false);
              return;
            }
            checkValue();
            if (changed) {
              resolve(true);
            } else if (Date.now() >= deadline) {
              resolve(false);
            } else {
              setTimeout(tick, 50);
            }
          }
          tick();
        });
      },
      waitForQuiet: function () {
        return new Promise(function (resolve) {
          function tick() {
            if (closed) {
              resolve('unconfirmed');
              return;
            }
            checkValue();
            if (Date.now() - started >= 30000) {
              cleanup();
              resolve('unconfirmed');
            } else if (changed && Date.now() - lastChange >= 1500) {
              cleanup();
              resolve('confirmed');
            } else {
              setTimeout(tick, 100);
            }
          }
          tick();
        });
      },
      cancel: function () {
        cleanup();
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
    var dropEvent = null;
    ['dragenter', 'dragover', 'drop'].forEach(function (type) {
      var event = new DragEvent(type, {
        dataTransfer: transfer,
        bubbles: true,
        cancelable: true,
        composed: true
      });
      target.dispatchEvent(event);
      if (type === 'drop') dropEvent = event;
    });
    return !!(dropEvent && dropEvent.defaultPrevented);
  }

  async function tryImageDispatch(target, dispatch) {
    var watcher = observeEditor(target);
    var prevented = false;
    var error = '';
    try {
      target.focus({ preventScroll: true });
      prevented = dispatch();
    } catch (dispatchError) {
      error = String(dispatchError && dispatchError.message || dispatchError);
    }
    var changed = await watcher.waitForChange(8000);
    if (prevented || changed) {
      if (changed) return { handled: true, result: await watcher.waitForQuiet(), error: error };
      watcher.cancel();
      return { handled: true, result: 'unconfirmed', error: error };
    }
    watcher.cancel();
    return { handled: false, error: error };
  }

  async function tryTrustedPaste(target, file) {
    var clipboardWritten = false;
    try {
      if (!navigator.clipboard || typeof navigator.clipboard.write !== 'function') {
        throw new Error('浏览器不支持图片剪贴板写入');
      }
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': file })]);
      clipboardWritten = true;
    } catch (error) {
      return {
        handled: false,
        clipboardWritten: clipboardWritten,
        blockedError: String(error && error.message || error)
      };
    }

    var watcher = null;
    var executed = false;
    var error = '';
    var pasteEvent = null;
    function captureTrustedPaste(event) {
      if (event.isTrusted) pasteEvent = event;
    }
    try {
      target.focus({ preventScroll: true });
      watcher = observeEditor(target);
      target.addEventListener('paste', captureTrustedPaste, true);
      executed = document.execCommand('paste');
    } catch (pasteError) {
      error = String(pasteError && pasteError.message || pasteError);
    } finally {
      target.removeEventListener('paste', captureTrustedPaste, true);
    }
    if (!watcher) {
      return {
        handled: false,
        clipboardWritten: clipboardWritten,
        blockedError: error || '无法聚焦编辑器执行粘贴'
      };
    }
    var changed = await watcher.waitForChange(8000);
    if (changed) {
      return {
        handled: true,
        clipboardWritten: clipboardWritten,
        result: await watcher.waitForQuiet(),
        error: error
      };
    }
    if (pasteEvent && pasteEvent.defaultPrevented) {
      watcher.cancel();
      return {
        handled: true,
        clipboardWritten: clipboardWritten,
        result: 'unconfirmed',
        error: error
      };
    }
    watcher.cancel();
    return {
      handled: false,
      clipboardWritten: clipboardWritten,
      blockedError: error || (!executed ? '浏览器拒绝执行 document.execCommand("paste")' : '')
    };
  }

  function logImage(token, path, result, error) {
    console.info('[md-paste]', {
      url: token.url,
      path: path,
      result: result,
      error: error || ''
    });
  }

  function looksLikeMarkdown(text) {
    return /(^|\n)\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|```|~~~|\[[^\]\n]+\]:\s*(?:<[^>\n]+>|\S+))|\[[^\]]+\]\([^)]*\)|\[[^\]]+\]\[[^\]]*\]|`[^`]+`|(?:\*\*|__|~~)|!\[|<img\b[^>]*\bsrc\s*=/i.test(text);
  }

  async function pasteMarkdown(target, text, hasHtmlImages) {
    if (!parser || typeof text !== 'string' || !text.trim() || !looksLikeMarkdown(text)) {
      if (hasHtmlImages) {
        showToast('剪贴板里是富文本，图片不在 Markdown 里。请在博客导出菜单点「复制 Markdown」后再试。', 8000);
        return { status: 'rich-text' };
      }
      showToast('剪贴板里没有可粘贴的 Markdown。', 5000);
      return { status: 'empty' };
    }

    var allImageTokens = parser.splitMarkdown(text, parser.isHttpUrl)
      .filter(function (token) { return token.type === 'image'; });
    var externalCount = allImageTokens.filter(function (token) {
      return !parser.isAllowedBlogImageUrl(token.url);
    }).length;
    allImageTokens.filter(function (token) {
      return !parser.isAllowedBlogImageUrl(token.url);
    }).forEach(function (token) {
      logImage(token, 'failed', 'retained', '非博客图片');
    });
    var tokens = parser.splitMarkdown(text, parser.isAllowedBlogImageUrl);
    var imageCount = tokens.filter(function (token) { return token.type === 'image'; }).length;
    if (!imageCount && !externalCount) {
      if (hasHtmlImages) {
        showToast('剪贴板里是富文本，图片不在 Markdown 里。请在博客导出菜单点「复制 Markdown」后再试。', 8000);
        return { status: 'rich-text' };
      }
      var rewrittenText = parser.relativizeBlogLinks(text);
      insertText(target, rewrittenText.text);
      var linkSuffix = rewrittenText.count ?
        '，' + rewrittenText.count + ' 个站内链接改成了相对路径' : '';
      showToast('已粘贴 ' + text.length + ' 字，里面没有博客图片链接。原文有图的话，请用博客的「复制 Markdown」。' + linkSuffix, 8000);
      return { status: 'text-only', chars: text.length, relativeLinks: rewrittenText.count };
    }

    var uploaded = 0;
    var unconfirmed = 0;
    var retained = externalCount;
    var relativeLinks = 0;
    var imageIndex = 0;
    var usedTrustedPaste = false;
    var reasons = {
      download: 0,
      downloadError: '',
      blocked: 0,
      blockedError: '',
      unhandled: 0
    };

    for (var index = 0; index < tokens.length; index += 1) {
      var token = tokens[index];
      target.focus({ preventScroll: true });
      if (token.type === 'text') {
        var rewrittenToken = parser.relativizeBlogLinks(token.text);
        relativeLinks += rewrittenToken.count;
        insertText(target, rewrittenToken.text);
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
        reasons.download += 1;
        var downloadError = String(result && result.error || '图片下载失败');
        if (!reasons.downloadError) {
          reasons.downloadError = downloadError;
        }
        logImage(token, 'failed', 'retained', downloadError);
        continue;
      }

      var transfer;
      var file;
      try {
        file = fileFromImage(result.image);
        transfer = makeDataTransfer('', file);
      } catch (error) {
        var fileError = String(error && error.message || error);
        insertText(target, token.raw);
        retained += 1;
        reasons.blocked += 1;
        if (!reasons.blockedError) reasons.blockedError = fileError;
        logImage(token, 'failed', 'retained', fileError);
        continue;
      }

      var attempt = await tryImageDispatch(target, function () {
        return dispatchPaste(target, transfer);
      });
      if (attempt.handled) {
        logImage(token, 'synthetic-paste', attempt.result === 'confirmed' ? 'uploaded' : 'unconfirmed', attempt.error);
        if (attempt.result === 'confirmed') uploaded += 1;
        else unconfirmed += 1;
        continue;
      }

      attempt = await tryImageDispatch(target, function () {
        return dispatchDrop(target, transfer);
      });
      if (attempt.handled) {
        logImage(token, 'drop', attempt.result === 'confirmed' ? 'uploaded' : 'unconfirmed', attempt.error);
        if (attempt.result === 'confirmed') uploaded += 1;
        else unconfirmed += 1;
        continue;
      }

      var trustedAttempt = await tryTrustedPaste(target, file);
      if (trustedAttempt.clipboardWritten) usedTrustedPaste = true;
      if (trustedAttempt.handled) {
        logImage(token, 'trusted-paste', trustedAttempt.result === 'confirmed' ? 'uploaded' : 'unconfirmed', trustedAttempt.error);
        if (trustedAttempt.result === 'confirmed') uploaded += 1;
        else unconfirmed += 1;
        continue;
      }

      if (trustedAttempt.blockedError) {
        reasons.blocked += 1;
        if (!reasons.blockedError) reasons.blockedError = trustedAttempt.blockedError;
      } else {
        reasons.unhandled += 1;
      }
      var retainedError = trustedAttempt.blockedError || '编辑器没有接收';
      insertText(target, token.raw);
      retained += 1;
      logImage(token, 'failed', 'retained', retainedError);
    }

    if (usedTrustedPaste && navigator.clipboard && navigator.clipboard.writeText) {
      try { await navigator.clipboard.writeText(text); } catch (_) {}
    }

    var toast = '完成：' + uploaded + ' 张已上传，' + unconfirmed + ' 张未确认，' + retained + ' 张保留外链';
    if (relativeLinks) toast += '，' + relativeLinks + ' 个站内链接改成了相对路径';
    if (retained > 0) {
      var reasonParts = [];
      if (reasons.download) {
        reasonParts.push('下载失败 ' + reasons.download + ' 张（' + reasons.downloadError + '）');
      }
      if (reasons.blocked) {
        reasonParts.push('浏览器不允许插件粘贴 ' + reasons.blocked + ' 张（' + reasons.blockedError + '）');
      }
      if (reasons.unhandled) reasonParts.push('编辑器没有接收 ' + reasons.unhandled + ' 张');
      if (externalCount) reasonParts.push('非博客图片 ' + externalCount + ' 张');
      toast += '\n保留外链原因：' + reasonParts.join('；');
    }
    showToast(toast, 6000);
    return {
      status: 'done',
      uploaded: uploaded,
      unconfirmed: unconfirmed,
      retained: retained,
      images: imageCount,
      relativeLinks: relativeLinks
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
    pasteMarkdown(target, message.text, !!message.hasHtmlImages).then(sendResponse).catch(function (error) {
      showToast('粘贴失败：' + String(error && error.message || error), 6000);
      sendResponse({ status: 'error', error: String(error && error.message || error) });
    });
    return true;
  });
})();
