(function () {
  'use strict';

  var ARTICLE_SELECTOR = '.post-container:not(.moments):not(.moments-comments)';
  var article = document.querySelector(ARTICLE_SELECTOR);
  if (!article) return;

  var root = document.documentElement;
  root.classList.add('rs-focus-page');

  var item = document.querySelector('.nav-reading-settings');
  var trigger = item && item.querySelector('a');
  if (!item || !trigger) return;
  var activeTrigger = trigger;

  var focusItem = document.querySelector('.nav-focus-mode');
  var focusTrigger = focusItem && focusItem.querySelector('a');
  var ownFullscreen = false;
  var pendingFullscreen = false;
  var prevFs = document.fullscreenElement || document.webkitFullscreenElement;
  var unloading = false;

  function fullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement;
  }

  function lockEscape() {
    if (!navigator.keyboard || !navigator.keyboard.lock) return;
    try {
      var request = navigator.keyboard.lock(['Escape']);
      if (request && typeof request.catch === 'function') request.catch(function () {});
    } catch (e) {}
  }

  function unlockKeyboard() {
    if (!navigator.keyboard || !navigator.keyboard.unlock) return;
    try { navigator.keyboard.unlock(); } catch (e) {}
  }

  function enterFullscreen() {
    var el = document.documentElement;
    var request = el.requestFullscreen || el.webkitRequestFullscreen;
    if (!request || fullscreenElement() || pendingFullscreen) return;
    pendingFullscreen = true;
    try {
      var promise = request.call(el);
      if (promise && typeof promise.catch === 'function') {
        promise.catch(function () {
          pendingFullscreen = false;
        });
      }
    } catch (e) {
      pendingFullscreen = false;
    }
  }

  function exitFullscreen() {
    ownFullscreen = false;
    unlockKeyboard();
    var exit = document.exitFullscreen || document.webkitExitFullscreen;
    if (!exit) return;
    try {
      var promise = exit.call(document);
      if (promise && typeof promise.catch === 'function') promise.catch(function () {});
    } catch (e) {}
  }

  function onFullscreenChange() {
    var currentFs = fullscreenElement();
    var previousFs = prevFs;
    prevFs = currentFs;

    if (currentFs === root && pendingFullscreen) {
      pendingFullscreen = false;
      ownFullscreen = true;
    }
    if (currentFs === root && ownFullscreen) lockEscape();
    else unlockKeyboard();

    if (previousFs === root && ownFullscreen && !currentFs) {
      ownFullscreen = false;
      if (!unloading && apply('focus', '')) changed();
    }
    if (currentFs === root && ownFullscreen && selected('focus') !== 'on') exitFullscreen();
  }

  var panel = document.createElement('div');
  panel.id = 'reading-settings-panel';
  panel.className = 'reading-settings-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', '阅读设置');
  panel.hidden = true;
  panel.innerHTML =
    '<div class="rs-setting-row rs-paper-row"><span class="rs-setting-label">配色</span><div class="rs-segments" role="group" aria-label="配色">' +
      '<button type="button" data-rs-paper="light" aria-pressed="false">白</button>' +
      '<button type="button" data-rs-paper="sepia" aria-pressed="false">米黄</button>' +
      '<button type="button" data-rs-paper="dark" aria-pressed="false">深色</button>' +
    '</div></div>' +
    '<div class="rs-setting-row"><span class="rs-setting-label">字号</span><div class="rs-segments" role="group" aria-label="字号">' +
      '<button type="button" data-rs-setting="size" data-rs-value="s" aria-pressed="false">小</button>' +
      '<button type="button" data-rs-setting="size" data-rs-value="" aria-pressed="false">标准</button>' +
      '<button type="button" data-rs-setting="size" data-rs-value="l" aria-pressed="false">大</button>' +
      '<button type="button" data-rs-setting="size" data-rs-value="xl" aria-pressed="false">特大</button>' +
    '</div></div>' +
    '<div class="rs-setting-row"><span class="rs-setting-label">宽度</span><div class="rs-segments" role="group" aria-label="宽度">' +
      '<button type="button" data-rs-setting="width" data-rs-value="" aria-pressed="false">默认</button>' +
      '<button type="button" data-rs-setting="width" data-rs-value="narrow" aria-pressed="false">窄栏</button>' +
    '</div></div>' +
    '<div class="rs-setting-row"><span class="rs-setting-label">字体</span><div class="rs-segments" role="group" aria-label="字体">' +
      '<button type="button" data-rs-setting="font" data-rs-value="" aria-pressed="false">无衬线</button>' +
      '<button type="button" data-rs-setting="font" data-rs-value="serif" aria-pressed="false">衬线</button>' +
    '</div></div>' +
    '<button type="button" class="rs-reset">恢复默认</button>';
  document.body.appendChild(panel);

  var helpCard = document.createElement('div');
  helpCard.className = 'rs-keys-help';
  helpCard.setAttribute('role', 'dialog');
  helpCard.setAttribute('aria-label', '键盘快捷键');
  helpCard.hidden = true;
  helpCard.innerHTML =
    '<h2>键盘快捷键</h2>' +
    '<div class="rs-key-row"><span><kbd>j</kbd> / <kbd>k</kbd></span><span>向下 / 向上滚动</span></div>' +
    '<div class="rs-key-row"><span><kbd>d</kbd> / <kbd>u</kbd></span><span>向下 / 向上半屏</span></div>' +
    '<div class="rs-key-row"><span><kbd>gg</kbd> / <kbd>G</kbd></span><span>回到开头 / 跳到结尾</span></div>' +
    '<div class="rs-key-row"><span><kbd>h</kbd> / <kbd>l</kbd></span><span>上一节 / 下一节</span></div>' +
    '<div class="rs-key-row"><span><kbd>空格</kbd> / <kbd>Shift</kbd>+<kbd>空格</kbd></span><span>整屏翻页</span></div>' +
    '<div class="rs-key-row"><span><kbd>?</kbd></span><span>显示 / 隐藏本帮助</span></div>' +
    '<div class="rs-key-row"><span><kbd>Esc</kbd></span><span>退出专注</span></div>';
  document.body.appendChild(helpCard);

  var focusControls = document.createElement('div');
  focusControls.className = 'rs-focus-controls';
  focusControls.hidden = true;

  var settingsButton = document.createElement('button');
  settingsButton.type = 'button';
  settingsButton.className = 'rs-focus-settings';
  settingsButton.textContent = 'Aa';
  settingsButton.setAttribute('aria-label', '阅读设置');
  settingsButton.setAttribute('title', '阅读设置（字号、宽度、字体、配色）');
  settingsButton.setAttribute('aria-haspopup', 'dialog');
  settingsButton.setAttribute('aria-controls', 'reading-settings-panel');
  settingsButton.setAttribute('aria-expanded', 'false');
  focusControls.appendChild(settingsButton);

  var exitButton = document.createElement('button');
  exitButton.type = 'button';
  exitButton.className = 'rs-focus-exit';
  exitButton.textContent = '退出专注 · Esc';
  exitButton.setAttribute('aria-label', '退出专注');
  exitButton.setAttribute('title', '退出专注（Esc）· 按 ? 查看快捷键');
  focusControls.appendChild(exitButton);
  document.body.appendChild(focusControls);

  var attributes = { size: 'data-rs-size', width: 'data-rs-width', font: 'data-rs-font', focus: 'data-rs-focus' };
  var values = {
    size: ['', 's', 'l', 'xl'],
    width: ['', 'narrow'],
    font: ['', 'serif'],
    focus: ['', 'on']
  };

  function selected(name) {
    return root.getAttribute(attributes[name]) || '';
  }

  function syncControls() {
    var buttons = panel.querySelectorAll('[data-rs-setting]');
    for (var i = 0; i < buttons.length; i++) {
      var button = buttons[i];
      button.setAttribute('aria-pressed', String(button.getAttribute('data-rs-value') === selected(button.getAttribute('data-rs-setting'))));
    }
    if (focusTrigger) focusTrigger.setAttribute('aria-pressed', String(selected('focus') === 'on'));
    var focusOn = selected('focus') === 'on';
    focusControls.hidden = !focusOn;
    if (!focusOn) helpCard.hidden = true;
    if (selected('focus') !== 'on' && activeTrigger === settingsButton && !panel.hidden) close(false);
    var theme = root.getAttribute('data-theme') || 'light';
    var paperButtons = panel.querySelectorAll('[data-rs-paper]');
    for (var j = 0; j < paperButtons.length; j++) {
      paperButtons[j].setAttribute('aria-pressed', String(paperButtons[j].getAttribute('data-rs-paper') === theme));
    }
  }

  function store() {
    var settings = {};
    Object.keys(attributes).forEach(function (name) {
      var value = selected(name);
      if (values[name].indexOf(value) > 0) settings[name] = value;
    });
    try {
      if (Object.keys(settings).length) localStorage.setItem('reading-settings', JSON.stringify(settings));
      else localStorage.removeItem('reading-settings');
    } catch (e) {}
  }

  function apply(name, value) {
    if (values[name].indexOf(value) < 0 || selected(name) === value) return false;
    if (value) root.setAttribute(attributes[name], value);
    else root.removeAttribute(attributes[name]);
    return true;
  }

  function changed() {
    store();
    syncControls();
    window.dispatchEvent(new Event('reading-settings-change'));
    window.dispatchEvent(new Event('resize'));
  }

  function positionPanel() {
    if (panel.hidden || window.innerWidth <= 767) return;
    var rect = activeTrigger.getBoundingClientRect();
    panel.style.top = Math.round(rect.bottom + 8) + 'px';
    panel.style.right = Math.max(12, window.innerWidth - rect.right) + 'px';
  }

  function openPanel(anchor) {
    activeTrigger = anchor;
    helpCard.hidden = true;
    panel.hidden = false;
    activeTrigger.setAttribute('aria-expanded', 'true');
    positionPanel();
  }

  function close(restoreFocus) {
    panel.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
    settingsButton.setAttribute('aria-expanded', 'false');
    if (restoreFocus) activeTrigger.focus();
  }

  trigger.addEventListener('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    if (panel.hidden) {
      if (window.innerWidth <= 767 && window.__HuxNav__) window.__HuxNav__.close();
      openPanel(trigger);
    } else {
      close(false);
    }
  });

  settingsButton.addEventListener('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    if (panel.hidden) openPanel(settingsButton);
    else close(false);
  });

  if (focusTrigger) {
    focusTrigger.addEventListener('click', function (event) {
      event.preventDefault();
      event.stopPropagation();
      if (window.innerWidth <= 767 && window.__HuxNav__) window.__HuxNav__.close();
      if (!panel.hidden) close(false);
      if (apply('focus', 'on')) {
        changed();
        enterFullscreen();
      }
    });
  }

  panel.addEventListener('click', function (event) {
    event.stopPropagation();
    var button = event.target.closest('button');
    if (!button) return;
    if (button.classList.contains('rs-reset')) {
      var didChange = false;
      Object.keys(attributes).forEach(function (name) {
        if (name !== 'focus') didChange = apply(name, '') || didChange;
      });
      if (didChange) changed();
      if (root.getAttribute('data-theme') === 'sepia') {
        if (typeof window.setTheme === 'function') window.setTheme('light');
      } else if (root.getAttribute('data-theme') === 'dark') {
        try { localStorage.removeItem('theme-day'); } catch (e) {}
      }
      syncControls();
      return;
    }
    var paper = button.getAttribute('data-rs-paper');
    if (paper) {
      if (typeof window.setTheme === 'function') window.setTheme(paper);
      return;
    }
    var name = button.getAttribute('data-rs-setting');
    var value = button.getAttribute('data-rs-value');
    if (name && apply(name, value)) {
      if (name === 'focus' && value === 'on') close(false);
      changed();
    }
  });

  exitButton.addEventListener('click', function () {
    if (apply('focus', '')) changed();
  });

  function visible(element) {
    if (!element || element.hidden) return false;
    var style = window.getComputedStyle(element);
    return style.display !== 'none' && style.visibility !== 'hidden' && element.getClientRects().length > 0;
  }

  function otherOverlayOpen() {
    if (document.querySelector('.mm-overlay, .diagram-zoom.open, .search-overlay.search-active')) return true;
    var overlays = document.querySelectorAll('.annotation-panel, .annotation-comments');
    for (var i = 0; i < overlays.length; i++) {
      if (visible(overlays[i])) return true;
    }
    return false;
  }

  function overlayOpen() {
    return !!document.querySelector('.rs-keys-help:not([hidden])') || otherOverlayOpen();
  }

  function isEditing(element) {
    return !!(element && (element.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(element.tagName) ||
      element.closest && element.closest('[contenteditable]:not([contenteditable="false"])')));
  }

  function canReenterFullscreen() {
    return selected('focus') === 'on' && !fullscreenElement() &&
      !!(root.requestFullscreen || root.webkitRequestFullscreen) && !overlayOpen() && panel.hidden;
  }

  document.addEventListener('click', function (event) {
    if (!canReenterFullscreen()) return;
    var target = event.target;
    if (target && target.nodeType !== 1) target = target.parentElement;
    if (target && target.closest && target.closest(
      'a, button, input, textarea, select, label, summary, video, audio, [contenteditable]:not([contenteditable="false"]), [role="button"], [tabindex]'
    )) return;
    var selection = window.getSelection && window.getSelection();
    if (selection && !selection.isCollapsed) return;
    enterFullscreen();
  }, true);

  document.addEventListener('keydown', function (event) {
    if (!canReenterFullscreen() || event.metaKey || event.ctrlKey || event.altKey ||
        event.defaultPrevented || event.isComposing) return;
    if (event.key !== ' ' && event.key !== 'PageDown' && event.key !== 'PageUp' &&
        event.key !== 'ArrowDown' && event.key !== 'ArrowUp' && event.key !== 'Home' && event.key !== 'End' &&
        event.key !== 'j' && event.key !== 'k' && event.key !== 'd' && event.key !== 'u' &&
        event.key !== 'G' && event.key !== 'h' && event.key !== 'l' && event.code !== 'Space') return;
    if (isEditing(document.activeElement)) return;
    enterFullscreen();
  }, true);

  var pendingG = 0;
  document.addEventListener('keydown', function (event) {
    var isG = event.key === 'g';
    if (!isG) pendingG = 0;
    if (selected('focus') !== 'on') {
      pendingG = 0;
      return;
    }

    var isHelpKey = event.key === '?';
    var isEscape = event.key === 'Escape' || event.key === 'Esc';
    if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented || event.isComposing ||
        (event.shiftKey && event.key !== 'G' && !isHelpKey) || isEditing(document.activeElement) || !panel.hidden) {
      if (isG) pendingG = 0;
      return;
    }
    if (overlayOpen() && !(isHelpKey && !otherOverlayOpen())) return;

    if (isHelpKey) {
      event.preventDefault();
      helpCard.hidden = !helpCard.hidden;
      return;
    }
    if (isEscape) return;

    var behavior = event.repeat || (window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches) ? 'auto' : 'smooth';
    var delta;
    if (event.key === 'j' || event.key === 'k') {
      var lineHeight = parseFloat(window.getComputedStyle(article).lineHeight);
      delta = lineHeight > 0 && isFinite(lineHeight) ? lineHeight * 3 : 80;
      if (event.key === 'k') delta = -delta;
      window.scrollBy({ top: delta, behavior: behavior });
    } else if (event.key === 'd' || event.key === 'u') {
      delta = window.innerHeight / 2;
      if (event.key === 'u') delta = -delta;
      window.scrollBy({ top: delta, behavior: behavior });
    } else if (event.key === 'g') {
      if (event.repeat) {
        pendingG = 0;
        return;
      }
      var now = Date.now();
      if (pendingG && now - pendingG <= 1000) {
        pendingG = 0;
        window.scrollTo({ top: 0, behavior: behavior });
      } else {
        pendingG = now;
      }
    } else if (event.key === 'G') {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: behavior });
    } else if (event.key === 'h' || event.key === 'l') {
      var headings = article.querySelectorAll('h2, h3');
      var target = null;
      for (var i = 0; i < headings.length; i++) {
        if (!headings[i].getClientRects().length) continue;
        var headingTop = headings[i].getBoundingClientRect().top;
        if (event.key === 'l' && headingTop > 24) {
          target = headings[i];
          break;
        }
        if (event.key === 'h' && headingTop < -8) target = headings[i];
      }
      if (target) {
        window.scrollTo({
          top: window.scrollY + target.getBoundingClientRect().top - 16,
          behavior: behavior
        });
      }
    } else {
      return;
    }
    event.preventDefault();
  });

  var escapeOverlayWasOpen = false;
  document.addEventListener('keydown', function (event) {
    escapeOverlayWasOpen = (event.key === 'Escape' || event.key === 'Esc') && overlayOpen();
  }, true);

  document.addEventListener('click', function (event) {
    if (!helpCard.hidden && !helpCard.contains(event.target)) helpCard.hidden = true;
  });
  document.addEventListener('click', function (event) {
    if (!panel.hidden && !panel.contains(event.target) && !trigger.contains(event.target) &&
        !settingsButton.contains(event.target)) close(true);
  });
  document.addEventListener('keydown', function (event) {
    if (!helpCard.hidden && (event.key === 'Escape' || event.key === 'Esc')) {
      helpCard.hidden = true;
      event.preventDefault();
    }
    if (!panel.hidden && (event.key === 'Escape' || event.key === 'Esc')) {
      event.preventDefault();
      close(true);
    }
    if (event.key !== 'Escape' && event.key !== 'Esc') return;
    var active = document.activeElement;
    if (panel.hidden && !event.defaultPrevented && !isEditing(active) && !escapeOverlayWasOpen && selected('focus') === 'on' &&
        apply('focus', '')) changed();
    escapeOverlayWasOpen = false;
  });
  window.addEventListener('resize', positionPanel);
  window.addEventListener('scroll', positionPanel, true);
  window.addEventListener('pagehide', function () { unloading = true; });
  window.addEventListener('beforeunload', function () { unloading = true; });
  window.addEventListener('pageshow', function (event) { if (event.persisted) unloading = false; });
  window.addEventListener('reading-settings-change', function () {
    syncControls();
    if (selected('focus') !== 'on' && ownFullscreen && fullscreenElement() === root) exitFullscreen();
  });
  document.addEventListener('fullscreenchange', onFullscreenChange);
  document.addEventListener('webkitfullscreenchange', onFullscreenChange);
  if (window.MutationObserver) {
    new MutationObserver(syncControls).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  }

  item.hidden = false;
  if (focusItem) focusItem.hidden = false;
  syncControls();
})();
