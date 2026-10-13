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

  var focusItem = document.querySelector('.nav-focus-mode');
  var focusTrigger = focusItem && focusItem.querySelector('a');
  var ownFullscreen = false;
  var pendingFullscreen = false;
  var prevFs = document.fullscreenElement || document.webkitFullscreenElement;

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
    if (!request || fullscreenElement()) return;
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
      if (apply('focus', '')) changed();
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
    '<div class="rs-setting-row rs-paper-row"><span class="rs-setting-label">底色</span><div class="rs-segments" role="group" aria-label="日间底色">' +
      '<button type="button" data-rs-paper="light" aria-pressed="false">白</button>' +
      '<button type="button" data-rs-paper="sepia" aria-pressed="false">米黄</button>' +
    '</div></div>' +
    '<p class="rs-paper-note" hidden>深色模式下不适用</p>' +
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

  var exitButton = document.createElement('button');
  exitButton.type = 'button';
  exitButton.className = 'rs-focus-exit';
  exitButton.textContent = '退出专注 · Esc';
  exitButton.setAttribute('aria-label', '退出专注');
  exitButton.hidden = true;
  document.body.appendChild(exitButton);

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

  function dayPaper() {
    var t = root.getAttribute('data-theme');
    if (t === 'light' || t === 'sepia') return t;
    try { return localStorage.getItem('theme-day') === 'sepia' ? 'sepia' : 'light'; } catch (e) { return 'light'; }
  }

  function syncControls() {
    var buttons = panel.querySelectorAll('[data-rs-setting]');
    for (var i = 0; i < buttons.length; i++) {
      var button = buttons[i];
      button.setAttribute('aria-pressed', String(button.getAttribute('data-rs-value') === selected(button.getAttribute('data-rs-setting'))));
    }
    if (focusTrigger) focusTrigger.setAttribute('aria-pressed', String(selected('focus') === 'on'));
    exitButton.hidden = selected('focus') !== 'on';
    var dark = root.getAttribute('data-theme') === 'dark';
    var paperButtons = panel.querySelectorAll('[data-rs-paper]');
    for (var j = 0; j < paperButtons.length; j++) {
      paperButtons[j].disabled = dark;
      paperButtons[j].setAttribute('aria-pressed', String(paperButtons[j].getAttribute('data-rs-paper') === dayPaper()));
    }
    panel.querySelector('.rs-paper-note').hidden = !dark;
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
    var rect = trigger.getBoundingClientRect();
    panel.style.top = Math.round(rect.bottom + 8) + 'px';
    panel.style.right = Math.max(12, window.innerWidth - rect.right) + 'px';
  }

  function close(restoreFocus) {
    panel.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
    if (restoreFocus) trigger.focus();
  }

  trigger.addEventListener('click', function (event) {
    event.preventDefault();
    event.stopPropagation();
    if (panel.hidden) {
      if (window.innerWidth <= 767 && window.__HuxNav__) window.__HuxNav__.close();
      panel.hidden = false;
      trigger.setAttribute('aria-expanded', 'true');
      positionPanel();
    } else {
      close(false);
    }
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
      if (root.getAttribute('data-theme') !== 'dark' && typeof window.setTheme === 'function') window.setTheme(paper);
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

  function overlayOpen() {
    if (document.querySelector('.mm-overlay, .diagram-zoom.open, .search-overlay.search-active')) return true;
    var overlays = document.querySelectorAll('.annotation-panel, .annotation-comments');
    for (var i = 0; i < overlays.length; i++) {
      if (visible(overlays[i])) return true;
    }
    return false;
  }

  var escapeOverlayWasOpen = false;
  document.addEventListener('keydown', function (event) {
    escapeOverlayWasOpen = (event.key === 'Escape' || event.key === 'Esc') && overlayOpen();
  }, true);

  document.addEventListener('click', function (event) {
    if (!panel.hidden && !panel.contains(event.target) && !trigger.contains(event.target)) close(true);
  });
  document.addEventListener('keydown', function (event) {
    if (!panel.hidden && (event.key === 'Escape' || event.key === 'Esc')) {
      event.preventDefault();
      close(true);
    }
    if (event.key !== 'Escape' && event.key !== 'Esc') return;
    var active = document.activeElement;
    var editing = active && (active.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName) ||
      active.closest && active.closest('[contenteditable]:not([contenteditable="false"])'));
    if (panel.hidden && !event.defaultPrevented && !editing && !escapeOverlayWasOpen && selected('focus') === 'on' &&
        apply('focus', '')) changed();
    escapeOverlayWasOpen = false;
  });
  window.addEventListener('resize', positionPanel);
  window.addEventListener('scroll', positionPanel, true);
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
