(function () {
  'use strict';

  var ARTICLE_SELECTOR = '.post-container:not(.moments):not(.moments-comments)';
  var article = document.querySelector(ARTICLE_SELECTOR);
  if (!article) return;

  var item = document.querySelector('.nav-reading-settings');
  var trigger = item && item.querySelector('a');
  if (!item || !trigger) return;

  var root = document.documentElement;
  var panel = document.createElement('div');
  panel.id = 'reading-settings-panel';
  panel.className = 'reading-settings-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', '阅读设置');
  panel.hidden = true;
  panel.innerHTML =
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

  var attributes = { size: 'data-rs-size', width: 'data-rs-width', font: 'data-rs-font' };
  var values = {
    size: ['', 's', 'l', 'xl'],
    width: ['', 'narrow'],
    font: ['', 'serif']
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

  panel.addEventListener('click', function (event) {
    event.stopPropagation();
    var button = event.target.closest('button');
    if (!button) return;
    if (button.classList.contains('rs-reset')) {
      var didChange = false;
      Object.keys(attributes).forEach(function (name) {
        didChange = apply(name, '') || didChange;
      });
      if (didChange) changed();
      return;
    }
    var name = button.getAttribute('data-rs-setting');
    if (name && apply(name, button.getAttribute('data-rs-value'))) changed();
  });

  document.addEventListener('click', function (event) {
    if (!panel.hidden && !panel.contains(event.target) && !trigger.contains(event.target)) close(true);
  });
  document.addEventListener('keydown', function (event) {
    if (!panel.hidden && (event.key === 'Escape' || event.key === 'Esc')) {
      event.preventDefault();
      close(true);
    }
  });
  window.addEventListener('resize', positionPanel);
  window.addEventListener('scroll', positionPanel, true);
  window.addEventListener('reading-settings-change', syncControls);

  item.hidden = false;
  syncControls();
})();
