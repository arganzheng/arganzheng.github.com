(function () {
  'use strict';

  var KEY = 'blogRead';
  var completeStates = new WeakMap();

  function normalizeUrl(value) {
    if (!value) return '';
    var path;
    try {
      path = new URL(value, location.href).pathname;
    } catch (e) {
      return '';
    }
    var baseurl = document.documentElement.getAttribute('data-baseurl') || '';
    if (baseurl && baseurl !== '/' && (path === baseurl || path.indexOf(baseurl + '/') === 0)) {
      path = path.slice(baseurl.length) || '/';
    }
    return path;
  }

  function readAll() {
    try {
      var value = JSON.parse(localStorage.getItem(KEY) || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (e) {
      return {};
    }
  }

  function hasRead(state, value) {
    var path = normalizeUrl(value);
    return !!(path && Object.prototype.hasOwnProperty.call(state, path));
  }

  function isRead(value) {
    return hasRead(readAll(), value);
  }

  function notify() {
    document.dispatchEvent(new Event('blog:read-progress'));
  }

  function setRead(path, read) {
    var all = readAll();
    if (read) all[path] = Date.now();
    else delete all[path];
    try {
      localStorage.setItem(KEY, JSON.stringify(all));
    } catch (e) {}
    notify();
  }

  function renderToggles(state) {
    Array.prototype.forEach.call(document.querySelectorAll('.pa-read'), function (button) {
      var read = hasRead(state, button.getAttribute('data-url'));
      button.classList.toggle('is-read', read);
      button.setAttribute('aria-pressed', String(read));
      button.title = read ? '点一下取消已读' : '标记已读';
      var label = button.querySelector('.pa-read-label');
      if (label) label.textContent = read ? '已读' : '标记已读';
    });
  }

  function showToast(button, message) {
    var toast = button.closest('.post-actions').querySelector('.pa-toast');
    if (!toast) return;
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(toast._readTimer);
    toast._readTimer = setTimeout(function () { toast.hidden = true; }, 2500);
  }

  function updateSummary(element, urls, state, distinct) {
    if (!element) return;
    var seen = Object.create(null), total = 0, count = 0;
    urls.forEach(function (url) {
      var path = normalizeUrl(url);
      if (!path || (distinct && seen[path])) return;
      seen[path] = true;
      total++;
      if (hasRead(state, path)) count++;
    });
    element.hidden = count === 0;
    element.textContent = count ? '已读 ' + count + '/' + total : '';
  }

  function renderSeries(state) {
    Array.prototype.forEach.call(document.querySelectorAll('.series-box .sb-list'), function (list) {
      var items = Array.prototype.slice.call(list.querySelectorAll('li[data-url]'));
      items.forEach(function (item) {
        item.classList.toggle('is-read', hasRead(state, item.getAttribute('data-url')));
      });
      updateSummary(list.closest('.series-box').querySelector('.sb-read'),
        items.map(function (item) { return item.getAttribute('data-url'); }), state, false);
    });

    Array.prototype.forEach.call(document.querySelectorAll('.series-tree .st-item'), function (series) {
      var items = Array.prototype.slice.call(series.querySelectorAll('.st-posts li[data-url]'));
      items.forEach(function (item) {
        item.classList.toggle('is-read', hasRead(state, item.getAttribute('data-url')));
      });
      updateSummary(series.querySelector('summary > .st-read'),
        items.map(function (item) { return item.getAttribute('data-url'); }), state, false);
    });

    Array.prototype.forEach.call(document.querySelectorAll('.series-tree .st-map'), function (map) {
      var items = Array.prototype.slice.call(map.querySelectorAll('.st-posts li[data-url]'));
      updateSummary(map.querySelector('summary > .st-read'),
        items.map(function (item) { return item.getAttribute('data-url'); }), state, true);
    });
  }

  function renderComplete(state) {
    Array.prototype.forEach.call(document.querySelectorAll('.series-toc .sb-complete'), function (card) {
      var box = card.closest('.series-toc');
      if (!box) return;
      var items = Array.prototype.slice.call(box.querySelectorAll('.sb-list li[data-url]'));
      var complete = items.length > 0 && items.every(function (item) {
        return hasRead(state, item.getAttribute('data-url'));
      });
      var previous = completeStates.has(card) ? completeStates.get(card) : null;
      var title = card.querySelector('.sb-complete-title');
      var remaining = box.querySelector('.sb-remaining');

      card.hidden = !complete;
      if (!complete) {
        if (title) title.textContent = '';
        if (remaining && remaining.textContent.trim()) remaining.hidden = false;
        completeStates.set(card, false);
        return;
      }

      var timestamps = [];
      items.forEach(function (item) {
        var path = normalizeUrl(item.getAttribute('data-url'));
        var timestamp = path ? Number(state[path]) : NaN;
        if (isFinite(timestamp) && timestamp > 0) timestamps.push(timestamp);
      });
      var duration = '';
      if (timestamps.length) {
        var earliest = Math.min.apply(Math, timestamps);
        var latest = Math.max.apply(Math, timestamps);
        var span = latest - earliest;
        duration = span < 86400000 ? '，前后不到一天' : '，前后用了 ' + Math.ceil(span / 86400000) + ' 天';
      }
      var name = card.getAttribute('data-name') || '';
      var count = items.length;
      var message = card.getAttribute('data-status') === '完结'
        ? '《' + name + '》全部 ' + count + ' 篇都读完了'
        : '《' + name + '》目前的 ' + count + ' 篇都读完了，后面的还在写';
      if (title) title.textContent = message + duration;
      if (remaining) remaining.hidden = true;

      if (previous === false) {
        if (card._sbCompleteTimer) clearTimeout(card._sbCompleteTimer);
        card.classList.remove('is-new');
        void card.offsetWidth;
        card.classList.add('is-new');
        card._sbCompleteTimer = setTimeout(function () {
          card.classList.remove('is-new');
          card._sbCompleteTimer = null;
        }, 1800);
        var button = document.querySelector('.pa-read');
        if (button) showToast(button, '整个系列读完了');
      }
      completeStates.set(card, true);
    });
  }

  function markerFor(element, read) {
    var marker = Array.prototype.filter.call(element.children, function (child) {
      return child.classList && child.classList.contains('rp-marker');
    })[0];
    if (!read) {
      if (marker) marker.remove();
      element.classList.remove('is-read');
      return;
    }
    element.classList.add('is-read');
    if (!marker) {
      marker = document.createElement('small');
      marker.className = 'rp-marker';
      marker.textContent = '✓ 已读';
      element.appendChild(marker);
    }
  }

  function renderLists(state) {
    Array.prototype.forEach.call(document.querySelectorAll(
      '.post-preview .post-title, .life-card h2, .archive-entry .archive-title, .archive-table-title, .archive-card-title'
    ), function (title) {
      var link = title.closest('a[href]');
      markerFor(title, !!(link && hasRead(state, link.getAttribute('href'))));
    });
  }

  function renderRoadmaps(state) {
    var script = document.getElementById('series-progress-data');
    if (!script) return;
    var data;
    try {
      data = JSON.parse(script.textContent);
    } catch (e) {
      return;
    }
    var links = document.querySelectorAll('.post-container table a[href]');
    Object.keys(data).forEach(function (key) {
      var series = data[key], overview = normalizeUrl(series.overview), link = null;
      Array.prototype.some.call(links, function (candidate) {
        if (normalizeUrl(candidate.getAttribute('href')) === overview) {
          link = candidate;
          return true;
        }
        return false;
      });
      if (!link) return;
      var cell = link.closest('td');
      if (!cell) return;
      var count = series.posts.filter(function (url) { return hasRead(state, url); }).length;
      var badge = cell.querySelector('.rp-badge');
      if (!count) {
        if (badge) badge.remove();
        return;
      }
      if (!badge) {
        badge = document.createElement('small');
        badge.className = 'rp-badge';
        cell.appendChild(badge);
      }
      badge.textContent = '已读 ' + count + '/' + series.posts.length;
    });
  }

  function renderAll() {
    var state = readAll();
    renderToggles(state);
    renderSeries(state);
    renderComplete(state);
    renderLists(state);
    renderRoadmaps(state);
  }

  document.addEventListener('click', function (event) {
    var button = event.target.closest('.pa-read');
    if (!button) return;
    var path = normalizeUrl(button.getAttribute('data-url'));
    if (!path) return;
    var read = !isRead(path);
    showToast(button, read ? '已标记为已读' : '已取消已读');
    setRead(path, read);
  });

  document.addEventListener('blog:read-progress', renderAll);
  window.addEventListener('storage', function (event) {
    if (event.key === KEY || event.key === null) notify();
  });

  window.BlogReadProgress = { isRead: isRead, normalizeUrl: normalizeUrl };
  renderAll();
  var archive = document.getElementById('archive');
  if (archive && window.MutationObserver) {
    new MutationObserver(renderAll).observe(archive, { childList: true, subtree: true });
  }
})();
