(function () {
  'use strict';

  var KEY = 'blogRead';
  var PROGRESS_KEY = 'blogReadProgress';
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

  function readProgressAll() {
    try {
      var value = JSON.parse(localStorage.getItem(PROGRESS_KEY) || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (e) {
      return {};
    }
  }

  function progressEntry(state, value) {
    var path = normalizeUrl(value), entry = path && state[path];
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return null;
    var p = Number(entry.p), a = Number(entry.a);
    if (!isFinite(p) || !isFinite(a)) return null;
    return {
      p: Math.max(0, Math.min(1, Math.round(p * 100) / 100)),
      a: Math.max(0, Math.floor(a)),
      t: Number(entry.t) || 0,
      u: entry.u === 1 ? 1 : 0
    };
  }

  function progressPct(state, value) {
    var entry = progressEntry(state, value);
    return entry ? Math.floor(entry.p * 10) * 10 : 0;
  }

  function saveProgress(state) {
    var keys = Object.keys(state);
    if (keys.length > 400) {
      keys.sort(function (a, b) {
        return (Number(state[a] && state[a].t) || 0) - (Number(state[b] && state[b].t) || 0);
      });
      for (var i = 0; i < keys.length - 400; i++) delete state[keys[i]];
    }
    try {
      localStorage.setItem(PROGRESS_KEY, JSON.stringify(state));
    } catch (e) {}
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

  function setManualOptOut(path, unmarked) {
    var all = readProgressAll(), entry = progressEntry(all, path);
    if (unmarked) {
      entry = entry || { p: 0, a: 0, t: 0, u: 0 };
      if (entry.u === 1) return;
      entry.u = 1;
      entry.t = Date.now();
      all[path] = entry;
      saveProgress(all);
    } else if (entry && entry.u === 1) {
      delete entry.u;
      entry.t = Date.now();
      all[path] = entry;
      saveProgress(all);
    }
  }

  function updateReading(event) {
    if (!document.querySelector('.post-actions .pa-read') ||
        document.documentElement.classList.contains('private-post')) return;
    var detail = event.detail || {}, path = normalizeUrl(detail.path || location.pathname);
    if (!path || path !== normalizeUrl(location.pathname)) return;
    var all = readProgressAll(), previous = progressEntry(all, path);
    var p = Number(detail.p), active = Number(detail.active);
    if (!isFinite(p)) p = 0;
    if (!isFinite(active) || active < 0) active = 0;
    p = Math.max(0, Math.min(1, Math.round(p * 100) / 100));
    var nextP = Math.max(previous ? previous.p : 0, p);
    var nextA = (previous ? previous.a : 0) + Math.floor(active);
    if (previous && nextP === previous.p && nextA === previous.a) return;
    if (!previous && nextP === 0 && nextA === 0) return;

    var oldPct = previous ? Math.floor(previous.p * 10) * 10 : 0;
    var entry = { p: nextP, a: nextA, t: Date.now() };
    if (previous && previous.u === 1) entry.u = 1;
    all[path] = entry;
    saveProgress(all);

    var meta = document.querySelector('.post-length[data-minutes]');
    var minutes = meta ? Number(meta.getAttribute('data-minutes')) : NaN;
    if (!isFinite(minutes) || minutes < 1) minutes = 1;
    var readTime = Math.max(20000, minutes * 60000 * 0.3);
    if (!isRead(path) && entry.u !== 1 && entry.p >= 0.9 && entry.a >= readTime) {
      var button = document.querySelector('.post-actions .pa-read');
      if (button) showToast(button, '读完了，已自动标记为已读');
      setRead(path, true);
      return;
    }
    if (Math.floor(entry.p * 10) * 10 !== oldPct) notify();
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

  function renderSeriesItem(item, state, progress) {
    var url = item.getAttribute('data-url'), read = hasRead(state, url);
    var pct = progressPct(progress, url);
    var partial = !read && pct >= 10;
    item.classList.toggle('is-read', read);
    item.classList.toggle('is-partial', partial);
    if (partial) item.setAttribute('data-rp-pct', String(pct));
    else item.removeAttribute('data-rp-pct');
    Array.prototype.forEach.call(item.children, function (child) {
      if (!child.matches('a, span')) return;
      if (partial) child.setAttribute('data-rp-pct', String(pct));
      else child.removeAttribute('data-rp-pct');
    });
  }

  function renderSeries(state, progress) {
    Array.prototype.forEach.call(document.querySelectorAll('.series-box .sb-list'), function (list) {
      var items = Array.prototype.slice.call(list.querySelectorAll('li[data-url]'));
      items.forEach(function (item) {
        renderSeriesItem(item, state, progress);
      });
      updateSummary(list.closest('.series-box').querySelector('.sb-read'),
        items.map(function (item) { return item.getAttribute('data-url'); }), state, false);
    });

    Array.prototype.forEach.call(document.querySelectorAll('.series-tree .st-item'), function (series) {
      var items = Array.prototype.slice.call(series.querySelectorAll('.st-posts li[data-url]'));
      items.forEach(function (item) {
        renderSeriesItem(item, state, progress);
      });
      updateSummary(series.querySelector('summary > .st-read'),
        items.map(function (item) { return item.getAttribute('data-url'); }), state, false);
    });

    Array.prototype.forEach.call(document.querySelectorAll('.series-tree .st-map'), function (map) {
      var items = Array.prototype.slice.call(map.querySelectorAll('.st-posts li[data-url]'));
      items.forEach(function (item) { renderSeriesItem(item, state, progress); });
      updateSummary(map.querySelector('summary > .st-read'),
        items.map(function (item) { return item.getAttribute('data-url'); }), state, true);
    });
  }

  function renderRemaining(state) {
    Array.prototype.forEach.call(document.querySelectorAll('.series-toc .sb-remaining'), function (line) {
      var box = line.closest('.series-box');
      var list = box && box.querySelector('.sb-list');
      var items = list ? Array.prototype.slice.call(list.querySelectorAll('li[data-url]')) : [];
      if (!items.length) {
        line.hidden = true;
        return;
      }

      var unread = items.filter(function (item) {
        return !hasRead(state, item.getAttribute('data-url'));
      });
      var minutes = unread.reduce(function (total, item) {
        var value = parseInt(item.getAttribute('data-minutes'), 10);
        return total + (isFinite(value) && value > 0 ? value : 0);
      }, 0);
      var time = minutes < 60 ? '约 ' + minutes + ' 分钟' :
        '约 ' + (minutes / 60).toFixed(1).replace(/\.0$/, '') + ' 小时';
      var currentIndex = -1;
      items.some(function (item, index) {
        if (item.classList.contains('is-current')) {
          currentIndex = index;
          return true;
        }
        return false;
      });
      var current = currentIndex >= 0 ? items[currentIndex] : null;
      var currentRead = !!(current && hasRead(state, current.getAttribute('data-url')));

      line.textContent = '';
      if (!unread.length) {
        line.textContent = '这个系列 ' + items.length + ' 篇都读完了';
      } else if (!currentRead) {
        line.textContent = '这个系列还剩 ' + unread.length + ' 篇没读（含本篇），' + time;
      } else {
        var message = '本篇已读完 · 这个系列还剩 ' + unread.length + ' 篇，' + time;
        var next = items[currentIndex + 1];
        if (!next || hasRead(state, next.getAttribute('data-url'))) {
          var target = null;
          for (var i = currentIndex + 1; i < items.length; i++) {
            if (!hasRead(state, items[i].getAttribute('data-url'))) {
              target = items[i];
              break;
            }
          }
          if (!target) target = unread[0];
          var targetLink = target && target.querySelector('a[href]');
          if (targetLink) {
            line.appendChild(document.createTextNode(message + ' · 还没读：'));
            var link = document.createElement('a');
            link.setAttribute('href', targetLink.getAttribute('href'));
            link.textContent = targetLink.textContent;
            line.appendChild(link);
          } else {
            line.textContent = message;
          }
        } else {
          line.textContent = message;
        }
      }
      line.hidden = !line.textContent;
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

  function markerFor(element, read, pct) {
    var marker = Array.prototype.filter.call(element.children, function (child) {
      return child.classList && child.classList.contains('rp-marker');
    })[0];
    element.classList.toggle('is-read', read);
    if (read || pct >= 10) {
      if (!marker) {
        marker = document.createElement('small');
        element.appendChild(marker);
      }
      var markerClass = read ? 'rp-marker' : 'rp-marker is-partial';
      var markerText = read ? '✓ 已读' : '读到 ' + pct + '%';
      if (marker.className !== markerClass) marker.className = markerClass;
      if (marker.textContent !== markerText) marker.textContent = markerText;
    } else if (marker) {
      marker.remove();
    }
  }

  function renderLists(state, progress) {
    Array.prototype.forEach.call(document.querySelectorAll(
      '.post-preview .post-title, .life-card h2, .archive-entry .archive-title, .archive-table-title, .archive-card-title'
    ), function (title) {
      var link = title.closest('a[href]');
      var url = link && link.getAttribute('href');
      markerFor(title, !!(url && hasRead(state, url)), progressPct(progress, url));
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
    var progress = readProgressAll();
    renderToggles(state);
    renderSeries(state, progress);
    renderRemaining(state);
    renderComplete(state);
    renderLists(state, progress);
    renderRoadmaps(state);
  }

  document.addEventListener('click', function (event) {
    var button = event.target.closest('.pa-read');
    if (!button) return;
    var path = normalizeUrl(button.getAttribute('data-url'));
    if (!path) return;
    var read = !isRead(path);
    setManualOptOut(path, !read);
    showToast(button, read ? '已标记为已读' : '已取消已读');
    setRead(path, read);
  });

  document.addEventListener('blog:read-progress', renderAll);
  document.addEventListener('blog:reading', updateReading);
  window.addEventListener('storage', function (event) {
    if (event.key === KEY || event.key === PROGRESS_KEY || event.key === null) notify();
  });

  window.BlogReadProgress = {
    isRead: isRead,
    normalizeUrl: normalizeUrl,
    progress: function (value) {
      var entry = progressEntry(readProgressAll(), value);
      return entry ? { p: entry.p, a: entry.a } : null;
    }
  };
  renderAll();
  var archive = document.getElementById('archive');
  if (archive && window.MutationObserver) {
    new MutationObserver(renderAll).observe(archive, { childList: true, subtree: true });
  }
})();
