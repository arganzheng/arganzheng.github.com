(function () {
  'use strict';

  var placeholder = document.querySelector('.continue-series');
  if (!placeholder) return;

  var DISMISS_KEY = 'blogContinueDismiss';
  var MAX_AGE = 180 * 86400000;
  var now = Date.now();
  var base = (placeholder.getAttribute('data-base') || '').replace(/\/$/, '');
  var activity = Object.create(null);
  var read = Object.create(null);

  function readObject(key) {
    try {
      var value = JSON.parse(localStorage.getItem(key) || '{}');
      return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
    } catch (e) {
      return {};
    }
  }

  function timestamp(value) {
    if (value === null || value === undefined || value === '') return 0;
    var time = Number(value);
    return isFinite(time) && time > 0 && now - time <= MAX_AGE ? time : 0;
  }

  function normalizePath(value) {
    if (!value) return '';
    var path = String(value);
    if (path.charAt(0) !== '/') path = '/' + path;
    if (base && (path === base || path.indexOf(base + '/') === 0)) {
      path = path.slice(base.length) || '/';
    }
    return path;
  }

  function progress(value) {
    var p = Number(value);
    return isFinite(p) && p >= 0 && p <= 1 ? p : null;
  }

  function touch(path, time, p, hasReadPos) {
    path = normalizePath(path);
    if (!path || !time) return;
    var entry = activity[path] || { t: 0, p: null, hasReadPos: false };
    if (time > entry.t) entry.t = time;
    if (p !== null && (entry.p === null || p > entry.p)) entry.p = p;
    if (hasReadPos) entry.hasReadPos = true;
    activity[path] = entry;
  }

  var readState = readObject('blogRead');
  Object.keys(readState).forEach(function (path) {
    var time = timestamp(readState[path]);
    var normalized = normalizePath(path);
    if (!normalized || !time) return;
    read[normalized] = true;
    touch(normalized, time, null, false);
  });

  var progressState = readObject('blogReadProgress');
  Object.keys(progressState).forEach(function (path) {
    var entry = progressState[path];
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return;
    var time = timestamp(entry.t);
    if (!time) return;
    var p = progress(entry.p);
    touch(path, time, p !== null && p > 0 ? p : null, false);
  });

  try {
    for (var i = 0; i < localStorage.length; i++) {
      var key = localStorage.key(i);
      if (!key || key.indexOf('readpos:') !== 0) continue;
      var position;
      try {
        position = JSON.parse(localStorage.getItem(key) || 'null');
      } catch (e) {
        position = null;
      }
      if (!position || typeof position !== 'object') continue;
      var p = progress(position.p);
      var time = timestamp(position.t);
      if (p === null || p < 0.05 || p > 0.95 || !time) continue;
      touch(key.slice(8), time, p, true);
    }
  } catch (e) {
    // Storage may be unavailable.
  }

  if (!Object.keys(activity).length) return;

  function hrefFor(url) {
    var path = String(url || '');
    if (base && path.charAt(0) === '/' && path.indexOf(base + '/') !== 0) {
      return base + path;
    }
    return path;
  }

  function postData(value) {
    if (!Array.isArray(value) || value.length < 2) return null;
    var url = String(value[0] || '');
    var path = normalizePath(url);
    if (!path) return null;
    return { url: url, path: path, subtitle: String(value[1] || '') };
  }

  function showBanner(data) {
    var series = data && data.series;
    if (!series || typeof series !== 'object') return;

    var dismissed = readObject(DISMISS_KEY);
    var choices = [];
    Object.keys(series).forEach(function (key) {
      var entry = series[key];
      var posts = entry && Array.isArray(entry.posts) ? entry.posts.map(postData).filter(Boolean) : [];
      if (!posts.length) return;

      var last = 0, readCount = 0, recent = null;
      posts.forEach(function (post, index) {
        if (read[post.path]) readCount++;
        var touched = activity[post.path];
        if (!touched) return;
        if (touched.t > last) last = touched.t;
        if (!recent || touched.t > recent.activity.t) {
          recent = { post: post, index: index, activity: touched };
        }
      });
      if (!last || readCount === posts.length || last <= timestamp(dismissed[key])) return;
      choices.push({
        key: key,
        name: String(entry.name || ''),
        posts: posts,
        last: last,
        readCount: readCount,
        recent: recent
      });
    });
    choices.sort(function (a, b) { return b.last - a.last; });

    var chosen = choices[0];
    if (!chosen || !chosen.recent) return;

    var targetIndex = -1, text;
    var recent = chosen.recent;
    if (!read[recent.post.path] && recent.activity.p !== null) {
      targetIndex = recent.index;
      text = '继续读《' + chosen.name + '》 · 第 ' + (targetIndex + 1) + '/' + chosen.posts.length +
        ' 篇：' + chosen.posts[targetIndex].subtitle + ' · 上次读到 ' +
        Math.round(recent.activity.p * 100) + '%';
    } else {
      for (var j = recent.index + 1; j < chosen.posts.length; j++) {
        if (!read[chosen.posts[j].path]) {
          targetIndex = j;
          break;
        }
      }
      if (targetIndex < 0) {
        for (var k = 0; k < chosen.posts.length; k++) {
          if (!read[chosen.posts[k].path]) {
            targetIndex = k;
            break;
          }
        }
      }
      if (targetIndex < 0) return;
      text = '继续读《' + chosen.name + '》 · 下一篇 第 ' + (targetIndex + 1) + '/' +
        chosen.posts.length + ' 篇：' + chosen.posts[targetIndex].subtitle;
    }
    if (chosen.readCount > 0) text += ' · 已读 ' + chosen.readCount + '/' + chosen.posts.length;

    var target = chosen.posts[targetIndex];
    var link = document.createElement('a');
    link.className = 'continue-series-link';
    link.setAttribute('href', hrefFor(target.url));
    link.textContent = text;
    if (activity[target.path] && activity[target.path].hasReadPos) {
      link.addEventListener('click', function () {
        try {
          sessionStorage.setItem('blogResume', new URL(link.href, location.href).pathname);
        } catch (e) {
          // Session storage may be unavailable.
        }
      });
    }

    var dismiss = document.createElement('button');
    dismiss.type = 'button';
    dismiss.className = 'continue-series-dismiss';
    dismiss.setAttribute('aria-label', '不再提示');
    dismiss.textContent = '×';
    dismiss.addEventListener('click', function () {
      var state = readObject(DISMISS_KEY);
      state[chosen.key] = Date.now();
      try {
        localStorage.setItem(DISMISS_KEY, JSON.stringify(state));
      } catch (e) {
        // Storage may be unavailable.
      }
      placeholder.hidden = true;
    });

    placeholder.textContent = '';
    placeholder.appendChild(link);
    placeholder.appendChild(dismiss);
    placeholder.hidden = false;
  }

  var mapUrl = (base || '') + '/series-map.json';
  fetch(mapUrl).then(function (response) {
    if (!response.ok) throw new Error('Unable to load series map');
    return response.json();
  }).then(showBanner).catch(function () {});
})();
