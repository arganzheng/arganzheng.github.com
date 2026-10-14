/**
 * favorites.js — browser-local saved articles and passage excerpts.
 *
 * There is deliberately no server-side account or worker endpoint here:
 * everything lives in localStorage["blogFavorites"].
 */
(function () {
  'use strict';

  var KEY = 'blogFavorites';
  var MAX_PASSAGES = 2000;
  var pageNotice = '';
  var pageSearch = '';
  var offlinePages = {};

  function emptyState() {
    return { posts: {}, passages: [] };
  }

  function normalizePath(value) {
    if (!value) return '';
    var path = String(value);
    try { path = new URL(path, location.href).pathname; } catch (e) { path = path.split(/[?#]/)[0]; }
    var base = document.documentElement.getAttribute('data-baseurl') || '';
    if (base && base !== '/' && path.indexOf(base + '/') === 0) path = path.slice(base.length);
    if (path.charAt(0) !== '/') path = '/' + path;
    return path.replace(/\/{2,}/g, '/');
  }

  function passageId(path, exact) {
    var text = normalizePath(path) + '\n' + String(exact || ''), hash = 0x811c9dc5;
    for (var i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return ('0000000' + hash.toString(16)).slice(-8);
  }

  function cleanState(value) {
    var state = emptyState();
    if (!value || typeof value !== 'object') return state;
    if (value.posts && typeof value.posts === 'object') {
      Object.keys(value.posts).forEach(function (key) {
        var path = normalizePath(key), item = value.posts[key];
        if (!path || !item || typeof item !== 'object') return;
        state.posts[path] = {
          title: String(item.title || path),
          t: Number(item.t) || 0
        };
      });
    }
    if (Array.isArray(value.passages)) {
      var seen = {};
      value.passages.forEach(function (item) {
        if (!item || typeof item !== 'object') return;
        var path = normalizePath(item.path), exact = String(item.exact || '');
        if (!path || !exact) return;
        var id = passageId(path, exact);
        if (seen[id]) return;
        seen[id] = true;
        state.passages.push({
          id: id,
          path: path,
          title: String(item.title || path),
          exact: exact,
          section: String(item.section || ''),
          t: Number(item.t) || 0
        });
      });
    }
    state.passages.sort(function (a, b) { return a.t - b.t; });
    if (state.passages.length > MAX_PASSAGES) state.passages = state.passages.slice(-MAX_PASSAGES);
    return state;
  }

  function readState() {
    try {
      var raw = localStorage.getItem(KEY);
      return cleanState(raw ? JSON.parse(raw) : null);
    } catch (e) {
      return emptyState();
    }
  }

  function writeState(state) {
    try {
      localStorage.setItem(KEY, JSON.stringify(cleanState(state)));
      return true;
    } catch (e) {
      return false;
    }
  }

  function copyState(state) {
    try { return cleanState(JSON.parse(JSON.stringify(state))); }
    catch (e) { return emptyState(); }
  }

  function hasPost(path) {
    return !!readState().posts[normalizePath(path)];
  }

  function hasPassage(path, exact) {
    var id = passageId(path, exact);
    return readState().passages.some(function (item) { return item.id === id; });
  }

  function togglePost(path, title) {
    var state = readState(), normalized = normalizePath(path), saved;
    if (!normalized) return false;
    if (state.posts[normalized]) {
      delete state.posts[normalized];
      saved = false;
    } else {
      state.posts[normalized] = { title: String(title || normalized), t: Date.now() };
      saved = true;
    }
    if (writeState(state)) notify();
    return saved;
  }

  function togglePassage(info) {
    info = info || {};
    var state = readState(), path = normalizePath(info.path), exact = String(info.exact || '');
    if (!path || !exact) return false;
    var id = passageId(path, exact), index = -1;
    for (var i = 0; i < state.passages.length; i++) {
      if (state.passages[i].id === id) { index = i; break; }
    }
    var saved;
    if (index !== -1) {
      state.passages.splice(index, 1);
      saved = false;
    } else {
      state.passages.push({
        id: id,
        path: path,
        title: String(info.title || document.title),
        exact: exact,
        section: String(info.section || ''),
        t: Date.now()
      });
      state.passages = state.passages.slice(-MAX_PASSAGES);
      saved = true;
    }
    if (writeState(state)) notify();
    return saved;
  }

  function hasAny(state) {
    return Object.keys(state.posts).length > 0 || state.passages.length > 0;
  }

  function notify() {
    var state = readState();
    syncNav(state);
    renderArticleButtons(state);
    renderFavoritesPage(state);
    try { document.dispatchEvent(new CustomEvent('blog:favorites', { detail: copyState(state) })); } catch (e) {}
  }

  function syncNav(state) {
    document.documentElement.classList.toggle('has-favorites', hasAny(state));
  }

  function renderArticleButtons(state) {
    document.querySelectorAll('.pa-fav').forEach(function (button) {
      var path = normalizePath(button.getAttribute('data-url')), saved = !!state.posts[path];
      button.classList.toggle('is-active', saved);
      button.setAttribute('aria-pressed', saved ? 'true' : 'false');
      button.title = saved ? '点一下取消收藏' : '收藏这篇文章';
      var icon = button.querySelector('.fa'), label = button.querySelector('.pa-fav-label');
      if (icon) icon.className = 'fa ' + (saved ? 'fa-bookmark' : 'fa-regular fa-bookmark');
      if (label) label.textContent = saved ? '已收藏' : '收藏';
    });
  }

  function formatDate(t) {
    var d = new Date(Number(t) || Date.now());
    return d.getFullYear() + '-' + ('0' + (d.getMonth() + 1)).slice(-2) + '-' + ('0' + d.getDate()).slice(-2);
  }

  function encodeFragmentPart(value) {
    return String(value).replace(/[\s%,&#()"'<>\[\]\\^`{}|]/g, function (c) {
      return '%' + ('0' + c.charCodeAt(0).toString(16).toUpperCase()).slice(-2);
    });
  }

  function siteUrl(app, path) {
    var base = (app.getAttribute('data-site-url') || location.origin).replace(/\/$/, '');
    return base + normalizePath(path);
  }

  function makeButton(text, className) {
    var button = document.createElement('button');
    button.type = 'button'; button.className = className || ''; button.textContent = text;
    return button;
  }

  function addText(parent, text, className) {
    var span = document.createElement('span');
    if (className) span.className = className;
    span.textContent = text; parent.appendChild(span); return span;
  }

  function groupState(state) {
    var groups = {};
    Object.keys(state.posts).forEach(function (path) {
      groups[path] = { path: path, title: state.posts[path].title || path, post: state.posts[path], passages: [] };
    });
    state.passages.forEach(function (item) {
      var group = groups[item.path] || (groups[item.path] = { path: item.path, title: item.title || item.path, post: null, passages: [] });
      if (!group.title && item.title) group.title = item.title;
      group.passages.push(item);
    });
    return Object.keys(groups).map(function (path) {
      var group = groups[path];
      group.passages.sort(function (a, b) { return b.t - a.t; });
      group.latest = Math.max(group.post ? group.post.t : 0, group.passages.reduce(function (n, p) { return Math.max(n, p.t); }, 0));
      return group;
    }).sort(function (a, b) { return b.latest - a.latest; });
  }

  function matchesGroup(group, query) {
    if (!query) return true;
    var text = (group.title + ' ' + group.path + ' ' + group.passages.map(function (p) {
      return p.exact + ' ' + p.section;
    }).join(' ')).toLowerCase();
    return text.indexOf(query.toLowerCase()) !== -1;
  }

  function renderFavoritesPage(state) {
    var app = document.getElementById('favorites-app');
    if (!app) return;
    var searchValue = pageSearch, groups = groupState(state), visible = groups.filter(function (g) { return matchesGroup(g, searchValue); });
    app.innerHTML = '';
    var toolbar = document.createElement('div'); toolbar.className = 'favorites-toolbar';
    var search = document.createElement('input');
    search.type = 'search'; search.className = 'favorites-search'; search.placeholder = '搜索文章、摘录或章节'; search.value = searchValue;
    search.setAttribute('aria-label', '搜索收藏');
    search.addEventListener('input', function () {
      pageSearch = search.value.trim();
      var caret = search.selectionStart;
      renderFavoritesPage(readState());
      var next = app.querySelector('.favorites-search');
      if (next) {
        next.focus();
        try { next.setSelectionRange(caret, caret); } catch (e) {}
      }
    });
    toolbar.appendChild(search);
    addText(toolbar, Object.keys(state.posts).length + ' 篇文章 · ' + state.passages.length + ' 段摘录', 'favorites-count');
    var exportButton = makeButton('导出收藏（Markdown）', 'favorites-export');
    exportButton.title = '把收藏的文章标题、链接和摘录导出成一个 .md 文件（不含文章全文）';
    exportButton.addEventListener('click', function () { exportMarkdown(app, state); });
    var backupButton = makeButton('备份', 'favorites-backup');
    backupButton.addEventListener('click', function () { downloadFile('blogFavorites-' + formatDate(Date.now()) + '.json', JSON.stringify(state, null, 2), 'application/json'); });
    var restoreButton = makeButton('恢复', 'favorites-restore');
    var file = document.createElement('input'); file.type = 'file'; file.accept = 'application/json,.json'; file.hidden = true;
    restoreButton.addEventListener('click', function () { file.click(); });
    file.addEventListener('change', function () { if (file.files[0]) restoreFile(file.files[0]); file.value = ''; });
    toolbar.appendChild(exportButton); toolbar.appendChild(backupButton); toolbar.appendChild(restoreButton); toolbar.appendChild(file);
    app.appendChild(toolbar);
    if (pageNotice) { addText(app, pageNotice, 'favorites-notice').setAttribute('role', 'status'); pageNotice = ''; }
    var offline = Object.keys(offlinePages).map(function (path) { return offlinePages[path]; }).filter(function (entry) {
      return !searchValue || (entry.title + ' ' + entry.path).toLowerCase().indexOf(searchValue.toLowerCase()) !== -1;
    });
    if (offline.length) {
      var section = document.createElement('section'); section.className = 'favorites-offline';
      var head = document.createElement('div'); head.className = 'favorites-offline-head';
      var bytes = offline.reduce(function (sum, entry) { return sum + (entry.bytes || 0); }, 0);
      var heading = document.createElement('h2');
      heading.textContent = '离线保存 · ' + offline.length + ' 篇 · 共 ' + window.BlogOfflineSave.formatSize(bytes);
      head.appendChild(heading);
      var clear = makeButton('全部清除', 'favorites-offline-clear');
      clear.addEventListener('click', function () {
        if (confirm('删除全部离线保存的文章？')) window.BlogOfflineSave.clear();
      });
      head.appendChild(clear); section.appendChild(head);
      var offlineList = document.createElement('ul'); offlineList.className = 'favorites-offline-list';
      offline.forEach(function (entry) {
        var row = document.createElement('li'); row.className = 'favorites-offline-item';
        var link = document.createElement('a'); link.className = 'favorites-offline-title';
        link.href = siteUrl(app, entry.path); link.textContent = entry.title; row.appendChild(link);
        var meta = document.createElement('span'); meta.className = 'favorites-offline-meta';
        meta.textContent = '保存于 ' + formatDate(entry.savedAt) + ' · ' + window.BlogOfflineSave.formatSize(entry.bytes);
        row.appendChild(meta);
        var remove = makeButton('删除', 'favorites-remove');
        remove.addEventListener('click', function () { window.BlogOfflineSave.remove(entry.path); });
        row.appendChild(remove); offlineList.appendChild(row);
      });
      section.appendChild(offlineList); app.appendChild(section);
    }
    if (!groups.length) {
      var empty = document.createElement('div'); empty.className = 'favorites-empty';
      empty.innerHTML = '<h2>还没有收藏</h2><p>在文章中选中文字后点「收藏」，或在文章底部操作栏点「收藏」。收藏只保存在当前浏览器里，建议定期备份。</p>';
      app.appendChild(empty); return;
    }
    if (!visible.length) { addText(app, '没有匹配的收藏。', 'favorites-no-results'); return; }
    var list = document.createElement('div'); list.className = 'favorites-groups';
    visible.forEach(function (group) { renderGroup(app, list, group, searchValue); });
    app.appendChild(list);
  }

  function renderGroup(app, list, group, query) {
    var article = document.createElement('article'); article.className = 'favorites-group';
    var head = document.createElement('header'); head.className = 'favorites-group-head';
    var link = document.createElement('a'); link.href = siteUrl(app, group.path); link.textContent = group.title; link.className = 'favorites-group-title';
    head.appendChild(link);
    if (offlinePages[normalizePath(group.path)]) addText(head, '已离线', 'favorites-offline-badge');
    addText(head, '收藏于 ' + formatDate(group.latest), 'favorites-date');
    if (group.post) {
      var remove = makeButton('移除文章', 'favorites-remove favorites-remove-post');
      remove.title = '取消收藏这篇文章'; remove.addEventListener('click', function () {
        togglePost(group.path, group.title);
      }); head.appendChild(remove);
    }
    article.appendChild(head);
    var titleMatches = !query || (group.title + ' ' + group.path).toLowerCase().indexOf(query.toLowerCase()) !== -1;
    var passages = titleMatches ? group.passages : group.passages.filter(function (p) {
      return (p.exact + ' ' + p.section).toLowerCase().indexOf(query.toLowerCase()) !== -1;
    });
    if (passages.length) {
      var quotes = document.createElement('ul'); quotes.className = 'favorites-passages';
      passages.forEach(function (passage) {
        var li = document.createElement('li'); li.className = 'favorites-passage';
        var quote = document.createElement('a');
        quote.href = siteUrl(app, passage.path) + '#hl=' + encodeFragmentPart(passage.exact);
        quote.textContent = passage.exact; quote.className = 'favorites-quote';
        li.appendChild(quote);
        var meta = document.createElement('div'); meta.className = 'favorites-passage-meta';
        if (passage.section) addText(meta, passage.section, 'favorites-section');
        addText(meta, formatDate(passage.t), 'favorites-date');
        var removePassage = makeButton('移除', 'favorites-remove');
        removePassage.addEventListener('click', function () { togglePassage(passage); });
        meta.appendChild(removePassage); li.appendChild(meta); quotes.appendChild(li);
      });
      article.appendChild(quotes);
    }
    list.appendChild(article);
  }

  function exportMarkdown(app, state) {
    var lines = ['# 我的收藏', ''];
    groupState(state).forEach(function (group) {
      lines.push('## [' + escapeMarkdown(group.title) + '](' + siteUrl(app, group.path) + ')', '');
      group.passages.slice().reverse().forEach(function (passage) {
        lines.push('> ' + passage.exact.replace(/\n/g, '\n> '), '');
        lines.push((passage.section || '未标注章节') + ' · ' + formatDate(passage.t), '');
      });
    });
    downloadFile('my-favorites-' + formatDate(Date.now()) + '.md', lines.join('\n'), 'text/markdown;charset=utf-8');
  }

  function escapeMarkdown(value) {
    return String(value).replace(/([\\`*_[\]<>~|])/g, '\\$1');
  }

  function downloadFile(name, text, type) {
    var url = URL.createObjectURL(new Blob([text], { type: type }));
    var link = document.createElement('a'); link.href = url; link.download = name; link.style.display = 'none';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function restoreFile(file) {
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var imported = cleanState(JSON.parse(reader.result)), state = readState(), posts = 0, passages = 0;
        Object.keys(imported.posts).forEach(function (path) {
          if (!state.posts[path]) posts++;
          if (!state.posts[path] || imported.posts[path].t > state.posts[path].t) state.posts[path] = imported.posts[path];
        });
        var seen = {}; state.passages.forEach(function (p) { seen[p.id] = true; });
        imported.passages.forEach(function (p) {
          if (!seen[p.id]) { state.passages.push(p); seen[p.id] = true; passages++; }
        });
        state.passages.sort(function (a, b) { return a.t - b.t; });
        state.passages = state.passages.slice(-MAX_PASSAGES);
        if (writeState(state)) { pageNotice = '已恢复 ' + posts + ' 篇新文章、' + passages + ' 段新摘录。'; notify(); }
        else { pageNotice = '恢复失败：浏览器拒绝写入本地数据。'; renderFavoritesPage(readState()); }
      } catch (e) { pageNotice = '恢复失败：这不是有效的收藏备份。'; renderFavoritesPage(readState()); }
    };
    reader.onerror = function () { pageNotice = '恢复失败：无法读取备份文件。'; renderFavoritesPage(readState()); };
    reader.readAsText(file);
  }

  document.addEventListener('click', function (event) {
    var button = event.target && event.target.closest && event.target.closest('.pa-fav');
    if (!button) return;
    event.preventDefault();
    var bar = button.closest('.post-actions'), saved = togglePost(button.getAttribute('data-url'), bar && bar.getAttribute('data-title'));
    if (bar) {
      var toast = bar.querySelector('.pa-toast');
      if (toast) {
        toast.textContent = saved ? '已收藏，可在导航栏的书签图标里查看' : '已取消收藏';
        toast.hidden = false; clearTimeout(toast._favoriteTimer);
        toast._favoriteTimer = setTimeout(function () { toast.hidden = true; }, 2500);
      }
    }
  });

  document.addEventListener('blog:favorites', function (event) {
    var state = event.detail && event.detail.posts ? cleanState(event.detail) : readState();
    syncNav(state); renderArticleButtons(state); renderFavoritesPage(state);
  });
  function loadOfflinePages() {
    if (!window.BlogOfflineSave) return;
    window.BlogOfflineSave.list().then(function (entries) {
      offlinePages = {};
      entries.forEach(function (entry) { offlinePages[normalizePath(entry.path)] = entry; });
      renderFavoritesPage(readState());
    });
  }
  document.addEventListener('blog:offline-saved', loadOfflinePages);
  setTimeout(loadOfflinePages, 0);
  window.addEventListener('storage', function (event) {
    if (event.key === KEY || event.key === null) notify();
  });

  window.BlogFavorites = {
    hasPost: hasPost,
    togglePost: togglePost,
    hasPassage: hasPassage,
    togglePassage: togglePassage,
    all: function () { return copyState(readState()); }
  };
  notify();
})();
