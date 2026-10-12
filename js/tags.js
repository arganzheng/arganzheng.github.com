(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.TagBrowser = api;
  if (root && root.document) {
    if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', api.init, { once: true });
    else api.init();
  }
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  function has(object, key) { return Object.prototype.hasOwnProperty.call(object || {}, key); }
  function decode(value) {
    try { return decodeURIComponent(value); } catch (error) { return value; }
  }
  function buildIndex(data) {
    var tagIds = Object.create(null), nodesByPath = Object.create(null), tagNodes = Object.create(null), tagPaths = Object.create(null);
    (data.tags || []).forEach(function (name, id) { tagIds[name] = id; });
    function visit(node) {
      nodesByPath[node.p] = node;
      if (node.t != null) { tagNodes[node.t] = node; tagPaths[node.t] = node.p; }
      (node.l || []).forEach(function (id) { tagPaths[id] = node.p; });
      (node.c || []).forEach(visit);
    }
    (data.tree || []).forEach(visit);
    return { tagIds: tagIds, nodesByPath: nodesByPath, tagNodes: tagNodes, tagPaths: tagPaths };
  }
  function selectionKey(item) {
    return item.type === 'node' ? 'node:' + item.path : 'tag:' + item.id;
  }
  function resolveItem(value, data, index) {
    index = index || buildIndex(data);
    var aliases = data.aliases || {}, canonical = value, redirected = false;
    if (has(aliases, value)) { canonical = aliases[value]; redirected = true; }
    if (has(index.tagIds, canonical)) {
      var id = index.tagIds[canonical], node = index.tagNodes[id];
      return node
        ? { type: 'node', path: node.p, label: node.n, id: id, redirected: redirected }
        : { type: 'tag', id: id, label: canonical, redirected: redirected };
    }
    var pathNode = index.nodesByPath[value];
    if (pathNode) return { type: 'node', path: value, label: pathNode.n, redirected: false };
    return null;
  }
  function parseHash(hash, data) {
    var raw = String(hash || '').replace(/^#/, '');
    if (!raw) return { items: [], unknown: [], redirected: false, hadHash: false };
    var whole = decode(raw), index = buildIndex(data), aliases = data.aliases || {};
    var exact = has(index.tagIds, whole) || has(aliases, whole);
    var tokens = exact ? [whole] : whole.split(',').map(decode);
    var items = [], unknown = [], redirected = false, seen = Object.create(null);
    tokens.forEach(function (token) {
      if (!token) return;
      var item = resolveItem(token, data, index);
      if (!item) { unknown.push(token); return; }
      redirected = redirected || item.redirected;
      var key = selectionKey(item);
      if (!seen[key]) { seen[key] = true; items.push(item); }
    });
    return { items: items, unknown: unknown, redirected: redirected, hadHash: true };
  }
  function formatHash(items, data) {
    if (!items || !items.length) return '';
    return '#' + items.map(function (item) {
      if (typeof item === 'string') return encodeURIComponent(item);
      if (item.type === 'node') return encodeURIComponent(item.path);
      return encodeURIComponent(data.tags[item.id]);
    }).join(',');
  }
  function nodeTagIds(node, out) {
    out = out || [];
    if (node.t != null) out.push(node.t);
    (node.l || []).forEach(function (id) { out.push(id); });
    (node.c || []).forEach(function (child) { nodeTagIds(child, out); });
    return out;
  }
  function tagsForSelection(item, data) {
    if (item.type === 'tag') return [item.id];
    var node = buildIndex(data).nodesByPath[item.path];
    return node ? nodeTagIds(node).filter(function (id, i, all) { return all.indexOf(id) === i; }) : [];
  }
  function postIndicesForTags(data, tagIds) {
    var wanted = Object.create(null);
    (tagIds || []).forEach(function (id) { wanted[id] = true; });
    var found = [];
    (data.posts || []).forEach(function (post, index) {
      if ((post[3] || []).some(function (id) { return wanted[id]; })) found.push(index);
    });
    return found;
  }
  function selectedPostIndices(data, items) {
    if (!items || !items.length) return [];
    var sets = items.map(function (item) {
      var found = Object.create(null);
      postIndicesForTags(data, tagsForSelection(item, data)).forEach(function (id) { found[id] = true; });
      return found;
    });
    return (data.posts || []).map(function (_post, id) { return id; }).filter(function (id) {
      return sets.every(function (set) { return !!set[id]; });
    });
  }
  function countDistinctPosts(data, tagIds) {
    return postIndicesForTags(data, tagIds).length;
  }
  function filterMatches(name, aliases, query) {
    var needle = String(query || '').trim().toLowerCase();
    if (!needle) return true;
    if (String(name || '').toLowerCase().indexOf(needle) !== -1) return true;
    return Object.keys(aliases || {}).some(function (alias) {
      return aliases[alias] === name && alias.toLowerCase().indexOf(needle) !== -1;
    });
  }

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }
  function appendText(parent, tag, className, text) {
    var node = element(tag, className, text);
    parent.appendChild(node);
    return node;
  }
  function init() {
    var dataElement = document.getElementById('tag-data');
    var browser = document.getElementById('tag-browser');
    if (!dataElement || !browser || browser.dataset.ready) return;
    browser.dataset.ready = 'true';

    var data;
    try { data = JSON.parse(dataElement.textContent); } catch (error) { return; }
    var index = buildIndex(data);
    var treeHost = document.getElementById('tag-tree');
    var filterInput = document.getElementById('tag-filter');
    var emptyTree = document.getElementById('tag-tree-empty');
    var panel = document.getElementById('tag-tree-panel');
    var home = document.getElementById('tag-results-home');
    var state = document.getElementById('tag-results-state');
    var expanded = Object.create(null);
    var selection = [];
    var unknown = [];
    var query = '';
    var mobile = window.matchMedia && window.matchMedia('(max-width: 991px)');
    var initial = parseHash(window.location.hash, data);
    selection = initial.items;
    unknown = initial.unknown;
    if (panel) panel.open = !mobile || !mobile.matches || !initial.items.length && !initial.unknown.length;

    function normalizedLocation(parsed) {
      if (parsed.redirected || (parsed.items.length && parsed.unknown.length)) {
        var hash = formatHash(parsed.items, data);
        history.replaceState(null, '', window.location.pathname + window.location.search + hash);
      }
    }
    normalizedLocation(initial);

    function selected(item) {
      return selection.some(function (entry) { return selectionKey(entry) === selectionKey(item); });
    }
    function itemForTag(id) {
      return resolveItem(data.tags[id], data, index) || { type: 'tag', id: id, label: data.tags[id] };
    }
    function ancestors(path, includeSelf) {
      var parts = path.split('/'), out = [];
      if (!includeSelf) parts.pop();
      while (parts.length) {
        out.push(parts.join('/'));
        parts.pop();
      }
      return out;
    }
    function requiredExpanded() {
      var required = Object.create(null);
      selection.forEach(function (item) {
        if (item.type === 'node') ancestors(item.path, false).forEach(function (path) { required[path] = true; });
        else if (index.tagPaths[item.id]) required[index.tagPaths[item.id]] = true;
      });
      return required;
    }
    function matchName(name) { return filterMatches(name, data.aliases, query); }
    function filteredNode(node) {
      if (!query) return { node: node, own: true, leaves: (node.l || []).slice(), children: (node.c || []).map(filteredNode), visible: true };
      var own = matchName(node.n);
      var leaves = (node.l || []).filter(function (id) { return matchName(data.tags[id]); });
      var children = (node.c || []).map(filteredNode).filter(function (child) { return child.visible; });
      return { node: node, own: own, leaves: leaves, children: children, visible: own || leaves.length > 0 || children.length > 0 };
    }
    function renderTree() {
      treeHost.replaceChildren();
      var roots = (data.tree || []).map(filteredNode).filter(function (entry) { return entry.visible; });
      emptyTree.hidden = roots.length > 0;
      var mustOpen = requiredExpanded();
      function renderLeaf(id) {
        var tag = data.tags[id], li = element('li', 'tag-tree-item tag-tree-leaf');
        li.setAttribute('role', 'treeitem');
        var row = element('div', 'tag-tree-row');
        var spacer = element('span', 'tag-tree-caret-space', '');
        spacer.setAttribute('aria-hidden', 'true');
        row.appendChild(spacer);
        var name = element('button', 'tag-tree-name', tag);
        name.type = 'button';
        name.setAttribute('aria-pressed', selected({ type: 'tag', id: id }) ? 'true' : 'false');
        if (selected({ type: 'tag', id: id })) row.classList.add('is-selected');
        name.addEventListener('click', function (event) { choose(itemForTag(id), event); });
        row.appendChild(name);
        appendText(row, 'span', 'tag-tree-count', String(data.counts[id] || 0));
        var add = element('button', 'tag-tree-add', '＋');
        add.type = 'button';
        add.title = '加入交集';
        add.setAttribute('aria-label', '加入交集：' + tag);
        add.addEventListener('click', function (event) { event.preventDefault(); event.stopPropagation(); addSelection({ type: 'tag', id: id, label: tag }); });
        row.appendChild(add);
        li.appendChild(row);
        return li;
      }
      function renderNode(entry) {
        var node = entry.node, children = entry.children, leaves = entry.leaves;
        var ids = nodeTagIds(node).filter(function (id, i, all) { return all.indexOf(id) === i; });
        var count = countDistinctPosts(data, ids);
        var li = element('li', 'tag-tree-item tag-tree-group');
        li.setAttribute('role', 'treeitem');
        var row = element('div', 'tag-tree-row');
        var expandable = (node.c || []).length > 0 || (node.l || []).length > 0;
        var isOpen = !!(expanded[node.p] || mustOpen[node.p] || (query && children.length));
        if (expandable) {
          var caret = element('button', 'tag-tree-caret', isOpen ? '▾' : '›');
          caret.type = 'button';
          caret.setAttribute('aria-label', (isOpen ? '收起' : '展开') + '：' + node.n);
          caret.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
          caret.dataset.path = node.p;
          caret.addEventListener('click', function () {
            expanded[node.p] = !isOpen;
            renderTree();
            Array.prototype.some.call(treeHost.querySelectorAll('.tag-tree-caret'), function (button) {
              if (button.dataset.path === node.p) { button.focus(); return true; }
              return false;
            });
          });
          row.appendChild(caret);
        } else {
          var spacer = element('span', 'tag-tree-caret-space', '');
          spacer.setAttribute('aria-hidden', 'true');
          row.appendChild(spacer);
        }
        var item = { type: 'node', path: node.p, label: node.n };
        var chosen = selected(item);
        var name = element('button', 'tag-tree-name', node.n);
        name.type = 'button';
        name.setAttribute('aria-pressed', chosen ? 'true' : 'false');
        if (chosen) row.classList.add('is-selected');
        name.addEventListener('click', function (event) { choose(item, event); });
        row.appendChild(name);
        appendText(row, 'span', 'tag-tree-count', String(count));
        var add = element('button', 'tag-tree-add', '＋');
        add.type = 'button';
        add.title = '加入交集';
        add.setAttribute('aria-label', '加入交集：' + node.n);
        add.addEventListener('click', function (event) { event.preventDefault(); event.stopPropagation(); addSelection(item); });
        row.appendChild(add);
        li.appendChild(row);
        if (isOpen && (leaves.length || children.length)) {
          var group = element('ul', 'tag-tree-children');
          group.setAttribute('role', 'group');
          leaves.forEach(function (id) { group.appendChild(renderLeaf(id)); });
          children.forEach(function (child) { group.appendChild(renderNode(child)); });
          li.appendChild(group);
        }
        return li;
      }
      var root = element('ul', 'tag-tree-root');
      root.setAttribute('role', 'tree');
      root.setAttribute('aria-label', '标签树');
      roots.forEach(function (entry) { root.appendChild(renderNode(entry)); });
      treeHost.appendChild(root);
    }
    function labelFor(item) {
      return item.label || (item.type === 'node' ? (index.nodesByPath[item.path] || {}).n : data.tags[item.id]);
    }
    function formatSelectionHash(items) { return formatHash(items, data); }
    function syncSelection(next, push) {
      var seen = Object.create(null);
      selection = next.filter(function (item) {
        var key = selectionKey(item);
        if (seen[key]) return false;
        seen[key] = true;
        return true;
      });
      unknown = [];
      var hash = formatSelectionHash(selection);
      if (push && window.location.hash !== hash) {
        history.pushState(null, '', window.location.pathname + window.location.search + hash);
      }
      render();
    }
    function addSelection(item) { syncSelection(selection.concat([item]), true); }
    function choose(item, event) {
      if (event.metaKey || event.ctrlKey) addSelection(item);
      else syncSelection([item], true);
    }
    function renderResults() {
      home.hidden = !!(selection.length || unknown.length);
      state.hidden = !home.hidden;
      state.replaceChildren();
      if (!home.hidden) return;
      if (!selection.length) {
        appendText(state, 'p', 'tag-no-match', '没有找到标签 ' + unknown.join('、'));
        return;
      }
      var indices = selectedPostIndices(data, selection);
      var bar = element('div', 'tag-selected-bar');
      selection.forEach(function (item) {
        var chip = element('span', 'tag-selected-chip');
        appendText(chip, 'span', '', labelFor(item));
        var remove = element('button', '', '×');
        remove.type = 'button';
        remove.setAttribute('aria-label', '移除标签 ' + labelFor(item));
        remove.addEventListener('click', function () {
          syncSelection(selection.filter(function (other) { return selectionKey(other) !== selectionKey(item); }), true);
        });
        chip.appendChild(remove);
        bar.appendChild(chip);
      });
      if (selection.length >= 2) {
        var clear = element('button', 'tag-selection-clear', '清除');
        clear.type = 'button';
        clear.addEventListener('click', function () { syncSelection([], true); });
        bar.appendChild(clear);
      }
      state.appendChild(bar);
      appendText(state, 'h2', 'tag-results-title', selection.map(labelFor).join(' ∩ ') + ' · ' + indices.length + ' 篇');
      if (!indices.length) {
        appendText(state, 'p', 'tag-results-empty', '没有同时带有这些标签的文章');
        appendText(state, 'p', 'tag-results-suggestion', '可以移除一个标签，放宽筛选条件。');
        return;
      }
      var list = element('div', 'tag-results-list');
      var shown = 0, pageSize = 100;
      function appendPosts() {
        indices.slice(shown, shown + pageSize).forEach(function (postId) {
          var post = data.posts[postId], article = element('article', 'tag-result');
          var link = element('a', 'tag-result-title', post[1]);
          link.href = post[0];
          article.appendChild(link);
          appendText(article, 'time', 'tag-result-date', post[2]);
          var chips = element('div', 'tag-result-tags');
          (post[3] || []).forEach(function (id) {
            var chip = element('button', 'tag-result-tag', data.tags[id]);
            chip.type = 'button';
            chip.addEventListener('click', function (event) { choose(itemForTag(id), event); });
            chips.appendChild(chip);
          });
          article.appendChild(chips);
          list.appendChild(article);
        });
        shown = Math.min(shown + pageSize, indices.length);
        more.hidden = shown >= indices.length;
      }
      var more = element('button', 'tag-more', '再显示 100 篇');
      more.type = 'button';
      more.hidden = true;
      more.addEventListener('click', appendPosts);
      appendPosts();
      state.appendChild(list);
      state.appendChild(more);
    }
    function render() {
      renderTree();
      renderResults();
    }
    function onLocationChange() {
      var parsed = parseHash(window.location.hash, data);
      selection = parsed.items;
      unknown = parsed.unknown;
      normalizedLocation(parsed);
      render();
    }
    function onFilterInput() {
      query = filterInput.value.trim();
      renderTree();
    }
    filterInput.addEventListener('input', onFilterInput);
    filterInput.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') { filterInput.value = ''; query = ''; renderTree(); }
    });
    window.addEventListener('hashchange', onLocationChange);
    window.addEventListener('popstate', onLocationChange);
    if (mobile && mobile.addEventListener) {
      mobile.addEventListener('change', function (event) {
        if (!event.matches) panel.open = true;
      });
    }
    render();
  }

  return {
    buildIndex: buildIndex,
    resolveItem: resolveItem,
    parseHash: parseHash,
    formatHash: formatHash,
    tagsForSelection: tagsForSelection,
    postIndicesForTags: postIndicesForTags,
    selectedPostIndices: selectedPostIndices,
    countDistinctPosts: countDistinctPosts,
    filterMatches: filterMatches,
    init: init
  };
}));
