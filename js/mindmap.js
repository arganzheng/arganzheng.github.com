(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var MODE_KEY = 'mindmapMode';
  var FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  var MAX_SCALE = 3;
  var MIN_SCALE = 0.2;
  var PALETTES = {
    light: {
      branches: ['#0969da', '#8250df', '#bf3989', '#cf222e', '#bc4c00', '#9a6700', '#1a7f37', '#0550ae'],
      tints: ['#ddf4ff', '#fbefff', '#ffeff7', '#ffebe9', '#fff1e5', '#fff8c5', '#dafbe1', '#eaeef2'],
      accent: '#0969da',
      rootText: '#ffffff',
      text: '#24292f',
      muted: '#1a7f37',
      background: '#ffffff'
    },
    dark: {
      branches: ['#58a6ff', '#d2a8ff', '#ff7bcb', '#ff7b72', '#ffa657', '#e3b341', '#7ee787', '#79c0ff'],
      tints: ['#102a43', '#271a3d', '#32182b', '#35191b', '#362318', '#332b17', '#142e22', '#1c2736'],
      accent: '#1f6feb',
      rootText: '#ffffff',
      text: '#e6edf3',
      muted: '#7ee787',
      background: '#1d1f21'
    }
  };

  function safeGet(key) {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }

  function safeSet(key, value) {
    try { localStorage.setItem(key, value); } catch (e) {}
  }

  function palette() {
    return document.documentElement.getAttribute('data-theme') === 'dark' ? PALETTES.dark : PALETTES.light;
  }

  function childrenOf(node) {
    return Array.isArray(node.children) ? node.children : [];
  }

  function leavesOf(node) {
    var children = childrenOf(node);
    if (!children.length) return 1;
    return children.reduce(function (total, child) { return total + leavesOf(child); }, 0);
  }

  function walk(node, callback) {
    callback(node);
    childrenOf(node).forEach(function (child) { walk(child, callback); });
  }

  function svgElement(name, attrs) {
    var element = document.createElementNS(SVG_NS, name);
    Object.keys(attrs || {}).forEach(function (key) { element.setAttribute(key, attrs[key]); });
    return element;
  }

  function textUnits(text) {
    return Array.from(text).reduce(function (total, character) {
      return total + (/[^\u0000-\u00ff]/.test(character) ? 2 : 1);
    }, 0);
  }

  function wrapText(text) {
    var chars = Array.from(text);
    var lines = [], line = '', units = 0, overflow = false;
    chars.forEach(function (character) {
      var amount = textUnits(character);
      if (units + amount > 36 && line) {
        lines.push(line.trim());
        line = '';
        units = 0;
        if (lines.length === 2) overflow = true;
      }
      if (lines.length < 3) {
        line += character;
        units += amount;
      } else {
        overflow = true;
      }
    });
    if (line && lines.length < 3) lines.push(line.trim());
    if (!lines.length) lines.push('');
    if (overflow) {
      var last = Array.from(lines[2] || lines[1] || lines[0]);
      var target = lines.length > 2 ? 2 : lines.length - 1;
      while (last.length && textUnits(last.join('')) > 33) last.pop();
      lines[target] = last.join('').replace(/\s+$/, '') + '…';
    }
    return lines.slice(0, 3);
  }

  function makeMeasureContext() {
    var canvas = document.createElement('canvas');
    return canvas.getContext && canvas.getContext('2d');
  }

  function measureText(context, text, font) {
    if (!context) return textUnits(text) * 7;
    context.font = font;
    return context.measureText(text).width;
  }

  function measureTree(root, colors, context) {
    var rootChildren = childrenOf(root);
    var infos = [];

    function measureNode(node, depth, branchIndex) {
      var label = String(node.text || '');
      var fontSize = depth === 0 ? 16 : depth === 1 ? 14 : 13;
      var fontWeight = depth < 2 ? 600 : 400;
      var font = fontWeight + ' ' + fontSize + 'px ' + FONT;
      var lines = wrapText(label);
      var widths = lines.map(function (line) { return measureText(context, line, font); });
      if (node.read && widths.length) widths[widths.length - 1] += 18;
      var maxWidth = Math.max.apply(null, widths.concat([24]));
      var box = depth < 2;
      var lineHeight = fontSize * 1.4;
      var height = Math.max(box ? 32 : lineHeight + 4, lines.length * lineHeight + (box ? 16 : 4));
      var info = {
        node: node,
        depth: depth,
        branchIndex: branchIndex,
        color: depth === 0 ? colors.accent : colors.branches[branchIndex % colors.branches.length],
        lines: lines,
        fontSize: fontSize,
        fontWeight: fontWeight,
        lineHeight: lineHeight,
        textWidth: Math.max.apply(null, widths),
        width: maxWidth + (box ? 28 : 8),
        height: height,
        children: [],
        totalHeight: height
      };
      infos.push(info);
      var sourceChildren = childrenOf(node);
      if (sourceChildren.length && !node.collapsed) {
        info.children = sourceChildren.map(function (child, index) {
          return measureNode(child, depth + 1, depth === 0 ? index : branchIndex);
        });
        var childrenHeight = info.children.reduce(function (total, child) { return total + child.totalHeight; }, 0);
        childrenHeight += Math.max(0, info.children.length - 1) * 20;
        info.totalHeight = Math.max(height, childrenHeight);
      }
      return info;
    }

    var rootInfo = measureNode(root, 0, -1);
    var branches = rootInfo.children.map(function (info, index) {
      return { info: info, index: index, leaves: leavesOf(info.node), side: 0 };
    });
    var sideHeights = [0, 0];
    var sides = [[], []];
    branches.forEach(function (branch) {
      var side = sideHeights[0] <= sideHeights[1] ? 0 : 1;
      branch.side = side;
      sides[side].push(branch);
      sideHeights[side] += branch.leaves;
    });

    var sideCenters = [[], []];
    sides.forEach(function (items, side) {
      var widths = [];
      items.forEach(function (branch) {
        (function collect(info) {
          widths[info.depth] = Math.max(widths[info.depth] || 0, info.width);
          info.children.forEach(collect);
        })(branch.info);
      });
      var direction = side === 0 ? -1 : 1;
      var distance = rootInfo.width / 2 + 72;
      for (var depth = 1; depth <= widths.length; depth++) {
        var width = widths[depth] || 0;
        sideCenters[side][depth] = direction * (distance + width / 2);
        distance += width + 66;
      }
    });

    function position(info, side, x, y) {
      info.x = x;
      info.y = y;
      if (!info.children.length) return;
      var total = info.children.reduce(function (sum, child) { return sum + child.totalHeight; }, 0);
      total += Math.max(0, info.children.length - 1) * 20;
      var cursor = y - total / 2;
      info.children.forEach(function (child) {
        var childY = cursor + child.totalHeight / 2;
        position(child, side, sideCenters[side][child.depth], childY);
        cursor += child.totalHeight + 20;
      });
    }

    position(rootInfo, 1, 0, 0);
    sides.forEach(function (items, side) {
      var total = items.reduce(function (sum, branch) { return sum + branch.info.totalHeight; }, 0);
      total += Math.max(0, items.length - 1) * 26;
      var cursor = -total / 2;
      items.forEach(function (branch) {
        var y = cursor + branch.info.totalHeight / 2;
        position(branch.info, side, sideCenters[side][1], y);
        cursor += branch.info.totalHeight + 26;
      });
    });

    var minX = 0, maxX = 0, minY = 0, maxY = 0;
    infos.forEach(function (info) {
      minX = Math.min(minX, info.x - info.width / 2);
      maxX = Math.max(maxX, info.x + info.width / 2);
      minY = Math.min(minY, info.y - info.height / 2);
      maxY = Math.max(maxY, info.y + info.height / 2);
    });
    var padding = 40;
    var offsetX = padding - minX, offsetY = padding - minY;
    infos.forEach(function (info) { info.x += offsetX; info.y += offsetY; });
    return {
      root: rootInfo,
      nodes: infos,
      width: maxX - minX + padding * 2,
      height: maxY - minY + padding * 2
    };
  }

  function createToolbar(container, options, instance) {
    var toolbar = document.createElement('div');
    toolbar.className = 'mm-controls';
    toolbar.setAttribute('aria-label', '思维导图操作');
    function add(label, className, handler) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'mm-control ' + className;
      button.textContent = label;
      button.title = label;
      button.addEventListener('click', handler);
      toolbar.appendChild(button);
      return button;
    }
    add('放大', 'mm-map-only', function () { instance.zoomAt(1.2); });
    add('缩小', 'mm-map-only', function () { instance.zoomAt(1 / 1.2); });
    add('适应窗口', 'mm-map-only', function () { instance.fit(); });
    add('全部展开', 'mm-expand', function () { instance.expandAll(false); });
    add('全部收起', 'mm-collapse', function () { instance.expandAll(true); });
    add('存为图片', 'mm-map-only', function () { instance.exportPng(); });
    if (options.fullscreen) {
      add('全屏', 'mm-fullscreen', function () {
        open({
          title: options.title,
          tree: options.tree,
          mode: 'map',
          filename: options.filename,
          fullscreen: true,
          onNavigate: options.onNavigate
        });
      });
    }
    container.appendChild(toolbar);
    return { element: toolbar };
  }

  function mount(container, options) {
    options = options || {};
    var tree = options.tree || { text: options.title || '大纲', children: [] };
    var workspace = document.createElement('div');
    workspace.className = 'mm-workspace';
    var toolbarInfo = createToolbar(workspace, options, {
      zoomAt: zoomAt,
      fit: fit,
      expandAll: expandAll,
      exportPng: exportPng
    });
    var outline = document.createElement('div');
    outline.className = 'mm-outline';
    var mapPanel = document.createElement('div');
    mapPanel.className = 'mm-map-panel';
    var viewport = document.createElement('div');
    viewport.className = 'mm-map-viewport';
    viewport.setAttribute('aria-label', options.title || '思维导图');
    mapPanel.appendChild(viewport);
    workspace.appendChild(outline);
    workspace.appendChild(mapPanel);
    container.textContent = '';
    container.appendChild(workspace);

    var context = makeMeasureContext();
    var colors = palette();
    var layout = null, svg = null, contentGroup = null;
    var scale = 1, panX = 0, panY = 0, hasView = false;
    var pointerPositions = Object.create(null), pointerIds = [];
    var panStart = null, pinchStart = null, moved = false, ignoreClick = false;
    var mode = options.mode === 'outline' ? 'outline' : 'map';
    var observer = null;
    var statusToast = null, statusTimer = null;

    function applyTransform() {
      if (contentGroup) contentGroup.setAttribute('transform', 'matrix(' + scale + ',0,0,' + scale + ',' + panX + ',' + panY + ')');
    }

    function fit() {
      if (!svg || !layout) return;
      var width = viewport.clientWidth, height = viewport.clientHeight;
      if (!width || !height) return;
      var topInset = toolbarInfo.element.offsetHeight + 12;
      var availableHeight = Math.max(1, height - topInset - 12);
      scale = Math.min((width - 56) / layout.width, availableHeight / layout.height, 1.5);
      scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
      panX = (width - layout.width * scale) / 2;
      panY = topInset + (availableHeight - layout.height * scale) / 2;
      hasView = true;
      applyTransform();
    }

    function initialView() {
      if (!svg || !layout) return;
      var width = viewport.clientWidth, height = viewport.clientHeight;
      if (!width || !height) return;
      var topInset = toolbarInfo.element.offsetHeight + 12;
      var availableHeight = Math.max(1, height - topInset - 12);
      var fittedScale = Math.min((width - 56) / layout.width, availableHeight / layout.height, 1.5);
      if (fittedScale >= 0.6) {
        fit();
        return;
      }
      scale = 0.85;
      var rootHalfWidth = layout.root.width * scale / 2;
      var minimumCenter = rootHalfWidth + 12;
      var maximumCenter = width - rootHalfWidth - 12;
      var rootCenter = width * 0.36;
      rootCenter = maximumCenter >= minimumCenter
        ? Math.min(maximumCenter, Math.max(minimumCenter, rootCenter))
        : Math.max(minimumCenter, rootCenter);
      panX = rootCenter - layout.root.x * scale;
      panY = topInset + availableHeight / 2 - layout.root.y * scale;
      hasView = true;
      applyTransform();
    }

    function zoomAt(factor, clientX, clientY) {
      if (!svg) return;
      var rect = viewport.getBoundingClientRect();
      var x = typeof clientX === 'number' ? clientX - rect.left : rect.width / 2;
      var y = typeof clientY === 'number' ? clientY - rect.top : rect.height / 2;
      var next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
      var worldX = (x - panX) / scale, worldY = (y - panY) / scale;
      scale = next;
      panX = x - worldX * scale;
      panY = y - worldY * scale;
      hasView = true;
      applyTransform();
    }

    function toggleNode(node, anchor) {
      node.collapsed = !node.collapsed;
      renderMap(anchor || node);
      renderOutline();
    }

    function renderMap(anchorNode) {
      colors = palette();
      layout = measureTree(tree, colors, context);
      var anchor = null;
      if (anchorNode && hasView) {
        layout.nodes.some(function (info) {
          if (info.node !== anchorNode) return false;
          var old = currentNodePositions.filter(function (position) { return position.node === anchorNode; })[0];
          if (old) anchor = { x: old.x * scale + panX, y: old.y * scale + panY };
          return true;
        });
      }
      currentNodePositions = layout.nodes.map(function (info) { return { node: info.node, x: info.x, y: info.y }; });
      var nextSvg = svgElement('svg', {
        'class': 'mm-map-svg',
        'xmlns': SVG_NS,
        'width': '100%',
        'height': '100%',
        'font-family': FONT,
        'role': 'img',
        'aria-label': options.title || '思维导图'
      });
      var group = svgElement('g', {});
      layout.nodes.forEach(function (info) {
        info.children.forEach(function (child) {
          var direction = child.x < info.x ? -1 : 1;
          var startX = info.x + direction * info.width / 2;
          var endX = child.x - direction * child.width / 2;
          var midX = (startX + endX) / 2;
          var edge = svgElement('path', {
            d: 'M' + startX + ' ' + info.y + ' C' + midX + ' ' + info.y + ', ' + midX + ' ' + child.y + ', ' + endX + ' ' + child.y,
            fill: 'none',
            stroke: child.color,
            'stroke-width': child.depth === 1 ? 2 : 1,
            'stroke-linecap': 'round'
          });
          group.appendChild(edge);
        });
      });
      layout.nodes.forEach(function (info) {
        var node = info.node;
        var nodeGroup = svgElement('g', {
          'class': 'mm-node mm-depth-' + info.depth,
          'data-depth': info.depth,
          transform: 'translate(' + info.x + ' ' + info.y + ')'
        });
        var action = childrenOf(node).length > 0 && node.toggleOnClick || node.id || node.href;
        if (action) {
          nodeGroup.setAttribute('tabindex', '0');
          nodeGroup.setAttribute('role', 'button');
          nodeGroup.setAttribute('aria-label', node.text + (node.read ? '，已读' : ''));
        }
        var title = svgElement('title', {});
        title.textContent = String(node.text || '') + (node.read ? ' · 已读' : '');
        nodeGroup.appendChild(title);
        if (info.depth < 2) {
          nodeGroup.appendChild(svgElement('rect', {
            x: -info.width / 2,
            y: -info.height / 2,
            width: info.width,
            height: info.height,
            rx: info.depth === 0 ? 12 : 9,
            fill: info.depth === 0 ? colors.accent : colors.tints[info.branchIndex % colors.tints.length],
            stroke: info.depth === 0 ? colors.accent : info.color,
            'stroke-width': info.depth === 0 ? 1 : 1.6
          }));
        }
        var text = svgElement('text', {
          'text-anchor': 'middle',
          'font-family': FONT,
          'font-size': info.fontSize,
          'font-weight': info.fontWeight,
          fill: info.depth === 0 ? colors.rootText : colors.text
        });
        var startY = -((info.lines.length - 1) * info.lineHeight) / 2 + info.fontSize * 0.35;
        info.lines.forEach(function (line, index) {
          var span = svgElement('tspan', { x: 0, y: startY + index * info.lineHeight });
          span.textContent = line;
          if (node.read && index === info.lines.length - 1) {
            var check = svgElement('tspan', { fill: colors.muted, dx: 5, 'font-weight': 700 });
            check.textContent = '✓';
            span.appendChild(check);
          }
          text.appendChild(span);
        });
        nodeGroup.appendChild(text);
        if (info.depth >= 2) {
          var underlineWidth = Math.max(24, info.textWidth);
          nodeGroup.appendChild(svgElement('line', {
            x1: -underlineWidth / 2,
            x2: underlineWidth / 2,
            y1: info.height / 2 - 1,
            y2: info.height / 2 - 1,
            stroke: info.color,
            'stroke-width': 1.4,
            'stroke-linecap': 'round'
          }));
        }
        if (childrenOf(node).length) {
          var direction = info.depth === 0 ? 1 : (info.x < layout.root.x + layout.root.width / 2 ? -1 : 1);
          var toggle = svgElement('g', {
            'class': 'mm-node-toggle',
            transform: 'translate(' + (direction * (info.width / 2 + 8)) + ' -2)'
          });
          toggle.appendChild(svgElement('circle', {
            r: 8,
            fill: colors.background,
            stroke: info.color,
            'stroke-width': 1.4
          }));
          var sign = svgElement('text', {
            'text-anchor': 'middle',
            'dominant-baseline': 'central',
            'font-family': FONT,
            'font-size': 13,
            'font-weight': 600,
            fill: info.color
          });
          sign.textContent = node.collapsed ? '+' : '−';
          toggle.appendChild(sign);
          toggle.setAttribute('role', 'button');
          toggle.setAttribute('tabindex', '0');
          toggle.setAttribute('aria-label', node.collapsed ? '展开 ' + node.text : '收起 ' + node.text);
          toggle.addEventListener('click', function (event) {
            event.stopPropagation();
            toggleNode(node, node);
          });
          toggle.addEventListener('keydown', function (event) {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              event.stopPropagation();
              toggleNode(node, node);
            }
          });
          nodeGroup.appendChild(toggle);
        }
        function activate(event) {
          if (ignoreClick || (event.type === 'keydown' && event.key !== 'Enter' && event.key !== ' ')) return;
          if (event.type === 'keydown') event.preventDefault();
          if (childrenOf(node).length && node.toggleOnClick) {
            toggleNode(node, node);
            return;
          }
          if ((node.id || node.href) && typeof options.onNavigate === 'function') {
            options.onNavigate(node);
          } else if (node.href) {
            window.location.href = node.href;
          }
        }
        nodeGroup.addEventListener('click', activate);
        nodeGroup.addEventListener('keydown', activate);
        group.appendChild(nodeGroup);
      });
      nextSvg.appendChild(group);
      viewport.textContent = '';
      viewport.appendChild(nextSvg);
      svg = nextSvg;
      contentGroup = group;
      applyTransform();
      if (anchor) {
        var newAnchor = layout.nodes.filter(function (info) { return info.node === anchorNode; })[0];
        if (newAnchor) {
          panX = anchor.x - newAnchor.x * scale;
          panY = anchor.y - newAnchor.y * scale;
          applyTransform();
        }
      } else if (!hasView) {
        window.requestAnimationFrame(function () { if (!hasView) initialView(); });
      }
    }

    var currentNodePositions = [];

    function renderOutline() {
      outline.textContent = '';
      var list = document.createElement('ul');
      list.className = 'mm-outline-list';
      function addNode(parent, node) {
        var item = document.createElement('li');
        item.className = 'mm-outline-item';
        var kids = childrenOf(node);
        if (kids.length) {
          var disclosure = document.createElement('button');
          disclosure.type = 'button';
          disclosure.className = 'mm-outline-toggle';
          disclosure.textContent = node.collapsed ? '▸' : '▾';
          disclosure.setAttribute('aria-expanded', String(!node.collapsed));
          disclosure.setAttribute('aria-label', (node.collapsed ? '展开 ' : '收起 ') + node.text);
          disclosure.addEventListener('click', function () {
            node.collapsed = !node.collapsed;
            renderOutline();
            renderMap(node);
          });
          item.appendChild(disclosure);
        } else {
          var spacer = document.createElement('span');
          spacer.className = 'mm-outline-spacer';
          item.appendChild(spacer);
        }
        var label;
        if (node.href && !node.toggleOnClick) {
          label = document.createElement('a');
          label.href = node.href;
          label.addEventListener('click', function (event) {
            if (event.metaKey || event.ctrlKey || event.shiftKey) return;
            if (typeof options.onNavigate !== 'function') return;
            event.preventDefault();
            options.onNavigate(node);
          });
        } else {
          label = document.createElement('button');
          label.type = 'button';
          if (node.toggleOnClick && kids.length) {
            label.addEventListener('click', function () {
              node.collapsed = !node.collapsed;
              renderOutline();
              renderMap(node);
            });
          } else if ((node.id || node.href) && typeof options.onNavigate === 'function') {
            label.addEventListener('click', function () { options.onNavigate(node); });
          }
        }
        label.className = 'mm-outline-label';
        label.appendChild(document.createTextNode(String(node.text || '')));
        if (node.read) {
          var mark = document.createElement('span');
          mark.className = 'mm-read';
          mark.textContent = ' ✓';
          label.appendChild(mark);
        }
        item.appendChild(label);
        if (kids.length) {
          var nested = document.createElement('ul');
          nested.hidden = !!node.collapsed;
          kids.forEach(function (child) { addNode(nested, child); });
          item.appendChild(nested);
        }
        parent.appendChild(item);
      }
      addNode(list, tree);
      outline.appendChild(list);
    }

    function setMode(nextMode, remember) {
      mode = nextMode === 'outline' ? 'outline' : 'map';
      outline.hidden = mode !== 'outline';
      mapPanel.hidden = mode !== 'map';
      Array.prototype.forEach.call(toolbarInfo.element.querySelectorAll('.mm-map-only'), function (button) {
        button.hidden = mode !== 'map';
      });
      if (remember) safeSet(MODE_KEY, mode);
      if (mode === 'map') window.requestAnimationFrame(function () { if (!hasView) initialView(); });
      if (typeof options.onMode === 'function') options.onMode(mode);
    }

    function expandAll(collapsed) {
      walk(tree, function (node) {
        if (childrenOf(node).length) node.collapsed = collapsed;
      });
      renderMap();
      renderOutline();
    }

    function setStatus(message) {
      if (!statusToast || !statusToast.parentNode) {
        statusToast = document.createElement('div');
        statusToast.className = 'mm-toast';
        statusToast.setAttribute('role', 'status');
        statusToast.setAttribute('aria-live', 'polite');
        var host = container.closest('.mm-overlay') || document.body;
        host.appendChild(statusToast);
      }
      statusToast.textContent = message;
      statusToast.classList.add('is-visible');
      if (statusTimer) window.clearTimeout(statusTimer);
      statusTimer = window.setTimeout(function () {
        if (!statusToast) return;
        statusToast.classList.remove('is-visible');
        statusTimer = window.setTimeout(function () {
          if (statusToast && statusToast.parentNode) statusToast.parentNode.removeChild(statusToast);
          statusToast = null;
          statusTimer = null;
        }, 180);
      }, 2000);
    }

    function exportPng() {
      if (!svg || !layout) return;
      var clone = svg.cloneNode(true);
      var content = clone.querySelector('g');
      content.setAttribute('transform', 'matrix(1,0,0,1,0,0)');
      clone.setAttribute('viewBox', '0 0 ' + layout.width + ' ' + layout.height);
      clone.setAttribute('width', layout.width);
      clone.setAttribute('height', layout.height);
      clone.setAttribute('font-family', FONT);
      var background = svgElement('rect', {
        x: 0, y: 0, width: layout.width, height: layout.height,
        fill: colors.background
      });
      clone.insertBefore(background, content);
      var source = new XMLSerializer().serializeToString(clone);
      var sourceUrl = URL.createObjectURL(new Blob([source], { type: 'image/svg+xml;charset=utf-8' }));
      var image = new Image();
      image.onload = function () {
        var factor = Math.min(2, 8192 / Math.max(layout.width, layout.height));
        var canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(layout.width * factor));
        canvas.height = Math.max(1, Math.round(layout.height * factor));
        var context2d = canvas.getContext('2d');
        if (!context2d) {
          URL.revokeObjectURL(sourceUrl);
          setStatus('图片导出失败');
          return;
        }
        context2d.drawImage(image, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function (blob) {
          URL.revokeObjectURL(sourceUrl);
          if (!blob) {
            setStatus('图片导出失败');
            return;
          }
          var link = document.createElement('a');
          link.href = URL.createObjectURL(blob);
          link.download = (options.filename || 'mindmap') + '.png';
          link.click();
          window.setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
          setStatus('图片已保存');
        }, 'image/png');
      };
      image.onerror = function () {
        URL.revokeObjectURL(sourceUrl);
        setStatus('图片导出失败');
      };
      image.src = sourceUrl;
    }

    function point(event) {
      return { x: event.clientX, y: event.clientY };
    }

    function distance(a, b) {
      var dx = a.x - b.x, dy = a.y - b.y;
      return Math.sqrt(dx * dx + dy * dy);
    }

    function midpoint(a, b) {
      return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    }

    function capturePointer(pointerId) {
      try { viewport.setPointerCapture(pointerId); } catch (e) {}
    }

    function startPinch() {
      var a = pointerPositions[pointerIds[0]], b = pointerPositions[pointerIds[1]];
      var mid = midpoint(a, b), rect = viewport.getBoundingClientRect();
      pinchStart = {
        distance: Math.max(1, distance(a, b)),
        scale: scale,
        midX: mid.x - rect.left,
        midY: mid.y - rect.top,
        worldX: (mid.x - rect.left - panX) / scale,
        worldY: (mid.y - rect.top - panY) / scale
      };
      panStart = null;
      moved = true;
      pointerIds.forEach(capturePointer);
    }

    viewport.addEventListener('pointerdown', function (event) {
      if (event.button !== undefined && event.button !== 0) return;
      pointerPositions[event.pointerId] = point(event);
      pointerIds.push(event.pointerId);
      if (pointerIds.length === 1) {
        panStart = { point: point(event), x: panX, y: panY };
        moved = false;
      } else if (pointerIds.length === 2) {
        startPinch();
      }
    });
    viewport.addEventListener('pointermove', function (event) {
      if (!Object.prototype.hasOwnProperty.call(pointerPositions, event.pointerId)) return;
      pointerPositions[event.pointerId] = point(event);
      if (pointerIds.length > 1 && pinchStart) {
        var a = pointerPositions[pointerIds[0]], b = pointerPositions[pointerIds[1]];
        var mid = midpoint(a, b), rect = viewport.getBoundingClientRect();
        var next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, pinchStart.scale * distance(a, b) / pinchStart.distance));
        scale = next;
        panX = mid.x - rect.left - pinchStart.worldX * scale;
        panY = mid.y - rect.top - pinchStart.worldY * scale;
        applyTransform();
        moved = true;
      } else if (panStart) {
        var dx = event.clientX - panStart.point.x, dy = event.clientY - panStart.point.y;
        if (!moved && Math.sqrt(dx * dx + dy * dy) > 4) {
          moved = true;
          capturePointer(event.pointerId);
        }
        if (moved) {
          panX = panStart.x + dx;
          panY = panStart.y + dy;
          applyTransform();
        }
      }
    });
    function endPointer(event) {
      if (!Object.prototype.hasOwnProperty.call(pointerPositions, event.pointerId)) return;
      delete pointerPositions[event.pointerId];
      pointerIds = pointerIds.filter(function (id) { return id !== event.pointerId; });
      if (pointerIds.length === 1) {
        var remaining = pointerPositions[pointerIds[0]];
        panStart = { point: remaining, x: panX, y: panY };
        pinchStart = null;
      } else if (!pointerIds.length) {
        panStart = null;
        pinchStart = null;
        if (moved) {
          ignoreClick = true;
          window.setTimeout(function () { ignoreClick = false; }, 0);
        }
        hasView = true;
      }
    }
    viewport.addEventListener('pointerup', endPointer);
    viewport.addEventListener('pointercancel', endPointer);
    viewport.addEventListener('wheel', function (event) {
      event.preventDefault();
      zoomAt(event.deltaY < 0 ? 1.1 : 1 / 1.1, event.clientX, event.clientY);
    }, { passive: false });

    renderOutline();
    renderMap();
    setMode(mode, false);
    window.requestAnimationFrame(function () { if (!hasView) initialView(); });
    if (window.MutationObserver) {
      observer = new MutationObserver(function (records) {
        if (records.some(function (record) { return record.attributeName === 'data-theme'; })) renderMap();
      });
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    }
    if (window.ResizeObserver) {
      var resizeObserver = new ResizeObserver(function () { if (!hasView) initialView(); });
      resizeObserver.observe(viewport);
      var disconnect = observer;
      observer = {
        disconnect: function () {
          resizeObserver.disconnect();
          if (disconnect) disconnect.disconnect();
        }
      };
    }

    return {
      setMode: setMode,
      fit: fit,
      refresh: function () { renderMap(); renderOutline(); },
      destroy: function () {
        if (observer) observer.disconnect();
        if (statusTimer) window.clearTimeout(statusTimer);
        if (statusToast && statusToast.parentNode) statusToast.parentNode.removeChild(statusToast);
        statusToast = null;
      }
    };
  }

  function open(options) {
    options = options || {};
    var opener = document.activeElement;
    var body = document.body;
    var previousOverflow = body.style.overflow;
    var previousMode = safeGet(MODE_KEY);
    var previousFullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
    var initialMode = previousMode === 'outline' || previousMode === 'map' ? previousMode :
      (options.mode === 'outline' ? 'outline' : 'map');
    var overlay = document.createElement('div');
    overlay.className = 'mm-overlay';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    var title = document.createElement('h2');
    title.className = 'mm-title';
    title.id = 'mm-dialog-title';
    title.textContent = options.title || '大纲 / 思维导图';
    overlay.setAttribute('aria-labelledby', title.id);
    var header = document.createElement('div');
    header.className = 'mm-header';
    header.appendChild(title);
    var tabs = document.createElement('div');
    tabs.className = 'mm-tabs';
    var tabButtons = {};
    ['outline', 'map'].forEach(function (mode) {
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = mode === 'outline' ? '大纲' : '导图';
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', function () { if (controller) controller.setMode(mode, true); });
      tabButtons[mode] = button;
      tabs.appendChild(button);
    });
    header.appendChild(tabs);
    var closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'mm-close';
    closeButton.textContent = '×';
    closeButton.title = '关闭';
    closeButton.setAttribute('aria-label', '关闭');
    header.appendChild(closeButton);
    var content = document.createElement('div');
    content.className = 'mm-dialog-content';
    overlay.appendChild(header);
    overlay.appendChild(content);
    body.appendChild(overlay);
    var fullscreenEntered = false;
    var closing = false;
    var closeFinished = false;
    var controller = null;
    function finishClose() {
      if (closeFinished || !overlay.parentNode) return;
      closeFinished = true;
      document.removeEventListener('fullscreenchange', onFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', onFullscreenChange);
      window.removeEventListener('keydown', onKeydown, true);
      if (controller) controller.destroy();
      overlay.remove();
      body.style.overflow = previousOverflow;
      if (opener && opener.focus) opener.focus();
    }
    function onFullscreenChange() {
      var fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
      if (fullscreenElement === overlay) {
        if (!fullscreenEntered) {
          fullscreenEntered = true;
          if (controller) controller.fit();
        }
      } else if (fullscreenEntered) {
        if (closing) finishClose();
        else close();
      }
    }
    if (options.fullscreen) {
      var requestFullscreen = overlay.requestFullscreen || overlay.webkitRequestFullscreen;
      if (requestFullscreen) {
        document.addEventListener('fullscreenchange', onFullscreenChange);
        document.addEventListener('webkitfullscreenchange', onFullscreenChange);
        try {
          var request = requestFullscreen.call(overlay);
          if (request && typeof request.catch === 'function') request.catch(function () {});
        } catch (e) {}
      }
    }
    body.style.overflow = 'hidden';
    var originalNavigate = options.onNavigate;
    function setTabs(mode) {
      Object.keys(tabButtons).forEach(function (key) {
        tabButtons[key].classList.toggle('is-active', key === mode);
        tabButtons[key].setAttribute('aria-pressed', String(key === mode));
      });
    }
    function close() {
      if (!overlay.parentNode || closing) return;
      closing = true;
      var fullscreenElement = document.fullscreenElement || document.webkitFullscreenElement;
      if (fullscreenElement === overlay) {
        var restoreFullscreen = previousFullscreenElement &&
          (previousFullscreenElement.requestFullscreen || previousFullscreenElement.webkitRequestFullscreen);
        if (restoreFullscreen) {
          try {
            var restore = restoreFullscreen.call(previousFullscreenElement);
            if (restore && typeof restore.then === 'function') restore.then(finishClose, finishClose);
            else window.setTimeout(finishClose, 1000);
            return;
          } catch (e) {}
        }
        var exitFullscreen = document.exitFullscreen || document.webkitExitFullscreen;
        if (exitFullscreen) {
          try {
            var exit = exitFullscreen.call(document);
            if (exit && typeof exit.catch === 'function') exit.catch(function () {});
          } catch (e) {}
        }
      }
      finishClose();
    }
    function onKeydown(event) {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        close();
      }
    }
    var mountOptions = {};
    Object.keys(options).forEach(function (key) { mountOptions[key] = options[key]; });
    mountOptions.mode = initialMode;
    mountOptions.fullscreen = false;
    mountOptions.onMode = setTabs;
    mountOptions.onNavigate = function (node) {
      close();
      if (typeof originalNavigate === 'function') originalNavigate(node);
      else if (node.href) window.location.href = node.href;
    };
    controller = mount(content, mountOptions);
    if (fullscreenEntered || document.fullscreenElement === overlay || document.webkitFullscreenElement === overlay) {
      fullscreenEntered = true;
      controller.fit();
    }
    controller.setMode(initialMode, false);
    closeButton.addEventListener('click', close);
    window.addEventListener('keydown', onKeydown, true);
    setTabs(initialMode);
    closeButton.focus();
    return { close: close };
  }

  function articleTree(headings, title) {
    var root = { text: title, children: [] };
    if (!headings.length) return root;
    var topLevel = Math.min.apply(null, headings.map(function (heading) { return heading.level; }));
    var stack = [{ level: topLevel - 1, node: root }];
    headings.forEach(function (heading) {
      while (stack.length > 1 && heading.level <= stack[stack.length - 1].level) stack.pop();
      var node = { text: heading.text, id: heading.id, children: [] };
      stack[stack.length - 1].node.children.push(node);
      stack.push({ level: heading.level, node: node });
    });
    return root;
  }

  function initArticleButtons() {
    var buttons = document.querySelectorAll('.catalog-mindmap');
    if (!buttons.length) return;
    var headings = window.BlogOutline ? window.BlogOutline.headings() : [];
    Array.prototype.forEach.call(buttons, function (button) {
      button.hidden = headings.length < 2;
      if (button.hidden) return;
      button.addEventListener('click', function () {
        var currentHeadings = window.BlogOutline ? window.BlogOutline.headings() : [];
        if (currentHeadings.length < 2) return;
        var postActions = document.querySelector('.post-actions[data-title]');
        var titleNode = document.querySelector('.post-heading h1');
        var title = (postActions && postActions.getAttribute('data-title')) ||
          (titleNode && titleNode.textContent.trim()) || document.title;
        var slug = window.location.pathname.replace(/\/+$/, '').split('/').pop().replace(/\.[^.]+$/, '') || 'article';
        open({
          title: title,
          tree: articleTree(currentHeadings, title),
          filename: slug + '-思维导图',
          mode: 'outline',
          onNavigate: function (node) {
            if (document.body.classList.contains('outline-open')) document.body.classList.remove('outline-open');
            var target = node.id && document.getElementById(node.id);
            if (!target) return;
            if (window.SectionFold) window.SectionFold.reveal(target);
            if (history.replaceState) history.replaceState(null, '', '#' + node.id);
            var top = target.getBoundingClientRect().top + window.pageYOffset - 80;
            var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            window.scrollTo({ top: Math.max(top, 0), behavior: reduced ? 'auto' : 'smooth' });
          }
        });
      });
    });
  }

  function seriesTreeFromDom() {
    var root = { text: '系列', children: [], toggleOnClick: true };
    Array.prototype.forEach.call(document.querySelectorAll('.series-tree > .st-map'), function (map) {
      var title = map.querySelector(':scope > summary .st-title');
      var mapNode = { text: title ? title.textContent.trim() : '学习地图', children: [], toggleOnClick: true };
      var series = map.querySelectorAll(':scope > .st-series > li > .st-item');
      Array.prototype.forEach.call(series, function (item) {
        var number = item.querySelector(':scope > summary .st-no');
        var name = item.querySelector(':scope > summary .st-name');
        var prefix = number ? number.textContent.trim() : '';
        var seriesNode = {
          text: [prefix, name ? name.textContent.trim() : item.querySelector('summary').textContent.trim()].filter(Boolean).join(' '),
          children: [],
          collapsed: true,
          toggleOnClick: true
        };
        Array.prototype.forEach.call(item.querySelectorAll(':scope > .st-posts > li[data-url]'), function (post) {
          var link = post.querySelector('a[href]');
          var url = post.getAttribute('data-url');
          if (!link || !url) return;
          seriesNode.children.push({
            text: link.textContent.trim(),
            href: link.href,
            url: url,
            read: !!(window.BlogReadProgress && window.BlogReadProgress.isRead(url)),
            children: []
          });
        });
        mapNode.children.push(seriesNode);
      });
      root.children.push(mapNode);
    });
    return root;
  }

  function initSeriesView() {
    var tree = document.querySelector('.series-tree');
    if (!tree || !window.BlogMindmap) return;
    var toggle = document.createElement('div');
    toggle.className = 'series-view-toggle';
    toggle.setAttribute('role', 'group');
    toggle.setAttribute('aria-label', '系列视图');
    var directoryButton = document.createElement('button');
    directoryButton.type = 'button';
    directoryButton.textContent = '目录';
    var mapButton = document.createElement('button');
    mapButton.type = 'button';
    mapButton.textContent = '导图';
    toggle.appendChild(directoryButton);
    toggle.appendChild(mapButton);
    var mapContainer = document.createElement('div');
    mapContainer.className = 'series-mindmap';
    mapContainer.hidden = true;
    tree.parentNode.insertBefore(toggle, tree);
    tree.parentNode.insertBefore(mapContainer, tree);
    var root = seriesTreeFromDom();
    var controller = null;

    function updateReadState() {
      if (!window.BlogReadProgress) return;
      walk(root, function (node) {
        if (node.url) node.read = window.BlogReadProgress.isRead(node.url);
      });
      if (controller) controller.refresh();
    }

    function setView(view, remember) {
      var isMap = view === 'map';
      tree.hidden = isMap;
      mapContainer.hidden = !isMap;
      directoryButton.classList.toggle('is-active', !isMap);
      mapButton.classList.toggle('is-active', isMap);
      directoryButton.setAttribute('aria-pressed', String(!isMap));
      mapButton.setAttribute('aria-pressed', String(isMap));
      if (remember) safeSet('seriesView', isMap ? 'map' : 'directory');
      if (isMap && !controller) {
        controller = window.BlogMindmap.mount(mapContainer, {
          title: '系列学习地图',
          tree: root,
          mode: 'map',
          filename: '系列-思维导图',
          fullscreen: true,
          onNavigate: function (node) {
            if (node.href) window.location.href = node.href;
          }
        });
      } else if (isMap && controller) {
        controller.refresh();
      }
    }

    directoryButton.addEventListener('click', function () { setView('directory', true); });
    mapButton.addEventListener('click', function () { setView('map', true); });
    document.addEventListener('blog:read-progress', updateReadState);
    window.addEventListener('storage', function (event) {
      if (event.key === 'blogRead' || event.key === null) updateReadState();
    });
    var hash = window.location.hash.slice(1);
    var hasDeepLink = !!(hash && document.getElementById(decodeURIComponent(hash)));
    var saved = safeGet('seriesView');
    setView(hasDeepLink ? 'directory' : saved === 'map' ? 'map' : 'directory', !hasDeepLink);
    window.addEventListener('hashchange', function () {
      var key = window.location.hash.slice(1);
      if (key && document.getElementById(decodeURIComponent(key))) setView('directory', false);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initArticleButtons);
    document.addEventListener('DOMContentLoaded', initSeriesView);
  } else {
    initArticleButtons();
    initSeriesView();
  }

  window.BlogMindmap = { open: open, mount: mount };
})();
