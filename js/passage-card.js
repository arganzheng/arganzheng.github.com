/*!
 * passage-card.js — canvas share images for selected passages.
 * Runs before annotations.js so passage sharing can use BlogPassageCard.render.
 */
(function () {
  'use strict';

  var WIDTH = 1080, PAD = 120, TEXT_WIDTH = WIDTH - PAD * 2;
  var FONT_SIZE = 54, LINE_HEIGHT = 92;
  var SERIF = '54px "Songti SC", "STSong", "Noto Serif SC", "Source Han Serif SC", serif';
  var SANS = '-apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif';
  var START_PUNCTUATION = /^[，。、：；！？）」』》”’…]$/;
  var scriptVersion = (document.currentScript && (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';
  var qrScript = null;

  function loadQRLibrary() {
    if (window.qrcode) return Promise.resolve();
    if (!qrScript) {
      qrScript = new Promise(function (resolve, reject) {
        var script = document.createElement('script');
        script.src = '/js/vendor/qrcode.min.js' + (scriptVersion ? '?v=' + scriptVersion : '');
        script.onload = resolve;
        script.onerror = function () { reject(new Error('二维码加载失败')); };
        document.head.appendChild(script);
      });
    }
    return qrScript;
  }

  function qrImage(url) {
    return loadQRLibrary().then(function () {
      var library = window.qrcode, qr = library(0, 'M');
      var encoder = library.stringToBytesFuncs && library.stringToBytesFuncs['UTF-8'];
      if (encoder) {
        var originalEncoder = library.stringToBytes;
        library.stringToBytes = encoder;
        try { qr.addData(url); } finally { library.stringToBytes = originalEncoder; }
      } else qr.addData(url);
      qr.make();
      var svg = qr.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
      var objectUrl = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
      return new Promise(function (resolve) {
        var image = new Image();
        image.onload = function () { URL.revokeObjectURL(objectUrl); resolve(image); };
        image.onerror = function () { URL.revokeObjectURL(objectUrl); resolve(null); };
        image.src = objectUrl;
      });
    });
  }

  function avatarImage() {
    var bar = document.querySelector('.post-actions[data-avatar]');
    var src = bar && bar.getAttribute('data-avatar');
    if (!src) return Promise.resolve(null);
    var url;
    try { url = new URL(src, location.href); } catch (e) { return Promise.resolve(null); }
    if (url.origin !== location.origin) return Promise.resolve(null);
    return new Promise(function (resolve) {
      var image = new Image();
      image.onload = function () { resolve(image); };
      image.onerror = function () { resolve(null); };
      image.src = url.href;
    });
  }

  function sourceChars(o) {
    var before = String(o.before || ''), quote = String(o.quote || ''), after = String(o.after || '');
    var raw = before + quote + after, quoteStart = before.length, quoteEnd = quoteStart + quote.length;
    var chars = [], points = Array.from(raw), offset = 0;
    for (var i = 0; i < points.length; i++) {
      var ch = points[i], start = offset;
      offset += ch.length;
      if (/\s/.test(ch)) {
        var quotedSpace = start < quoteEnd && offset > quoteStart;
        while (i + 1 < points.length && /\s/.test(points[i + 1])) {
          i++;
          start = offset;
          offset += points[i].length;
          quotedSpace = quotedSpace || (start < quoteEnd && offset > quoteStart);
        }
        if (chars.length && chars[chars.length - 1].ch === ' ') {
          chars[chars.length - 1].quote = chars[chars.length - 1].quote || quotedSpace;
        } else if (chars.length) {
          chars.push({ ch: ' ', quote: quotedSpace });
        }
      } else {
        chars.push({ ch: ch, quote: start >= quoteStart && start < quoteEnd });
      }
    }
    while (chars.length && chars[0].ch === ' ') chars.shift();
    while (chars.length && chars[chars.length - 1].ch === ' ') chars.pop();
    return chars;
  }

  function lineText(line) {
    return line.chars.map(function (entry) { return entry.ch; }).join('');
  }

  function wordChar(ch) { return /^[A-Za-z0-9_.-]$/.test(ch); }

  function wrap(ctx, chars) {
    var lines = [{ chars: [] }];
    function current() { return lines[lines.length - 1]; }
    function add(entry) { current().chars.push(entry); }
    function nextLine() { if (current().chars.length) lines.push({ chars: [] }); }
    function fits(text) { return ctx.measureText(lineText(current()) + text).width <= TEXT_WIDTH; }
    function addChar(entry) {
      if (current().chars.length && !fits(entry.ch) && !START_PUNCTUATION.test(entry.ch)) nextLine();
      add(entry);
    }

    for (var i = 0; i < chars.length;) {
      var entry = chars[i];
      if (entry.ch === ' ') {
        if (current().chars.length && fits(' ')) add(entry);
        else if (current().chars.length) nextLine();
        i++;
        continue;
      }
      if (wordChar(entry.ch)) {
        var end = i + 1;
        while (end < chars.length && wordChar(chars[end].ch)) end++;
        var word = chars.slice(i, end), text = word.map(function (part) { return part.ch; }).join('');
        if (ctx.measureText(text).width <= TEXT_WIDTH) {
          if (current().chars.length && !fits(text)) nextLine();
          for (var w = 0; w < word.length; w++) add(word[w]);
        } else {
          for (var c = 0; c < word.length; c++) addChar(word[c]);
        }
        i = end;
        continue;
      }
      addChar(entry);
      i++;
    }
    while (lines.length > 1 && !current().chars.length) lines.pop();
    return lines.filter(function (line) { return line.chars.length; });
  }

  function quoteLineIndexes(lines) {
    var indexes = [];
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].chars.some(function (entry) { return entry.quote; })) indexes.push(i);
    }
    return indexes;
  }

  function trimLines(lines, ctx) {
    var quoteLines = quoteLineIndexes(lines), truncated = quoteLines.length > 10;
    if (truncated) {
      var cut = quoteLines[9];
      lines = lines.slice(0, cut + 1);
      while (lines[cut].chars.length && ctx.measureText(lineText(lines[cut]) + '…').width > TEXT_WIDTH) lines[cut].chars.pop();
      lines[cut].chars.push({ ch: '…', quote: true });
      quoteLines = quoteLineIndexes(lines);
    }
    if (!quoteLines.length) return { lines: lines, quoteLines: quoteLines };
    var first = quoteLines[0], last = quoteLines[quoteLines.length - 1];
    lines = lines.slice(Math.max(0, first - 2), Math.min(lines.length, last + 3));
    quoteLines = quoteLineIndexes(lines);
    return { lines: lines, quoteLines: quoteLines, truncated: truncated };
  }

  function distanceToQuote(lineIndex, quoteLines) {
    var distance = Infinity;
    for (var i = 0; i < quoteLines.length; i++) distance = Math.min(distance, Math.abs(lineIndex - quoteLines[i]));
    return distance;
  }

  function contextColor(distance) {
    return distance === 0 ? '#b8b8b8' : distance === 1 ? '#cdcdcd' : '#e0e0e0';
  }

  function drawLines(ctx, lines, quoteLines, topSpace) {
    ctx.font = SERIF;
    ctx.textBaseline = 'alphabetic';
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i], distance = distanceToQuote(i, quoteLines);
      var runs = [], color = null, text = '';
      for (var j = 0; j < line.chars.length; j++) {
        var entry = line.chars[j], nextColor = entry.quote ? '#1a1a1a' : contextColor(distance);
        if (color !== null && color !== nextColor) {
          runs.push({ text: text, color: color });
          text = '';
        }
        color = nextColor;
        text += entry.ch;
      }
      if (text) runs.push({ text: text, color: color });
      var prefix = '';
      for (var r = 0; r < runs.length; r++) {
        ctx.fillStyle = runs[r].color;
        ctx.fillText(runs[r].text, PAD + ctx.measureText(prefix).width, topSpace + FONT_SIZE + i * LINE_HEIGHT);
        prefix += runs[r].text;
      }
    }
  }

  function truncate(ctx, text, maxWidth) {
    var chars = Array.from(String(text || '').replace(/\s+/g, ' ').trim());
    ctx.font = '34px ' + SANS;
    var value = chars.join('');
    if (ctx.measureText(value).width <= maxWidth) return value;
    while (chars.length && ctx.measureText(chars.join('') + '…').width > maxWidth) chars.pop();
    return chars.join('') + '…';
  }

  function renderCard(o, assets) {
    var ctxProbe = document.createElement('canvas').getContext('2d');
    ctxProbe.font = SERIF;
    var lines = wrap(ctxProbe, sourceChars(o));
    var trimmed = trimLines(lines, ctxProbe);
    lines = trimmed.lines;
    var quoteLines = trimmed.quoteLines;
    var textH = lines.length * LINE_HEIGHT, gap = 180, footerH = 180, bottomPad = 120;
    var topSpace = 360;
    if (topSpace + textH + gap + footerH + bottomPad > 1440) topSpace = 200;
    var height = Math.max(1440, topSpace + textH + gap + footerH + bottomPad);
    var canvas = document.createElement('canvas');
    canvas.width = WIDTH;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = '#f7f7f7';
    ctx.fillRect(0, 0, WIDTH, height);
    drawLines(ctx, lines, quoteLines, topSpace);

    var footerTop = height - bottomPad - footerH;
    var avatar = assets.avatar, qr = assets.qr;
    var avatarX = PAD, avatarY = footerTop + (footerH - 120) / 2;
    if (avatar) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(avatarX + 60, avatarY + 60, 60, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(avatar, avatarX, avatarY, 120, 120);
      ctx.restore();
    }

    var qrX = WIDTH - PAD - 180, authorX = avatarX + 120 + 28;
    var titleWidth = Math.max(0, qrX - authorX - 32);
    ctx.fillStyle = '#1a1a1a';
    ctx.font = 'bold 40px ' + SANS;
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(o.author || 'arganzheng', authorX, footerTop + 58, titleWidth);
    ctx.fillStyle = '#333333';
    ctx.fillText(truncate(ctx, o.title, titleWidth), authorX, footerTop + 112);
    if (qr) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(qrX, footerTop, 180, 180);
      ctx.drawImage(qr, qrX, footerTop, 180, 180);
    }
    return new Promise(function (resolve, reject) {
      canvas.toBlob(function (blob) {
        if (blob) resolve(blob); else reject(new Error('图片生成失败'));
      }, 'image/png');
    });
  }

  function render(o) {
    o = o || {};
    return Promise.all([
      document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve(),
      avatarImage(),
      qrImage(o.url || '')
    ]).then(function (assets) {
      var bar = document.querySelector('.post-actions[data-author]');
      o.author = bar ? bar.getAttribute('data-author') : 'arganzheng';
      return renderCard(o, { avatar: assets[1], qr: assets[2] });
    });
  }

  window.BlogPassageCard = { render: render };
})();
