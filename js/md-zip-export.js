/*!
 * md-zip-export.js — author-only Markdown ZIP export.
 * window.MdZipExport.build({source, site, title, subtitle, filename}) ->
 *   Promise<{blob, name, images, failed}>
 */
(function () {
  'use strict';

  var scriptVersion = (typeof document !== 'undefined' && document.currentScript &&
    (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';
  var richPromise = null;
  var wechatPromise = null;
  var crcTable = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement('script');
      script.src = src + (scriptVersion ? '?v=' + scriptVersion : '');
      script.onload = resolve;
      script.onerror = function () { reject(new Error('failed to load ' + src)); };
      document.head.appendChild(script);
    });
  }

  function ensureRichExport() {
    if (window.RichExport && window.RichExport.svgUrlToPng) return Promise.resolve(window.RichExport);
    if (!richPromise) {
      richPromise = loadScript('/js/rich-export.js').then(function () {
        if (!window.RichExport || !window.RichExport.svgUrlToPng) throw new Error('RichExport helpers unavailable');
        return window.RichExport;
      }).catch(function (error) {
        richPromise = null;
        throw error;
      });
    }
    return richPromise;
  }

  function ensureWechatExport() {
    if (window.WechatExport && window.WechatExport.imgToJpeg) return Promise.resolve(window.WechatExport);
    if (!wechatPromise) {
      wechatPromise = loadScript('/js/wechat-export.js').then(function () {
        if (!window.WechatExport || !window.WechatExport.imgToJpeg) throw new Error('WechatExport helpers unavailable');
        return window.WechatExport;
      }).catch(function (error) {
        wechatPromise = null;
        throw error;
      });
    }
    return wechatPromise;
  }

  function zipImageName(url, used) {
    used = used || new Map();
    url = String(url);
    var urlKey = '\u0000url:' + url;
    if (used.has(urlKey)) return used.get(urlKey);

    var path = url.split(/[?#]/, 1)[0];
    var name = path.slice(path.lastIndexOf('/') + 1) || 'image';
    try { name = decodeURIComponent(name); } catch (e) {}
    name = name.replace(/[\/\\\u0000-\u001f]/g, '_');
    if (!name || name === '.' || name === '..') name = 'image';

    var extension = /\.([^.]+)$/.exec(name);
    var suffix = extension ? '.' + extension[1].toLowerCase() : '';
    var stem = extension ? name.slice(0, -extension[0].length) : name;
    if (/\.svg$/i.test(name)) suffix = '.png';
    else if (/\.webp$/i.test(name)) suffix = '.jpg';
    if (/\.svg$/i.test(name) || /\.webp$/i.test(name)) stem = name.slice(0, name.lastIndexOf('.'));

    var candidate = stem + suffix;
    var number = 2;
    while (used.has(candidate) && used.get(candidate) !== url) {
      candidate = stem + '-' + number + suffix;
      number += 1;
    }
    used.set(candidate, url);
    used.set(urlKey, candidate);
    return candidate;
  }

  function encodeImageName(name) {
    return encodeURIComponent(name).replace(/[!'()*]/g, function (character) {
      return '%' + character.charCodeAt(0).toString(16).toUpperCase();
    });
  }

  function absoluteUrl(site, url) {
    return String(site || '').replace(/\/+$/, '') + url;
  }

  function imageHref(url) {
    return new URL(url, location.href).href;
  }

  function imageExtension(url) {
    var pathname = new URL(url, location.href).pathname;
    var match = /\.([^.\/]+)$/.exec(pathname);
    return match ? match[1].toLowerCase() : '';
  }

  function dataUrlToBytes(dataUrl) {
    return fetch(dataUrl).then(function (response) { return response.arrayBuffer(); })
      .then(function (buffer) { return new Uint8Array(buffer); });
  }

  function convertImage(entry) {
    var href = imageHref(entry.url);
    var extension = imageExtension(entry.url);
    if (extension === 'svg') {
      return ensureRichExport().then(function (rich) {
        return rich.svgUrlToPng(href);
      }).then(dataUrlToBytes);
    }
    if (extension === 'webp') {
      return ensureWechatExport().then(function (wechat) {
        return wechat.imgToJpeg(href);
      }).then(dataUrlToBytes);
    }
    return fetch(href).then(function (response) {
      if (!response.ok) throw new Error('image HTTP ' + response.status);
      return response.arrayBuffer();
    }).then(function (buffer) { return new Uint8Array(buffer); });
  }

  function crc32(bytes) {
    if (!crcTable) {
      crcTable = new Uint32Array(256);
      for (var tableIndex = 0; tableIndex < 256; tableIndex += 1) {
        var value = tableIndex;
        for (var bit = 0; bit < 8; bit += 1) value = value & 1 ? (value >>> 1) ^ 0xedb88320 : value >>> 1;
        crcTable[tableIndex] = value >>> 0;
      }
    }
    var crc = 0xffffffff;
    for (var i = 0; i < bytes.length; i += 1) crc = (crc >>> 8) ^ crcTable[(crc ^ bytes[i]) & 0xff];
    return (crc ^ 0xffffffff) >>> 0;
  }

  function zipStore(files) {
    if (!Array.isArray(files) || files.length > 65535) throw new Error('invalid ZIP file list');
    var now = new Date();
    var year = Math.max(1980, Math.min(2107, now.getFullYear()));
    var dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | Math.floor(now.getSeconds() / 2);
    var dosDate = ((year - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    var localRecords = [];
    var centralRecords = [];
    var localSize = 0;
    var centralSize = 0;

    files.forEach(function (file) {
      var name = String(file.name || '').replace(/\\/g, '/');
      if (!name || name.charAt(0) === '/' || name.split('/').some(function (part) { return !part || part === '..'; })) {
        throw new Error('unsafe ZIP entry name');
      }
      var nameBytes = new TextEncoder().encode(name);
      var data = file.data instanceof Uint8Array ? file.data : new Uint8Array(file.data || []);
      if (nameBytes.length > 65535 || data.length > 0xffffffff) throw new Error('ZIP entry is too large');
      var crc = crc32(data);
      var local = new Uint8Array(30 + nameBytes.length + data.length);
      var localView = new DataView(local.buffer);
      localView.setUint32(0, 0x04034b50, true);
      localView.setUint16(4, 20, true);
      localView.setUint16(6, 0x0800, true);
      localView.setUint16(8, 0, true);
      localView.setUint16(10, dosTime, true);
      localView.setUint16(12, dosDate, true);
      localView.setUint32(14, crc, true);
      localView.setUint32(18, data.length, true);
      localView.setUint32(22, data.length, true);
      localView.setUint16(26, nameBytes.length, true);
      localView.setUint16(28, 0, true);
      local.set(nameBytes, 30);
      local.set(data, 30 + nameBytes.length);
      localRecords.push(local);

      var central = new Uint8Array(46 + nameBytes.length);
      var centralView = new DataView(central.buffer);
      centralView.setUint32(0, 0x02014b50, true);
      centralView.setUint16(4, 20, true);
      centralView.setUint16(6, 20, true);
      centralView.setUint16(8, 0x0800, true);
      centralView.setUint16(10, 0, true);
      centralView.setUint16(12, dosTime, true);
      centralView.setUint16(14, dosDate, true);
      centralView.setUint32(16, crc, true);
      centralView.setUint32(20, data.length, true);
      centralView.setUint32(24, data.length, true);
      centralView.setUint16(28, nameBytes.length, true);
      centralView.setUint16(30, 0, true);
      centralView.setUint16(32, 0, true);
      centralView.setUint16(34, 0, true);
      centralView.setUint16(36, 0, true);
      centralView.setUint32(38, 0, true);
      centralView.setUint32(42, localSize, true);
      central.set(nameBytes, 46);
      centralRecords.push(central);

      localSize += local.length;
      centralSize += central.length;
    });
    if (localSize > 0xffffffff || centralSize > 0xffffffff) throw new Error('ZIP archive is too large');

    var end = new Uint8Array(22);
    var endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(4, 0, true);
    endView.setUint16(6, 0, true);
    endView.setUint16(8, files.length, true);
    endView.setUint16(10, files.length, true);
    endView.setUint32(12, centralSize, true);
    endView.setUint32(16, localSize, true);
    endView.setUint16(20, 0, true);

    var output = new Uint8Array(localSize + centralSize + end.length);
    var offset = 0;
    localRecords.forEach(function (record) { output.set(record, offset); offset += record.length; });
    centralRecords.forEach(function (record) { output.set(record, offset); offset += record.length; });
    output.set(end, offset);
    return output;
  }

  function build(options) {
    options = options || {};
    if (!window.MarkdownExport || !window.MarkdownExport.buildPayload ||
        !window.MarkdownExport.rewriteMarkdownUrls) {
      return Promise.reject(new Error('Markdown payload helpers unavailable'));
    }
    var source = String(options.source == null ? '' : options.source);
    var site = String(options.site || location.origin).replace(/\/+$/, '');
    var entries = new Map();
    var usedNames = new Map();
    window.MarkdownExport.buildPayload(source, site, options.title, options.subtitle, function (body) {
      return window.MarkdownExport.rewriteMarkdownUrls(body, function (url, kind) {
        if (kind === 'image' && !entries.has(url)) {
          entries.set(url, { url: url, name: zipImageName(url, usedNames) });
        }
        return url;
      });
    });

    var failed = 0;
    var jobs = Array.from(entries.values()).map(function (entry) {
      return convertImage(entry).then(function (data) {
        entry.data = data;
      }, function (error) {
        entry.error = error;
        failed += 1;
      });
    });
    return Promise.all(jobs).then(function () {
      var transformed = window.MarkdownExport.buildPayload(
        source, site, options.title, options.subtitle,
        function (body) {
          return window.MarkdownExport.rewriteMarkdownUrls(body, function (url, kind) {
            if (kind === 'image') {
              var entry = entries.get(url);
              if (entry && entry.data) return 'images/' + encodeImageName(entry.name);
            }
            return absoluteUrl(site, url);
          });
        }
      );
      var filename = String(options.filename || 'article.md').split(/[\\/]/).pop();
      var basename = filename.replace(/\.md$/i, '') || 'article';
      var files = [{ name: basename + '.md', data: new TextEncoder().encode(transformed.text) }];
      var images = 0;
      entries.forEach(function (entry) {
        if (!entry.data) return;
        files.push({ name: 'images/' + entry.name, data: entry.data });
        images += 1;
      });
      return {
        blob: new Blob([zipStore(files)], { type: 'application/zip' }),
        name: basename + '.zip',
        images: images,
        failed: failed
      };
    });
  }

  var api = { build: build, zipImageName: zipImageName, zipStore: zipStore };
  if (typeof window !== 'undefined') window.MdZipExport = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();
