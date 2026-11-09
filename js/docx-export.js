/*!
 * docx-export.js — author-only DOCX export.
 * window.DocxExport.build({url, title, subtitle, filename}) ->
 *   Promise<{blob, images, formulas, failed}>
 */
(function () {
  'use strict';

  var XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>';
  var W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main';
  var R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
  var REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships';
  var IMAGE_REL = R_NS + '/image';
  var HYPERLINK_REL = R_NS + '/hyperlink';
  var PX_TO_EMU = 9525;
  var PAGE_WIDTH_TWIPS = 11906;
  var PAGE_HEIGHT_TWIPS = 16838;
  var CONTENT_WIDTH_TWIPS = PAGE_WIDTH_TWIPS - 2880;
  var MAX_IMAGE_WIDTH_EMU = Math.round(6.27 * 914400);
  var scriptVersion = (typeof document !== 'undefined' && document.currentScript &&
    (document.currentScript.src.match(/[?&]v=([^&]+)/) || [])[1]) || '';
  var richPromise = null;
  var zipPromise = null;

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
    if (window.RichExport && window.RichExport.articleHtml) return Promise.resolve(window.RichExport);
    if (!richPromise) {
      richPromise = loadScript('/js/rich-export.js').then(function () {
        if (!window.RichExport || !window.RichExport.articleHtml) throw new Error('RichExport unavailable');
        return window.RichExport;
      }).catch(function (error) {
        richPromise = null;
        throw error;
      });
    }
    return richPromise;
  }

  function ensureZipExport() {
    if (window.MdZipExport && window.MdZipExport.zipStore) return Promise.resolve(window.MdZipExport);
    if (!zipPromise) {
      zipPromise = loadScript('/js/md-zip-export.js').then(function () {
        if (!window.MdZipExport || !window.MdZipExport.zipStore) throw new Error('Markdown ZIP helpers unavailable');
        return window.MdZipExport;
      }).catch(function (error) {
        zipPromise = null;
        throw error;
      });
    }
    return zipPromise;
  }

  function xmlEscape(value) {
    return String(value == null ? '' : value)
      .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffe\uffff]/g, '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&apos;');
  }

  function imageExtent(width, height, maxWidth) {
    width = Number(width);
    height = Number(height);
    if (!isFinite(width) || width <= 0) width = 1;
    if (!isFinite(height) || height <= 0) height = 1;
    var cx = Math.max(1, Math.round(width * PX_TO_EMU));
    var cy = Math.max(1, Math.round(height * PX_TO_EMU));
    maxWidth = Number(maxWidth) || MAX_IMAGE_WIDTH_EMU;
    if (!isFinite(maxWidth) || maxWidth <= 0) maxWidth = MAX_IMAGE_WIDTH_EMU;
    if (cx > maxWidth) {
      var scale = maxWidth / cx;
      cx = maxWidth;
      cy = Math.max(1, Math.round(cy * scale));
    }
    return { cx: cx, cy: cy };
  }

  function numberingXml(listInstances) {
    var bullets = ['•', '○', '▪', '▫', '–', '•', '○', '▪', '▫'];
    var bulletLevels = '';
    var decimalLevels = '';
    for (var level = 0; level <= 8; level += 1) {
      var indent = (level + 1) * 720;
      bulletLevels += '<w:lvl w:ilvl="' + level + '"><w:start w:val="1"/><w:numFmt w:val="bullet"/>' +
        '<w:lvlText w:val="' + xmlEscape(bullets[level]) + '"/><w:lvlJc w:val="left"/>' +
        '<w:pPr><w:tabs><w:tab w:val="num" w:pos="' + indent + '"/></w:tabs>' +
        '<w:ind w:left="' + indent + '" w:hanging="360"/></w:pPr></w:lvl>';
      decimalLevels += '<w:lvl w:ilvl="' + level + '"><w:start w:val="1"/><w:numFmt w:val="decimal"/>' +
        '<w:lvlText w:val="%' + (level + 1) + '."/><w:lvlJc w:val="left"/>' +
        '<w:pPr><w:tabs><w:tab w:val="num" w:pos="' + indent + '"/></w:tabs>' +
        '<w:ind w:left="' + indent + '" w:hanging="360"/></w:pPr></w:lvl>';
    }
    var instances = '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>';
    (listInstances || []).forEach(function (item) {
      var overrideLevel = isFinite(item.ilvl) ? Math.min(8, Math.max(0, item.ilvl)) : 0;
      instances += '<w:num w:numId="' + item.numId + '"><w:abstractNumId w:val="1"/>' +
        '<w:lvlOverride w:ilvl="' + overrideLevel + '"><w:startOverride w:val="' + item.start + '"/></w:lvlOverride></w:num>';
    });
    return XML_DECL + '<w:numbering xmlns:w="' + W_NS + '">' +
      '<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="multilevel"/>' + bulletLevels + '</w:abstractNum>' +
      '<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="multilevel"/>' + decimalLevels + '</w:abstractNum>' +
      instances + '</w:numbering>';
  }

  function stylesXml() {
    var headingSizes = [32, 28, 26, 24, 22, 22];
    var headings = '';
    headingSizes.forEach(function (size, index) {
      var level = index;
      headings += '<w:style w:type="paragraph" w:styleId="Heading' + (index + 1) + '">' +
        '<w:name w:val="heading ' + (index + 1) + '"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>' +
        '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="' + (index < 2 ? 240 : 180) +
        '" w:after="100"/><w:outlineLvl w:val="' + level + '"/></w:pPr>' +
        '<w:rPr><w:b/><w:color w:val="' + (index === 0 ? '1F1F1F' : '2F5597') +
        '"/><w:sz w:val="' + size + '"/><w:szCs w:val="' + size + '"/></w:rPr></w:style>';
    });
    return XML_DECL + '<w:styles xmlns:w="' + W_NS + '">' +
      '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" ' +
      'w:eastAsia="Microsoft YaHei" w:cs="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault>' +
      '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>' +
      '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>' +
      '<w:qFormat/></w:style>' +
      '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>' +
      '<w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="0" w:after="240"/></w:pPr>' +
      '<w:rPr><w:b/><w:color w:val="1F1F1F"/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>' +
      '<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/>' +
      '<w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:after="240"/></w:pPr>' +
      '<w:rPr><w:color w:val="666666"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>' +
      headings +
      '<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:basedOn w:val="Normal"/>' +
      '<w:pPr><w:ind w:left="480" w:right="240"/><w:spacing w:before="80" w:after="120"/>' +
      '<w:pBdr><w:left w:val="single" w:sz="14" w:space="8" w:color="B4C7E7"/></w:pBdr></w:pPr>' +
      '<w:rPr><w:color w:val="555555"/></w:rPr></w:style>' +
      '<w:style w:type="paragraph" w:styleId="SourceCode"><w:name w:val="Source Code"/><w:basedOn w:val="Normal"/>' +
      '<w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:ind w:left="240"/>' +
      '<w:shd w:val="clear" w:fill="F3F4F6"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" ' +
      'w:eastAsia="Consolas"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>' +
      '<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/>' +
      '<w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="180"/></w:pPr>' +
      '<w:rPr><w:color w:val="666666"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>' +
      '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:unhideWhenUsed/>' +
      '<w:rPr><w:color w:val="0563C1"/><w:u w:val="single"/></w:rPr></w:style>' +
      '<w:style w:type="character" w:styleId="InlineCode"><w:name w:val="Inline Code"/><w:unhideWhenUsed/>' +
      '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Consolas"/>' +
      '<w:shd w:val="clear" w:fill="F3F4F6"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>' +
      '</w:styles>';
  }

  function settingsXml() {
    return XML_DECL + '<w:settings xmlns:w="' + W_NS + '"><w:zoom w:percent="100"/>' +
      '<w:defaultTabStop w:val="720"/><w:characterSpacingControl w:val="doNotCompress"/></w:settings>';
  }

  function packageRelationshipsXml() {
    return XML_DECL + '<Relationships xmlns="' + REL_NS + '">' +
      '<Relationship Id="rId1" Type="' + R_NS + '/officeDocument" Target="word/document.xml"/>' +
      '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>' +
      '<Relationship Id="rId3" Type="' + R_NS + '/extended-properties" Target="docProps/app.xml"/>' +
      '</Relationships>';
  }

  function documentRelationshipsXml(relationships) {
    return XML_DECL + '<Relationships xmlns="' + REL_NS + '">' + relationships.map(function (relationship) {
      return '<Relationship Id="' + xmlEscape(relationship.id) + '" Type="' + xmlEscape(relationship.type) +
        '" Target="' + xmlEscape(relationship.target) + '"' +
        (relationship.external ? ' TargetMode="External"' : '') + '/>';
    }).join('') + '</Relationships>';
  }

  function contentTypesXml() {
    return XML_DECL + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
      '<Default Extension="xml" ContentType="application/xml"/>' +
      '<Default Extension="png" ContentType="image/png"/>' +
      '<Default Extension="jpeg" ContentType="image/jpeg"/>' +
      '<Default Extension="gif" ContentType="image/gif"/>' +
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>' +
      '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>' +
      '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>' +
      '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>' +
      '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>' +
      '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>' +
      '</Types>';
  }

  function corePropertiesXml(title) {
    var timestamp = new Date().toISOString();
    return XML_DECL + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" ' +
      'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" ' +
      'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>' + xmlEscape(title) +
      '</dc:title><dc:creator>Arganzheng</dc:creator><cp:lastModifiedBy>Arganzheng</cp:lastModifiedBy>' +
      '<dcterms:created xsi:type="dcterms:W3CDTF">' + timestamp + '</dcterms:created>' +
      '<dcterms:modified xsi:type="dcterms:W3CDTF">' + timestamp + '</dcterms:modified></cp:coreProperties>';
  }

  function appPropertiesXml() {
    return XML_DECL + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" ' +
      'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">' +
      '<Application>Arganzheng Blog</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop>' +
      '<LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged>' +
      '<AppVersion>16.0000</AppVersion></Properties>';
  }

  function documentXml(body) {
    return XML_DECL + '<w:document xmlns:w="' + W_NS + '" xmlns:r="' + R_NS +
      '" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" ' +
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" ' +
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>' + body +
      '<w:sectPr><w:pgSz w:w="' + PAGE_WIDTH_TWIPS + '" w:h="' + PAGE_HEIGHT_TWIPS + '"/>' +
      '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>' +
      '<w:cols w:space="720"/></w:sectPr></w:body></w:document>';
  }

  function dataUrlBytes(dataUrl) {
    var match = /^data:([^;,]*)(;base64)?,([\s\S]*)$/i.exec(String(dataUrl || ''));
    if (!match) throw new Error('invalid image data URL');
    var mime = (match[1] || 'text/plain').toLowerCase();
    var payload = match[3] || '';
    if (match[2]) {
      var binary = atob(payload.replace(/\s/g, ''));
      var bytes = new Uint8Array(binary.length);
      for (var i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      return { mime: mime, bytes: bytes };
    }
    var decoded;
    try { decoded = decodeURIComponent(payload); } catch (error) { decoded = payload; }
    return { mime: mime, bytes: new TextEncoder().encode(decoded) };
  }

  function mimeFromPath(url) {
    var pathname = '';
    try { pathname = new URL(url, location.href).pathname; } catch (error) {}
    var match = /\.([^.\/]+)$/.exec(pathname);
    var extension = match ? match[1].toLowerCase() : '';
    if (extension === 'png') return 'image/png';
    if (extension === 'jpg' || extension === 'jpeg') return 'image/jpeg';
    if (extension === 'gif') return 'image/gif';
    if (extension === 'webp') return 'image/webp';
    if (extension === 'svg') return 'image/svg+xml';
    return '';
  }

  function imageType(mime) {
    mime = String(mime || '').split(';')[0].trim().toLowerCase();
    if (mime === 'image/png' || mime === 'image/x-png') return { mime: 'image/png', extension: 'png' };
    if (mime === 'image/jpeg' || mime === 'image/jpg' || mime === 'image/pjpeg') return { mime: 'image/jpeg', extension: 'jpeg' };
    if (mime === 'image/gif') return { mime: 'image/gif', extension: 'gif' };
    return null;
  }

  function imageHint(node, name) {
    var value = node.getAttribute(name);
    if (value && /^\s*\d+(?:\.\d+)?\s*$/.test(value)) return parseFloat(value);
    var style = node.getAttribute('style') || '';
    var match = new RegExp('(?:^|;)\\s*' + name + '\\s*:\\s*(\\d+(?:\\.\\d+)?)px\\s*(?:;|$)', 'i').exec(style);
    return match ? parseFloat(match[1]) : 0;
  }

  function naturalSize(bytes, mime) {
    return new Promise(function (resolve, reject) {
      var objectUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
      var image = new Image();
      image.onload = function () {
        var size = { width: image.naturalWidth || image.width, height: image.naturalHeight || image.height };
        URL.revokeObjectURL(objectUrl);
        if (!size.width || !size.height) reject(new Error('image dimensions unavailable'));
        else resolve(size);
      };
      image.onerror = function () {
        URL.revokeObjectURL(objectUrl);
        reject(new Error('image decode failed'));
      };
      image.src = objectUrl;
    });
  }

  function imageDimensions(node, bytes, mime) {
    var width = imageHint(node, 'width');
    var height = imageHint(node, 'height');
    if (width > 0 && height > 0) return Promise.resolve({ width: width, height: height });
    return naturalSize(bytes, mime).then(function (natural) {
      if (width > 0) height = width * natural.height / natural.width;
      else if (height > 0) width = height * natural.width / natural.height;
      else { width = natural.width; height = natural.height; }
      return { width: width, height: height };
    });
  }

  function directChildren(node) {
    return Array.prototype.slice.call(node.childNodes || []);
  }

  function tagName(node) {
    return node && node.nodeType === 1 ? node.tagName.toLowerCase() : '';
  }

  function isBlockTag(tag) {
    return /^(address|article|aside|blockquote|dd|div|dl|dt|figure|figcaption|footer|h[1-6]|header|hr|li|main|ol|p|pre|section|table|ul)$/.test(tag);
  }

  function textRun(text, context) {
    text = String(text == null ? '' : text).replace(/\r\n?/g, '\n').replace(/\n/g, ' ');
    if (!text) return '';
    var props = [];
    if (context && context.characterStyle) props.push('<w:rStyle w:val="' + context.characterStyle + '"/>');
    if (context && context.bold) props.push('<w:b/>');
    if (context && context.italic) props.push('<w:i/>');
    if (context && context.strike) props.push('<w:strike/>');
    if (context && context.vertAlign) props.push('<w:vertAlign w:val="' + context.vertAlign + '"/>');
    return '<w:r>' + (props.length ? '<w:rPr>' + props.join('') + '</w:rPr>' : '') +
      '<w:t xml:space="preserve">' + xmlEscape(text) + '</w:t></w:r>';
  }

  function paragraphXml(runs, style, node, context, alignment) {
    context = context || {};
    style = context.quote && style === 'Normal' ? 'Quote' : (style || 'Normal');
    var props = ['<w:pStyle w:val="' + xmlEscape(style) + '"/>'];
    if (context.numId) {
      props.push('<w:numPr><w:ilvl w:val="' + Math.min(8, context.ilvl || 0) +
        '"/><w:numId w:val="' + context.numId + '"/></w:numPr>');
    }
    if (context.extraIndent) props.push('<w:ind w:left="' + context.extraIndent + '"/>');
    var align = alignment;
    if (!align && node && node.nodeType === 1) {
      var inlineStyle = node.getAttribute('style') || '';
      var match = /(?:^|;)\s*text-align\s*:\s*(left|center|right|justify)\s*(?:;|$)/i.exec(inlineStyle);
      align = match && match[1].toLowerCase();
    }
    if (align) props.push('<w:jc w:val="' + xmlEscape(align) + '"/>');
    return '<w:p><w:pPr>' + props.join('') + '</w:pPr>' + runs + '</w:p>';
  }

  function resolveUrl(url, baseUrl) {
    try { return new URL(url, baseUrl).href; } catch (error) { return String(url || ''); }
  }

  function stylesForContext(context, override) {
    var next = {};
    Object.keys(context || {}).forEach(function (key) { next[key] = context[key]; });
    Object.keys(override || {}).forEach(function (key) { next[key] = override[key]; });
    return next;
  }

  function imageMarkup(media, extent, alt, drawingId) {
    var name = 'Picture ' + drawingId;
    var escapedAlt = xmlEscape(alt);
    return '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">' +
      '<wp:extent cx="' + extent.cx + '" cy="' + extent.cy + '"/>' +
      '<wp:docPr id="' + drawingId + '" name="' + name + '" descr="' + escapedAlt + '"/>' +
      '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>' +
      '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">' +
      '<pic:pic><pic:nvPicPr><pic:cNvPr id="' + drawingId + '" name="' + name + '" descr="' + escapedAlt + '"/>' +
      '<pic:cNvPicPr><a:picLocks noChangeAspect="1"/></pic:cNvPicPr></pic:nvPicPr>' +
      '<pic:blipFill><a:blip r:embed="' + media.relationshipId + '"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>' +
      '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="' + extent.cx + '" cy="' + extent.cy + '"/></a:xfrm>' +
      '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:ln><a:noFill/></a:ln></pic:spPr>' +
      '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>';
  }

  function imageBytes(source, baseUrl, richExport) {
    if (/^data:/i.test(source)) {
      var parsed = dataUrlBytes(source);
      var type = imageType(parsed.mime);
      if (type) return Promise.resolve({ bytes: parsed.bytes, mime: type.mime, extension: type.extension });
      if (parsed.mime === 'image/svg+xml' && richExport.svgUrlToPng) {
        return richExport.svgUrlToPng(source).then(dataUrlBytes).then(function (result) {
          return { bytes: result.bytes, mime: 'image/png', extension: 'png' };
        });
      }
      if (parsed.mime === 'image/webp' && window.WechatExport && window.WechatExport.imgToJpeg) {
        return window.WechatExport.imgToJpeg(source).then(dataUrlBytes).then(function (result) {
          return { bytes: result.bytes, mime: 'image/jpeg', extension: 'jpeg' };
        });
      }
      return Promise.reject(new Error('unsupported image type ' + parsed.mime));
    }
    var url = resolveUrl(source, baseUrl);
    return fetch(url).then(function (response) {
      if (!response.ok) throw new Error('image HTTP ' + response.status);
      var headerMime = response.headers.get('content-type') || '';
      var mime = (headerMime || mimeFromPath(url) || '').split(';')[0].toLowerCase();
      if (!headerMime || mime === 'application/octet-stream') mime = mimeFromPath(url) || mime;
      var type = imageType(mime);
      if (type) return response.arrayBuffer().then(function (buffer) {
        return { bytes: new Uint8Array(buffer), mime: type.mime, extension: type.extension };
      });
      if ((mime === 'image/svg+xml' || mimeFromPath(url) === 'image/svg+xml') && richExport.svgUrlToPng) {
        return richExport.svgUrlToPng(url).then(dataUrlBytes).then(function (result) {
          return { bytes: result.bytes, mime: 'image/png', extension: 'png' };
        });
      }
      if (mime === 'image/webp' || mimeFromPath(url) === 'image/webp') {
        var wechat = window.WechatExport;
        if (!wechat || !wechat.imgToJpeg) throw new Error('WebP conversion unavailable');
        return wechat.imgToJpeg(url).then(dataUrlBytes).then(function (result) {
          return { bytes: result.bytes, mime: 'image/jpeg', extension: 'jpeg' };
        });
      }
      throw new Error('unsupported image type ' + mime);
    });
  }

  function positiveTableSpan(value) {
    var span = parseInt(value, 10);
    return isFinite(span) && span > 0 ? span : 1;
  }

  function tableGrid(sourceRows) {
    sourceRows = Array.isArray(sourceRows) ? sourceRows : [];
    var carries = [];
    var rows = [];
    var columnCount = 0;

    function nextCarryColumn(start) {
      for (var column = start; column < carries.length; column += 1) {
        if (carries[column] && carries[column].remaining > 0) return column;
      }
      return -1;
    }

    function overlapsCarry(start, end) {
      for (var column = 0; column < carries.length; column += 1) {
        var carry = carries[column];
        if (carry && carry.remaining > 0 && column < end && column + carry.span > start) return true;
      }
      return false;
    }

    for (var rowIndex = 0; rowIndex < sourceRows.length; rowIndex += 1) {
      var sourceRow = Array.isArray(sourceRows[rowIndex]) ? sourceRows[rowIndex] : [];
      var sourceIndex = 0;
      var column = 0;
      var cells = [];

      while (sourceIndex < sourceRow.length || nextCarryColumn(column) >= 0) {
        var carry = carries[column];
        if (carry && carry.remaining > 0) {
          cells.push({ kind: 'continuation', column: column, span: carry.span });
          carry.remaining -= 1;
          if (!carry.remaining) carries[column] = null;
          column += carry.span;
          continue;
        }

        if (sourceIndex < sourceRow.length) {
          var source = sourceRow[sourceIndex] || {};
          var span = positiveTableSpan(source.colspan);
          if (overlapsCarry(column, column + span)) {
            cells.push({ kind: 'padding', column: column, span: 1 });
            column += 1;
            continue;
          }

          var rowspan = Math.min(positiveTableSpan(source.rowspan), sourceRows.length - rowIndex);
          cells.push({
            kind: 'source',
            column: column,
            span: span,
            rowspan: rowspan,
            header: !!source.header,
            sourceIndex: sourceIndex
          });
          if (rowspan > 1) carries[column] = { remaining: rowspan - 1, span: span };
          sourceIndex += 1;
          column += span;
          continue;
        }

        if (nextCarryColumn(column) >= 0) {
          cells.push({ kind: 'padding', column: column, span: 1 });
          column += 1;
        }
      }

      columnCount = Math.max(columnCount, column);
      rows.push({ cells: cells });
    }

    if (sourceRows.length && !columnCount) columnCount = 1;
    rows.forEach(function (row) {
      var byColumn = Object.create(null);
      row.cells.forEach(function (cell) { byColumn[cell.column] = cell; });
      var cells = [];
      for (var column = 0; column < columnCount;) {
        var cell = byColumn[column];
        if (cell) {
          cells.push(cell);
          column += cell.span;
        } else {
          cells.push({ kind: 'padding', column: column, span: 1 });
          column += 1;
        }
      }
      row.cells = cells;
    });

    return { columnCount: columnCount, rows: rows };
  }

  function tableRowsXml(grid, cellContents, gridWidth) {
    gridWidth = Number(gridWidth);
    if (!isFinite(gridWidth) || gridWidth < 1) gridWidth = 1;
    return grid.rows.map(function (row, rowIndex) {
      var cells = row.cells.map(function (cell) {
        var width = gridWidth * cell.span;
        var properties = '<w:tcW w:w="' + width + '" w:type="dxa"/>';
        if (cell.span > 1) properties += '<w:gridSpan w:val="' + cell.span + '"/>';
        if (cell.kind === 'source' && cell.rowspan > 1) properties += '<w:vMerge w:val="restart"/>';
        else if (cell.kind === 'continuation') properties += '<w:vMerge/>';
        if (cell.kind === 'source' && cell.header) properties += '<w:shd w:val="clear" w:fill="E7E6E6"/>';
        properties += '<w:vAlign w:val="center"/>';

        var content = cell.kind === 'source' && cellContents &&
          cellContents[rowIndex] && cellContents[rowIndex][cell.sourceIndex] || '<w:p/>';
        if (!/(?:<\/w:p>|<w:p\/>)\s*$/.test(content)) content += '<w:p/>';
        return '<w:tc><w:tcPr>' + properties + '</w:tcPr>' + content + '</w:tc>';
      }).join('');
      return '<w:tr>' + cells + '</w:tr>';
    }).join('');
  }

  async function buildDocument(richResult, options, zipApi) {
    var baseUrl = options.url || location.href;
    var parsed = new DOMParser().parseFromString(richResult.html, 'text/html');
    var wrapper = parsed.body.firstElementChild || parsed.body;
    var state = {
      relationships: [
        { id: 'rId1', type: R_NS + '/styles', target: 'styles.xml' },
        { id: 'rId2', type: R_NS + '/numbering', target: 'numbering.xml' },
        { id: 'rId3', type: R_NS + '/settings', target: 'settings.xml' }
      ],
      relationshipMap: new Map(),
      nextRelationshipId: 4,
      mediaBySource: new Map(),
      media: [],
      listInstances: [],
      nextNumId: 2,
      nextDrawingId: 1,
      failed: 0,
      extraImages: 0,
      richExport: window.RichExport,
      baseUrl: baseUrl
    };

    state.relationships.forEach(function (relationship) {
      state.relationshipMap.set(relationship.type + '\u0000' + relationship.target, relationship.id);
    });

    state.addRelationship = function (type, target, external) {
      var key = type + '\u0000' + target + '\u0000' + (external ? 'external' : 'internal');
      if (state.relationshipMap.has(key)) return state.relationshipMap.get(key);
      var id = 'rId' + state.nextRelationshipId++;
      state.relationships.push({ id: id, type: type, target: target, external: !!external });
      state.relationshipMap.set(key, id);
      return id;
    };

    async function addMedia(source) {
      var url = /^data:/i.test(source) ? source : resolveUrl(source, baseUrl);
      if (state.mediaBySource.has(url)) return state.mediaBySource.get(url);
      var promise = imageBytes(source, baseUrl, state.richExport).then(function (image) {
        var mediaName = 'image' + (state.media.length + 1) + '.' + image.extension;
        var relationshipId = state.addRelationship(IMAGE_REL, 'media/' + mediaName, false);
        var media = {
          name: mediaName,
          data: image.bytes,
          mime: image.mime,
          extension: image.extension,
          relationshipId: relationshipId
        };
        state.media.push(media);
        return media;
      });
      state.mediaBySource.set(url, promise);
      try { return await promise; }
      catch (error) {
        state.mediaBySource.delete(url);
        throw error;
      }
    }

    async function renderImage(node) {
      var source = node.getAttribute('src') || '';
      var alt = node.getAttribute('alt') || '';
      if (!source) {
        state.failed += 1;
        return textRun('[图片：' + alt + ']', {});
      }
      try {
        var media = await addMedia(source);
        var dimensions = await imageDimensions(node, media.data, media.mime);
        if (!/^data:/i.test(source)) state.extraImages += 1;
        var extent = imageExtent(dimensions.width, dimensions.height);
        var drawingId = state.nextDrawingId++;
        return imageMarkup(media, extent, alt, drawingId);
      } catch (error) {
        state.failed += 1;
        var absolute = resolveUrl(source, baseUrl);
        var label = '[图片：' + (alt || '图片') + ']';
        if (!absolute || /^data:/i.test(absolute)) return textRun(label, {});
        var relId = state.addRelationship(HYPERLINK_REL, absolute, true);
        return '<w:hyperlink r:id="' + relId + '">' + textRun(label, { characterStyle: 'Hyperlink' }) + '</w:hyperlink>';
      }
    }

    async function renderInline(node, context) {
      if (!node) return '';
      if (node.nodeType === 3) return textRun(node.nodeValue, context);
      if (node.nodeType !== 1) return '';
      var tag = tagName(node);
      if (tag === 'br') return '<w:r><w:br/></w:r>';
      if (tag === 'img') return renderImage(node);
      if (tag === 'a') {
        var href = node.getAttribute('href') || '';
        if (!href) return renderInlineChildren(node, context);
        var absolute = resolveUrl(href, baseUrl);
        if (/^(javascript|data):/i.test(absolute)) return renderInlineChildren(node, context);
        var relId = state.addRelationship(HYPERLINK_REL, absolute, true);
        var hyperlinkRuns = await renderInlineChildren(node, stylesForContext(context, { characterStyle: 'Hyperlink' }));
        return '<w:hyperlink r:id="' + relId + '">' + hyperlinkRuns + '</w:hyperlink>';
      }
      if (tag === 'strong' || tag === 'b') return renderInlineChildren(node, stylesForContext(context, { bold: true }));
      if (tag === 'em' || tag === 'i') return renderInlineChildren(node, stylesForContext(context, { italic: true }));
      if (tag === 'del' || tag === 's' || tag === 'strike') return renderInlineChildren(node, stylesForContext(context, { strike: true }));
      if (tag === 'sup') return renderInlineChildren(node, stylesForContext(context, { vertAlign: 'superscript' }));
      if (tag === 'sub') return renderInlineChildren(node, stylesForContext(context, { vertAlign: 'subscript' }));
      if (tag === 'code') return renderInlineChildren(node, stylesForContext(context, { characterStyle: 'InlineCode' }));
      return renderInlineChildren(node, context);
    }

    async function renderInlineChildren(node, context) {
      var children = directChildren(node);
      var output = '';
      for (var i = 0; i < children.length; i += 1) output += await renderInline(children[i], context || {});
      return output;
    }

    async function renderParagraph(children, node, style, context, alignment) {
      var runs = '';
      for (var i = 0; i < children.length; i += 1) runs += await renderInline(children[i], context || {});
      if (!alignment && context && context.centerImages) alignment = 'center';
      return paragraphXml(runs, style, node, context, alignment);
    }

    function emptyNumberedParagraph(listInfo) {
      return paragraphXml('', 'Normal', null, { numId: listInfo.numId, ilvl: listInfo.ilvl });
    }

    async function renderList(listNode, depth, context) {
      var tag = tagName(listNode);
      var ilvl = Math.min(8, depth);
      var listInfo;
      if (tag === 'ol') {
        var start = parseInt(listNode.getAttribute('start'), 10);
        if (!isFinite(start)) start = 1;
        listInfo = { numId: state.nextNumId++, ilvl: ilvl };
        state.listInstances.push({ numId: listInfo.numId, start: start, ilvl: ilvl });
      } else listInfo = { numId: 1, ilvl: ilvl };
      var items = Array.prototype.filter.call(listNode.children, function (child) { return tagName(child) === 'li'; });
      var output = '';

      for (var itemIndex = 0; itemIndex < items.length; itemIndex += 1) {
        var item = items[itemIndex];
        var itemOutput = '';
        var hasNumberedParagraph = false;
        var pending = [];
        var children = directChildren(item);

        async function flushPending() {
          if (!pending.length) return;
          var hasContent = pending.some(function (node) {
            return node.nodeType === 1 || (node.nodeType === 3 && /\S/.test(node.nodeValue));
          });
          if (!hasContent) { pending = []; return; }
          var paragraphContext = hasNumberedParagraph ?
            stylesForContext(context, { extraIndent: (depth + 1) * 720 }) :
            stylesForContext(context, { numId: listInfo.numId, ilvl: listInfo.ilvl });
          itemOutput += await renderParagraph(pending, item, 'Normal', paragraphContext);
          hasNumberedParagraph = true;
          pending = [];
        }

        for (var childIndex = 0; childIndex < children.length; childIndex += 1) {
          var child = children[childIndex];
          var childTag = tagName(child);
          if (child.nodeType === 3) {
            if (/\S/.test(child.nodeValue)) pending.push(child);
            continue;
          }
          if (childTag === 'ul' || childTag === 'ol') {
            await flushPending();
            if (!hasNumberedParagraph) {
              itemOutput += emptyNumberedParagraph(listInfo);
              hasNumberedParagraph = true;
            }
            itemOutput += await renderList(child, depth + 1, context);
          } else if (childTag === 'p' && !hasNumberedParagraph && !pending.length) {
            itemOutput += await renderParagraph(directChildren(child), child, 'Normal',
              stylesForContext(context, { numId: listInfo.numId, ilvl: listInfo.ilvl }));
            hasNumberedParagraph = true;
          } else if (isBlockTag(childTag)) {
            await flushPending();
            if (!hasNumberedParagraph) {
              if (childTag === 'div') {
                var directInline = directChildren(child).filter(function (node) {
                  if (node.nodeType === 3) return /\S/.test(node.nodeValue);
                  return !isBlockTag(tagName(node));
                });
                if (directInline.length) {
                  itemOutput += await renderParagraph(directInline, child, 'Normal',
                    stylesForContext(context, { numId: listInfo.numId, ilvl: listInfo.ilvl }));
                  hasNumberedParagraph = true;
                  var blockChildren = directChildren(child).filter(function (node) {
                    return node.nodeType === 1 && isBlockTag(tagName(node));
                  });
                  for (var blockIndex = 0; blockIndex < blockChildren.length; blockIndex += 1) {
                    itemOutput += await renderBlock(blockChildren[blockIndex],
                      stylesForContext(context, { extraIndent: (depth + 1) * 720 }));
                  }
                  continue;
                }
              }
              if (!hasNumberedParagraph) {
                itemOutput += emptyNumberedParagraph(listInfo);
                hasNumberedParagraph = true;
              }
            }
            itemOutput += await renderBlock(child, stylesForContext(context, { extraIndent: (depth + 1) * 720 }));
          } else pending.push(child);
        }
        await flushPending();
        if (!hasNumberedParagraph) itemOutput += emptyNumberedParagraph(listInfo);
        output += itemOutput;
      }
      return output;
    }

    async function renderPre(node, context) {
      var lines = String(node.textContent || '').replace(/\r\n?/g, '\n').split('\n');
      return lines.map(function (line) {
        var pieces = line.split('\t');
        var runs = '';
        pieces.forEach(function (piece, index) {
          if (piece) runs += textRun(piece, {});
          if (index < pieces.length - 1) runs += '<w:r><w:tab/></w:r>';
        });
        return paragraphXml(runs, 'SourceCode', null, context || {});
      }).join('');
    }

    async function renderTable(table, context) {
      var rows = [];
      directChildren(table).forEach(function (child) {
        var tag = tagName(child);
        if (tag === 'tr') rows.push(child);
        else if (tag === 'thead' || tag === 'tbody' || tag === 'tfoot') {
          directChildren(child).forEach(function (row) { if (tagName(row) === 'tr') rows.push(row); });
        }
      });
      if (!rows.length) return '';
      var rowCells = rows.map(function (row) {
        return Array.prototype.filter.call(row.children, function (cell) { return /^(td|th)$/.test(tagName(cell)); });
      });
      var grid = tableGrid(rowCells.map(function (cells) {
        return cells.map(function (cell) {
          return {
            colspan: cell.getAttribute('colspan'),
            rowspan: cell.getAttribute('rowspan'),
            header: tagName(cell) === 'th'
          };
        });
      }));
      var columnCount = Math.max(1, grid.columnCount);
      var gridWidth = Math.floor(CONTENT_WIDTH_TWIPS / columnCount);
      var gridXml = '';
      for (var column = 0; column < columnCount; column += 1) gridXml += '<w:gridCol w:w="' + gridWidth + '"/>';
      var tableIndent = context && context.extraIndent ?
        '<w:tblInd w:w="' + context.extraIndent + '" w:type="dxa"/>' : '';
      var tableProperties = '<w:tblPr><w:tblW w:w="5000" w:type="pct"/>' + tableIndent + '<w:tblLayout w:type="fixed"/>' +
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="B7B7B7"/>' +
        '<w:left w:val="single" w:sz="4" w:color="B7B7B7"/><w:bottom w:val="single" w:sz="4" w:color="B7B7B7"/>' +
        '<w:right w:val="single" w:sz="4" w:color="B7B7B7"/><w:insideH w:val="single" w:sz="4" w:color="B7B7B7"/>' +
        '<w:insideV w:val="single" w:sz="4" w:color="B7B7B7"/></w:tblBorders>' +
        '<w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>' +
        '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>';
      var cellContents = [];
      for (var rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
        var contents = [];
        for (var cellIndex = 0; cellIndex < rowCells[rowIndex].length; cellIndex += 1) {
          var cell = rowCells[rowIndex][cellIndex];
          var isHeader = tagName(cell) === 'th';
          contents.push(await renderBlocks(cell, stylesForContext(context, { bold: isHeader, quote: false })));
        }
        cellContents.push(contents);
      }
      return '<w:tbl>' + tableProperties + '<w:tblGrid>' + gridXml + '</w:tblGrid>' +
        tableRowsXml(grid, cellContents, gridWidth) + '</w:tbl>';
    }

    async function renderFigure(node, context) {
      var output = '';
      var figureContext = stylesForContext(context, { centerImages: true });
      var children = directChildren(node);
      for (var i = 0; i < children.length; i += 1) {
        var child = children[i];
        var tag = tagName(child);
        if (tag === 'img') output += await renderParagraph([child], node, 'Normal', figureContext, 'center');
        else if (tag === 'figcaption') output += await renderParagraph(directChildren(child), child, 'Caption', context, 'center');
        else if (tag) output += await renderBlock(child, figureContext);
      }
      return output;
    }

    async function renderBlock(node, context) {
      var tag = tagName(node);
      if (!tag) return '';
      if (tag === 'p') return renderParagraph(directChildren(node), node, 'Normal', context);
      if (/^h[1-6]$/.test(tag)) return renderParagraph(directChildren(node), node, 'Heading' + tag.charAt(1), context);
      if (tag === 'blockquote') return renderBlocks(node, stylesForContext(context, { quote: true }));
      if (tag === 'ul' || tag === 'ol') return renderList(node, context && context.listDepth || 0, context);
      if (tag === 'pre') return renderPre(node, context);
      if (tag === 'table') return renderTable(node, context || {});
      if (tag === 'figure') return renderFigure(node, context || {});
      if (tag === 'figcaption') return renderParagraph(directChildren(node), node, 'Caption', context, 'center');
      if (tag === 'hr') {
        var indent = context && context.extraIndent ? '<w:ind w:left="' + context.extraIndent + '"/>' : '';
        return '<w:p><w:pPr><w:pStyle w:val="Normal"/>' + indent + '<w:pBdr><w:bottom w:val="single" ' +
          'w:sz="6" w:space="1" w:color="B7B7B7"/></w:pBdr></w:pPr></w:p>';
      }
      if (tag === 'img') return renderParagraph([node], node, 'Normal', context);
      if (tag === 'li') return renderBlocks(node, context);
      return renderBlocks(node, context);
    }

    async function renderBlocks(parent, context) {
      var output = '';
      var pending = [];
      var children = directChildren(parent);
      async function flushInline() {
        if (!pending.length) return;
        var hasContent = pending.some(function (node) {
          return node.nodeType === 1 || (node.nodeType === 3 && /\S/.test(node.nodeValue));
        });
        if (hasContent) output += await renderParagraph(pending, null, 'Normal', context);
        pending = [];
      }
      for (var i = 0; i < children.length; i += 1) {
        var child = children[i];
        var tag = tagName(child);
        if (child.nodeType === 3) {
          if (/\S/.test(child.nodeValue)) pending.push(child);
        } else if (isBlockTag(tag)) {
          await flushInline();
          output += await renderBlock(child, context);
        } else pending.push(child);
      }
      await flushInline();
      return output;
    }

    var wrapperChildren = directChildren(wrapper);
    var titleNode = null;
    var subtitleNode = null;
    for (var topIndex = 0; topIndex < wrapperChildren.length; topIndex += 1) {
      var topTag = tagName(wrapperChildren[topIndex]);
      if (!titleNode && topTag === 'h1') { titleNode = wrapperChildren[topIndex]; continue; }
      if (titleNode && options.subtitle && !subtitleNode && topTag === 'blockquote') {
        subtitleNode = wrapperChildren[topIndex];
        continue;
      }
      if (wrapperChildren[topIndex].nodeType === 3 && !/\S/.test(wrapperChildren[topIndex].nodeValue)) continue;
    }
    var title = String(options.title || (titleNode && titleNode.textContent) || '');
    var subtitle = String(options.subtitle || (subtitleNode && subtitleNode.textContent) || '');
    var body = '';
    if (title) body += paragraphXml(textRun(title, { bold: true }), 'Title', null, {});
    if (subtitle) body += paragraphXml(textRun(subtitle, {}), 'Subtitle', null, {});
    for (var bodyIndex = 0; bodyIndex < wrapperChildren.length; bodyIndex += 1) {
      var childNode = wrapperChildren[bodyIndex];
      if (childNode === titleNode || childNode === subtitleNode) continue;
      if (childNode.nodeType === 3 && !/\S/.test(childNode.nodeValue)) continue;
      body += await renderBlock(childNode, {});
    }

    var relationshipXml = documentRelationshipsXml(state.relationships);
    var files = [
      { name: '[Content_Types].xml', data: new TextEncoder().encode(contentTypesXml()) },
      { name: '_rels/.rels', data: new TextEncoder().encode(packageRelationshipsXml()) },
      { name: 'docProps/core.xml', data: new TextEncoder().encode(corePropertiesXml(title)) },
      { name: 'docProps/app.xml', data: new TextEncoder().encode(appPropertiesXml()) },
      { name: 'word/document.xml', data: new TextEncoder().encode(documentXml(body)) },
      { name: 'word/styles.xml', data: new TextEncoder().encode(stylesXml()) },
      { name: 'word/numbering.xml', data: new TextEncoder().encode(numberingXml(state.listInstances)) },
      { name: 'word/settings.xml', data: new TextEncoder().encode(settingsXml()) },
      { name: 'word/_rels/document.xml.rels', data: new TextEncoder().encode(relationshipXml) }
    ];
    state.media.forEach(function (media) {
      files.push({ name: 'word/media/' + media.name, data: media.data });
    });
    var filename = String(options.filename || 'article.md').split(/[\\/]/).pop();
    var basename = filename.replace(/\.md$/i, '') || 'article';
    return {
      blob: new Blob([zipApi.zipStore(files)], {
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
      }),
      name: basename + '.docx',
      images: Number(richResult.images || 0) + state.extraImages,
      formulas: Number(richResult.formulas || 0),
      failed: state.failed
    };
  }

  function build(options) {
    options = options || {};
    var richPromise = ensureRichExport().then(function (rich) {
      return rich.articleHtml({
        url: options.url || location.href,
        title: options.title || '',
        subtitle: options.subtitle || ''
      });
    });
    return Promise.all([richPromise, ensureZipExport()]).then(function (values) {
      return buildDocument(values[0], options, values[1]);
    });
  }

  var api = { build: build };
  if (typeof window !== 'undefined') window.DocxExport = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = Object.assign({}, api, {
      xmlEscape: xmlEscape,
      imageExtent: imageExtent,
      numberingXml: numberingXml,
      tableGrid: tableGrid,
      tableRowsXml: tableRowsXml,
      maxImageWidthEmu: MAX_IMAGE_WIDTH_EMU
    });
  }
})();
