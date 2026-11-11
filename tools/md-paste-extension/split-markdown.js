(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MdPasteMarkdown = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function isAllowedBlogImageUrl(value) {
    try {
      var url = new URL(value);
      return (url.protocol === 'https:' || url.protocol === 'http:') &&
        url.hostname.toLowerCase() === 'arganzheng.life';
    } catch (_) {
      return false;
    }
  }

  function isHttpUrl(value) {
    try {
      var url = new URL(value);
      return url.protocol === 'https:' || url.protocol === 'http:';
    } catch (_) {
      return false;
    }
  }

  function isEscaped(text, index) {
    var backslashes = 0;
    for (var cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
      backslashes += 1;
    }
    return backslashes % 2 === 1;
  }

  function fenceRanges(text) {
    var ranges = [];
    var offset = 0;
    var open = null;

    while (offset < text.length) {
      var newline = text.indexOf('\n', offset);
      var end = newline < 0 ? text.length : newline + 1;
      var line = text.slice(offset, end).replace(/\r?\n$/, '');
      var marker = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);

      if (!open && marker) {
        var delimiter = marker[1];
        if (delimiter[0] !== '`' || marker[2].indexOf('`') < 0) {
          open = {
            start: offset,
            char: delimiter[0],
            length: delimiter.length
          };
        }
      } else if (open) {
        var closing = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(line);
        if (closing && closing[1][0] === open.char && closing[1].length >= open.length) {
          ranges.push([open.start, end]);
          open = null;
        }
      }

      offset = end;
    }

    if (open) ranges.push([open.start, text.length]);
    return ranges;
  }

  function inlineCodeRanges(text, fences) {
    var ranges = [];
    var fenceIndex = 0;
    var index = 0;

    while (index < text.length) {
      while (fenceIndex < fences.length && fences[fenceIndex][1] <= index) fenceIndex += 1;
      if (fenceIndex < fences.length && index >= fences[fenceIndex][0]) {
        index = fences[fenceIndex][1];
        continue;
      }
      if (text[index] !== '`' || isEscaped(text, index)) {
        index += 1;
        continue;
      }

      var runEnd = index;
      while (text[runEnd] === '`') runEnd += 1;
      var runLength = runEnd - index;
      var search = runEnd;
      var close = -1;

      while (search < text.length) {
        while (fenceIndex < fences.length && fences[fenceIndex][1] <= search) fenceIndex += 1;
        if (fenceIndex < fences.length && search >= fences[fenceIndex][0]) {
          search = fences[fenceIndex][1];
          continue;
        }
        if (text[search] !== '`' || isEscaped(text, search)) {
          search += 1;
          continue;
        }
        var candidateEnd = search;
        while (text[candidateEnd] === '`') candidateEnd += 1;
        if (candidateEnd - search === runLength) {
          close = candidateEnd;
          break;
        }
        search = candidateEnd;
      }

      if (close >= 0) {
        ranges.push([index, close]);
        index = close;
      } else {
        index = runEnd;
      }
    }

    return ranges;
  }

  function protectedRanges(text) {
    var fences = fenceRanges(text);
    var ranges = fences.concat(inlineCodeRanges(text, fences));
    ranges.sort(function (left, right) { return left[0] - right[0]; });
    var merged = [];

    ranges.forEach(function (range) {
      var last = merged[merged.length - 1];
      if (last && range[0] <= last[1]) last[1] = Math.max(last[1], range[1]);
      else merged.push(range.slice());
    });
    return merged;
  }

  function tagEnd(text, start) {
    var quote = '';
    for (var index = start; index < text.length; index += 1) {
      var character = text[index];
      if (quote) {
        if (character === quote) quote = '';
      } else if (character === '"' || character === "'") {
        quote = character;
      } else if (character === '>') {
        return index + 1;
      }
    }
    return -1;
  }

  function decodeHtmlAttribute(value) {
    return value
      .replace(/&amp;/gi, '&')
      .replace(/&quot;/gi, '"')
      .replace(/&#39;|&apos;/gi, "'")
      .replace(/&lt;/gi, '<')
      .replace(/&gt;/gi, '>');
  }

  function parseHtmlImage(text, start, isAllowed) {
    if (text.slice(start, start + 4).toLowerCase() !== '<img' || !/[\s/>]/.test(text[start + 4] || '')) {
      return null;
    }
    var end = tagEnd(text, start + 4);
    if (end < 0) return null;
    var tag = text.slice(start, end);
    var source = /\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(tag);
    if (!source) return null;
    var url = decodeHtmlAttribute(source[1] || source[2] || source[3] || '');
    if (!url || !isAllowed(url)) return null;
    var altMatch = /\balt\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i.exec(tag);
    var alt = altMatch ? (altMatch[1] || altMatch[2] || altMatch[3] || '') : '';
    return {
      end: end,
      token: { type: 'image', url: url, alt: decodeHtmlAttribute(alt), raw: text.slice(start, end) }
    };
  }

  function findBracketEnd(text, start) {
    var depth = 1;
    for (var index = start; index < text.length; index += 1) {
      if (isEscaped(text, index)) continue;
      if (text[index] === '[') depth += 1;
      else if (text[index] === ']') {
        depth -= 1;
        if (!depth) return index;
      }
    }
    return -1;
  }

  function parseMarkdownImage(text, start, isAllowed) {
    if (text.slice(start, start + 2) !== '![' || isEscaped(text, start)) return null;
    var bracketEnd = findBracketEnd(text, start + 2);
    if (bracketEnd < 0 || text[bracketEnd + 1] !== '(') return null;

    var index = bracketEnd + 2;
    while (/\s/.test(text[index] || '') && index < text.length) index += 1;
    var url = '';

    if (text[index] === '<') {
      var destinationEnd = index + 1;
      while (destinationEnd < text.length && (text[destinationEnd] !== '>' || isEscaped(text, destinationEnd))) {
        destinationEnd += 1;
      }
      if (text[destinationEnd] !== '>') return null;
      url = text.slice(index + 1, destinationEnd).replace(/\\([\\<>])/g, '$1');
      index = destinationEnd + 1;
    } else {
      var parentheses = 0;
      while (index < text.length) {
        var character = text[index];
        if (character === '\\' && index + 1 < text.length) {
          url += text[index + 1];
          index += 2;
          continue;
        }
        if (character === '(') parentheses += 1;
        else if (character === ')') {
          if (!parentheses) break;
          parentheses -= 1;
        } else if (/\s/.test(character) && !parentheses) {
          break;
        }
        url += character;
        index += 1;
      }
    }

    if (!url) return null;
    while (/\s/.test(text[index] || '') && index < text.length) index += 1;
    if (text[index] !== ')') {
      var titleQuote = text[index];
      if (titleQuote !== '"' && titleQuote !== "'" && titleQuote !== '(') return null;
      var titleEndChar = titleQuote === '(' ? ')' : titleQuote;
      index += 1;
      while (index < text.length) {
        if (text[index] === '\\' && index + 1 < text.length) {
          index += 2;
        } else if (text[index] === titleEndChar) {
          index += 1;
          break;
        } else {
          index += 1;
        }
      }
      while (/\s/.test(text[index] || '') && index < text.length) index += 1;
    }
    if (text[index] !== ')') return null;
    index += 1;

    if (!isAllowed(url)) return null;
    return {
      end: index,
      token: {
        type: 'image',
        url: url,
        alt: text.slice(start + 2, bracketEnd).replace(/\\([\\`*{}\[\]()#+.!_>~-])/g, '$1'),
        raw: text.slice(start, index)
      }
    };
  }

  function splitMarkdown(value, isAllowed) {
    var text = String(value == null ? '' : value);
    var allow = typeof isAllowed === 'function' ? isAllowed : isAllowedBlogImageUrl;
    var ranges = protectedRanges(text);
    var rangeIndex = 0;
    var tokens = [];
    var lastTextEnd = 0;
    var index = 0;

    while (index < text.length) {
      while (rangeIndex < ranges.length && ranges[rangeIndex][1] <= index) rangeIndex += 1;
      if (rangeIndex < ranges.length && index >= ranges[rangeIndex][0]) {
        index = ranges[rangeIndex][1];
        continue;
      }

      var parsed = text[index] === '<' ?
        parseHtmlImage(text, index, allow) : parseMarkdownImage(text, index, allow);
      if (parsed) {
        if (index > lastTextEnd) tokens.push({ type: 'text', text: text.slice(lastTextEnd, index) });
        tokens.push(parsed.token);
        index = parsed.end;
        lastTextEnd = index;
      } else {
        index += 1;
      }
    }

    if (lastTextEnd < text.length) tokens.push({ type: 'text', text: text.slice(lastTextEnd) });
    return tokens;
  }

  function relativeBlogUrl(value) {
    if (!isAllowedBlogImageUrl(value)) return null;
    var url = new URL(value);
    return (url.pathname || '/') + url.search + url.hash;
  }

  function overlapsProtected(ranges, start, end) {
    for (var index = 0; index < ranges.length; index += 1) {
      if (ranges[index][0] >= end) return false;
      if (ranges[index][1] > start) return true;
    }
    return false;
  }

  function parseLinkDestination(text, openParen) {
    var index = openParen + 1;
    while (/\s/.test(text[index] || '') && index < text.length) index += 1;

    var start;
    var end;
    var url;
    if (text[index] === '<') {
      start = index + 1;
      end = start;
      while (end < text.length && (text[end] !== '>' || isEscaped(text, end))) end += 1;
      if (text[end] !== '>') return null;
      url = text.slice(start, end);
      index = end + 1;
    } else {
      start = index;
      var parentheses = 0;
      while (index < text.length) {
        var character = text[index];
        if (character === '\\' && index + 1 < text.length) {
          index += 2;
          continue;
        }
        if (character === '(') parentheses += 1;
        else if (character === ')') {
          if (!parentheses) break;
          parentheses -= 1;
        } else if (/\s/.test(character) && !parentheses) {
          break;
        }
        index += 1;
      }
      end = index;
      url = text.slice(start, end);
      if (!url) return null;
    }

    while (/\s/.test(text[index] || '') && index < text.length) index += 1;
    if (text[index] !== ')') {
      var titleOpen = text[index];
      var titleClose = titleOpen === '(' ? ')' : titleOpen;
      if (titleOpen !== '"' && titleOpen !== "'" && titleOpen !== '(') return null;
      index += 1;
      while (index < text.length && (text[index] !== titleClose || isEscaped(text, index))) index += 1;
      if (text[index] !== titleClose) return null;
      index += 1;
      while (/\s/.test(text[index] || '') && index < text.length) index += 1;
    }
    if (text[index] !== ')') return null;
    return { start: start, end: end, url: url };
  }

  function addInlineLinkEdits(text, ranges, edits) {
    var rangeIndex = 0;
    var index = 0;
    while (index < text.length) {
      while (rangeIndex < ranges.length && ranges[rangeIndex][1] <= index) rangeIndex += 1;
      if (rangeIndex < ranges.length && index >= ranges[rangeIndex][0]) {
        index = ranges[rangeIndex][1];
        continue;
      }
      if (text[index] !== '[' || isEscaped(text, index) ||
          (index > 0 && text[index - 1] === '!' && !isEscaped(text, index - 1))) {
        index += 1;
        continue;
      }
      var labelEnd = findBracketEnd(text, index + 1);
      if (labelEnd < 0 || text[labelEnd + 1] !== '(') {
        index += 1;
        continue;
      }
      var destination = parseLinkDestination(text, labelEnd + 1);
      if (!destination || overlapsProtected(ranges, destination.start, destination.end)) {
        index += 1;
        continue;
      }
      var replacement = relativeBlogUrl(destination.url);
      if (replacement !== null) {
        edits.push({
          start: destination.start,
          end: destination.end,
          replacement: replacement
        });
      }
      index += 1;
    }
  }

  function addReferenceDefinitionEdits(text, ranges, edits) {
    var lineStart = 0;
    while (lineStart < text.length) {
      var newline = text.indexOf('\n', lineStart);
      var lineEnd = newline < 0 ? text.length : newline;
      var line = text.slice(lineStart, lineEnd).replace(/\r$/, '');
      var match = /^ {0,3}\[[^\]\r\n]+\]:[ \t]*(?:<([^>\r\n]*)>|(\S+))(?:[ \t]+(?:"[^"\r\n]*"|'[^'\r\n]*'|\([^)\r\n]*\)))?[ \t]*$/.exec(line);
      if (match) {
        var destination = match[1] === undefined ? match[2] : match[1];
        var replacement = relativeBlogUrl(destination);
        if (replacement !== null) {
          var colon = line.indexOf(']:') + 1;
          var destinationStart = match[1] === undefined ?
            line.indexOf(destination, colon + 1) :
            line.indexOf('<', colon + 1) + 1;
          var start = lineStart + destinationStart;
          var end = start + destination.length;
          if (!overlapsProtected(ranges, start, end)) {
            edits.push({ start: start, end: end, replacement: replacement });
          }
        }
      }
      if (newline < 0) break;
      lineStart = newline + 1;
    }
  }

  function addHtmlHrefEdits(text, ranges, edits) {
    var rangeIndex = 0;
    var index = 0;
    while (index < text.length) {
      while (rangeIndex < ranges.length && ranges[rangeIndex][1] <= index) rangeIndex += 1;
      if (rangeIndex < ranges.length && index >= ranges[rangeIndex][0]) {
        index = ranges[rangeIndex][1];
        continue;
      }
      if (text[index] !== '<') {
        index += 1;
        continue;
      }
      var end = tagEnd(text, index + 1);
      if (end < 0) {
        index += 1;
        continue;
      }
      var tag = text.slice(index, end);
      if (!/^<\s*[A-Za-z][A-Za-z0-9:.-]*(?:\s|\/|>)/.test(tag)) {
        index = end;
        continue;
      }

      var cursor = 1;
      while (/\s/.test(tag[cursor] || '') && cursor < tag.length) cursor += 1;
      while (cursor < tag.length && !/[\s/>]/.test(tag[cursor])) cursor += 1;
      while (cursor < tag.length - 1) {
        while (/[\s/]/.test(tag[cursor] || '') && cursor < tag.length - 1) cursor += 1;
        var nameStart = cursor;
        while (cursor < tag.length && !/[\s=/>]/.test(tag[cursor])) cursor += 1;
        if (cursor === nameStart) {
          cursor += 1;
          continue;
        }
        var name = tag.slice(nameStart, cursor).toLowerCase();
        while (/\s/.test(tag[cursor] || '') && cursor < tag.length) cursor += 1;
        if (tag[cursor] !== '=') continue;
        cursor += 1;
        while (/\s/.test(tag[cursor] || '') && cursor < tag.length) cursor += 1;
        var quote = tag[cursor];
        if (quote === '"' || quote === "'") {
          var valueStart = cursor + 1;
          var valueEnd = tag.indexOf(quote, valueStart);
          if (valueEnd < 0) break;
          if (name === 'href') {
            var value = tag.slice(valueStart, valueEnd);
            var relative = relativeBlogUrl(value);
            var absoluteStart = index + valueStart;
            var absoluteEnd = index + valueEnd;
            if (relative !== null && !overlapsProtected(ranges, absoluteStart, absoluteEnd)) {
              edits.push({ start: absoluteStart, end: absoluteEnd, replacement: relative });
            }
          }
          cursor = valueEnd + 1;
        } else {
          while (cursor < tag.length && !/[\s>]/.test(tag[cursor])) cursor += 1;
        }
      }
      index = end;
    }
  }

  function relativizeBlogLinks(value) {
    var text = String(value == null ? '' : value);
    var ranges = protectedRanges(text);
    var edits = [];
    addInlineLinkEdits(text, ranges, edits);
    addReferenceDefinitionEdits(text, ranges, edits);
    addHtmlHrefEdits(text, ranges, edits);
    edits.sort(function (left, right) {
      return left.start - right.start || left.end - right.end;
    });

    var output = '';
    var last = 0;
    var count = 0;
    edits.forEach(function (edit) {
      if (edit.start < last) return;
      output += text.slice(last, edit.start) + edit.replacement;
      last = edit.end;
      count += 1;
    });
    return { text: output + text.slice(last), count: count };
  }

  return {
    isAllowedBlogImageUrl: isAllowedBlogImageUrl,
    isHttpUrl: isHttpUrl,
    splitMarkdown: splitMarkdown,
    relativizeBlogLinks: relativizeBlogLinks
  };
});
