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

  return {
    isAllowedBlogImageUrl: isAllowedBlogImageUrl,
    isHttpUrl: isHttpUrl,
    splitMarkdown: splitMarkdown
  };
});
