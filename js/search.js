(function(window) {
  var FULLTEXT_CACHE_LIMIT = 120;

  function getParam(name) {
    var params = new URLSearchParams(window.location.search);
    return (params.get(name) || "").trim();
  }

  function escapeHtml(text) {
    return (text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function tokenize(query) {
    return query
      .toLowerCase()
      .split(/[\s,，。！？、;；:：]+/)
      .filter(Boolean);
  }

  function normalize(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }

  function getTagsText(item) {
    if (typeof item.tags_text === "string") return item.tags_text;
    if (Array.isArray(item.tags)) return item.tags.join(" ");
    if (typeof item.tags === "string") return item.tags;
    return "";
  }

  function lower(text) {
    return String(text || "").toLowerCase();
  }

  function containsText(text, lowerNeedle, regexNeedle) {
    if (!text) return false;
    if (lowerNeedle && text.length < 3000) return lower(text).indexOf(lowerNeedle) >= 0;
    return regexNeedle ? regexNeedle.test(text) : false;
  }

  function makeSnippet(text, keyword) {
    var cleanText = normalize(text);
    if (!cleanText) return "";
    if (!keyword) return cleanText.slice(0, 320);

    var lower = cleanText.toLowerCase();
    var lowerKeyword = keyword.toLowerCase();
    var pos = lower.indexOf(lowerKeyword);
    if (pos < 0) return cleanText.slice(0, 320);

    var start = Math.max(0, pos - 90);
    var end = Math.min(cleanText.length, pos + lowerKeyword.length + 220);
    var prefix = start > 0 ? "..." : "";
    var suffix = end < cleanText.length ? "..." : "";
    return prefix + cleanText.slice(start, end) + suffix;
  }

  function highlightText(text, terms) {
    var output = escapeHtml(text || "");
    var sortedTerms = terms.slice().sort(function(a, b) {
      return b.length - a.length;
    });

    sortedTerms.forEach(function(term) {
      if (!term) return;
      var re = new RegExp("(" + escapeRegExp(term) + ")", "ig");
      output = output.replace(re, "<mark>$1</mark>");
    });
    return output;
  }

  function scoreResult(item, contentText, ctx) {
    var title = lower(item.title);
    var tags = lower(getTagsText(item));
    var content = contentText || "";

    var i;
    for (i = 0; i < ctx.terms.length; i += 1) {
      if (
        title.indexOf(ctx.terms[i]) < 0 &&
        tags.indexOf(ctx.terms[i]) < 0 &&
        !containsText(content, ctx.terms[i], ctx.termRegexes[i])
      ) {
        return -1;
      }
    }

    var score = 0;
    if (ctx.query && title.indexOf(ctx.query) >= 0) score += 20;
    if (ctx.query && tags.indexOf(ctx.query) >= 0) score += 10;

    for (i = 0; i < ctx.terms.length; i += 1) {
      if (title.indexOf(ctx.terms[i]) >= 0) score += 6;
      if (tags.indexOf(ctx.terms[i]) >= 0) score += 3;
      if (containsText(content, ctx.terms[i], ctx.termRegexes[i])) score += 1;
    }
    return score;
  }

  function buildResultList(resultMap) {
    return Object.keys(resultMap)
      .map(function(url) { return resultMap[url]; })
      .sort(function(a, b) {
        if (b.score !== a.score) return b.score - a.score;
        return (b.item.date || "").localeCompare(a.item.date || "");
      })
      .slice(0, 50)
      .map(function(entry) { return entry.item; });
  }

  function renderResults(results, statsNode, listNode, rawQuery, terms, extra) {
    if (!rawQuery) {
      statsNode.innerHTML = "输入关键词后开始搜索（支持中文和英文）";
      listNode.innerHTML = "";
      return;
    }

    var stats = "关键词 “" + escapeHtml(rawQuery) + "” ，找到 " + results.length + " 篇文章";
    if (extra && extra.streaming) {
      stats += "（正在增量扫描全文...";
      if (extra.newMatches > 0) stats += " 已补充 " + extra.newMatches + " 篇";
      stats += "）";
    }
    statsNode.innerHTML = stats;

    if (!results.length) {
      listNode.innerHTML = "<p>没有找到结果，试试更短的关键词或同义词。</p>";
      return;
    }

    listNode.innerHTML = results.map(function(item) {
      var snippetKeyword = terms.length ? terms[0] : rawQuery;
      var snippet = makeSnippet(item.content || item.excerpt, snippetKeyword);
      var highlightedSnippet = highlightText(snippet, terms);
      var highlightedTitle = highlightText(item.title || "", terms);
      var tagsText = getTagsText(item);

      return (
        '<article class="search-result-item">' +
          "<h3><a href=\"" + escapeHtml(item.url) + "\">" + highlightedTitle + "</a></h3>" +
          '<div class="search-result-meta">' + escapeHtml(item.date || "") + (tagsText ? " · " + escapeHtml(tagsText) : "") + "</div>" +
          "<p>" + highlightedSnippet + "</p>" +
        "</article>"
      );
    }).join("");
  }

  function setCache(cache, key, value) {
    if (!value) return;
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    if (cache.size <= FULLTEXT_CACHE_LIMIT) return;
    var oldestKey = cache.keys().next().value;
    cache.delete(oldestKey);
  }

  function processNDJSONLine(line, onItem) {
    var trimmed = line.trim();
    if (!trimmed || trimmed[0] !== "{") return;
    try {
      onItem(JSON.parse(trimmed));
    } catch (_) {
      // Ignore malformed lines instead of breaking the whole stream.
    }
  }

  function streamFulltext(url, signal, onItem, onDone) {
    fetch(url + "?t=" + Date.now(), { signal: signal })
      .then(function(resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        if (!resp.body || !resp.body.getReader) return resp.text().then(function(text) {
          text.split("\n").forEach(function(line) { processNDJSONLine(line, onItem); });
        });

        var reader = resp.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";

        function readChunk() {
          return reader.read().then(function(result) {
            if (result.done) {
              if (buffer) processNDJSONLine(buffer, onItem);
              return;
            }
            buffer += decoder.decode(result.value, { stream: true });
            var lines = buffer.split("\n");
            buffer = lines.pop() || "";
            lines.forEach(function(line) { processNDJSONLine(line, onItem); });
            return readChunk();
          });
        }
        return readChunk();
      })
      .then(function() { onDone(null); })
      .catch(function(err) {
        if (err && err.name === "AbortError") return;
        onDone(err);
      });
  }

  function doSearch(state, inputNode, statsNode, listNode) {
    state.searchSeq += 1;
    var thisSearch = state.searchSeq;
    if (state.abortController) state.abortController.abort();

    var rawQuery = (inputNode.value || "").trim();
    var query = lower(rawQuery);
    var terms = tokenize(rawQuery);
    var queryRegex = new RegExp(escapeRegExp(query), "i");
    var termRegexes = terms.map(function(term) {
      return new RegExp(escapeRegExp(term), "i");
    });
    var ctx = {
      query: query,
      terms: terms,
      queryRegex: queryRegex,
      termRegexes: termRegexes
    };

    if (!query) {
      renderResults([], statsNode, listNode, "", []);
      return;
    }

    var resultMap = {};
    state.indexData.forEach(function(item) {
      var score = scoreResult(item, item.excerpt || "", ctx);
      if (score < 0) return;
      resultMap[item.url] = { item: item, score: score };
    });

    var newMatches = 0;
    var renderNow = function(streaming) {
      if (thisSearch !== state.searchSeq) return;
      renderResults(buildResultList(resultMap), statsNode, listNode, rawQuery, terms, {
        streaming: streaming,
        newMatches: newMatches
      });
    };

    renderNow(Boolean(state.contentUrl));
    if (!state.contentUrl) return;

    state.abortController = typeof AbortController !== "undefined" ? new AbortController() : null;
    var signal = state.abortController ? state.abortController.signal : undefined;
    var seen = 0;

    streamFulltext(state.contentUrl, signal, function(row) {
      if (thisSearch !== state.searchSeq) return;
      var meta = state.indexByUrl[row.url];
      if (!meta) return;
      var content = row.content || "";
      if (!content) return;

      seen += 1;
      var score = scoreResult(meta, content, ctx);
      if (score < 0) return;

      setCache(state.fulltextCache, row.url, content);
      var existing = resultMap[row.url];
      if (!existing) {
        newMatches += 1;
        resultMap[row.url] = {
          item: {
            title: meta.title,
            url: meta.url,
            date: meta.date,
            tags: meta.tags,
            tags_text: meta.tags_text,
            excerpt: meta.excerpt,
            content: content
          },
          score: score + 1
        };
      } else {
        // Upgrade snippet quality for index-hit items without keeping all full text.
        if (!existing.item.content) existing.item.content = content;
        if (score > existing.score) existing.score = score;
      }

      if (seen % 12 === 0) renderNow(true);
    }, function(err) {
      if (thisSearch !== state.searchSeq) return;
      if (err) {
        statsNode.innerHTML = "全文扫描失败，已展示快速索引结果。";
        return;
      }
      renderNow(false);
    });
  }

  function init(options) {
    var inputNode = document.getElementById(options.inputId);
    var statsNode = document.getElementById(options.statsId);
    var listNode = document.getElementById(options.resultsId);
    if (!inputNode || !statsNode || !listNode) return;

    var cacheBust = "?t=" + Date.now();
    fetch(options.dataUrl + cacheBust)
      .then(function(resp) { return resp.json(); })
      .then(function(data) {
        var initialQuery = getParam("q");
        if (initialQuery) inputNode.value = initialQuery;

        var indexByUrl = {};
        data.forEach(function(item) {
          indexByUrl[item.url] = item;
        });

        var state = {
          indexData: data,
          indexByUrl: indexByUrl,
          contentUrl: options.contentUrl || "",
          abortController: null,
          searchSeq: 0,
          fulltextCache: new Map()
        };

        var runSearch = function() {
          doSearch(state, inputNode, statsNode, listNode);
        };

        runSearch();
        inputNode.addEventListener("input", runSearch);
      })
      .catch(function() {
        statsNode.innerHTML = "搜索索引加载失败，请稍后重试。";
      });
  }

  window.BLOG_SEARCH = {
    init: init
  };
})(window);
