(function(window) {
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

  // English/number terms match whole words only, so "AI" does not hit
  // "WAIT" or "RAID". CJK terms keep substring matching.
  function isWordTerm(term) {
    return /^[a-z0-9]+$/i.test(term);
  }

  // Build per-query matchers once (not per item) to keep searching fast.
  function buildQueryContext(rawQuery) {
    var query = rawQuery.toLowerCase();
    var terms = tokenize(rawQuery);
    return {
      query: query,
      terms: terms,
      // Boundary-aware matchers against pre-lowered haystacks.
      matchers: terms.map(function(term) {
        return isWordTerm(term) ? new RegExp("\\b" + escapeRegExp(term) + "\\b") : null;
      }),
      // Case-insensitive matchers for locating snippets in original text.
      snippetMatchers: terms.map(function(term) {
        return isWordTerm(term)
          ? new RegExp("\\b" + escapeRegExp(term) + "\\b", "i")
          : new RegExp(escapeRegExp(term), "i");
      }),
      // Global matchers for highlighting rendered text.
      highlightMatchers: terms
        .slice()
        .sort(function(a, b) { return b.length - a.length; })
        .map(function(term) {
          return isWordTerm(term)
            ? new RegExp("\\b(" + escapeRegExp(term) + ")\\b", "ig")
            : new RegExp("(" + escapeRegExp(term) + ")", "ig");
        }),
      queryMatcher: isWordTerm(query)
        ? new RegExp("\\b" + escapeRegExp(query) + "\\b")
        : null
    };
  }

  function fieldHits(field, ctx, i) {
    var matcher = ctx.matchers[i];
    return matcher ? matcher.test(field) : field.indexOf(ctx.terms[i]) >= 0;
  }

  function makeSnippet(text, ctx) {
    var cleanText = normalize(text);
    if (!cleanText) return "";

    var pos = -1;
    var matchLength = 0;
    for (var i = 0; i < ctx.snippetMatchers.length && pos < 0; i += 1) {
      var m = ctx.snippetMatchers[i].exec(cleanText);
      if (m) {
        pos = m.index;
        matchLength = m[0].length;
      }
    }
    if (pos < 0) return cleanText.slice(0, 320);

    var start = Math.max(0, pos - 90);
    var end = Math.min(cleanText.length, pos + matchLength + 220);
    var prefix = start > 0 ? "..." : "";
    var suffix = end < cleanText.length ? "..." : "";
    return prefix + cleanText.slice(start, end) + suffix;
  }

  function highlightText(text, ctx) {
    var output = escapeHtml(text || "");
    ctx.highlightMatchers.forEach(function(re) {
      output = output.replace(re, "<mark>$1</mark>");
    });
    return output;
  }

  function scoreResult(item, ctx) {
    // item._search is the pre-lowered "title content" haystack built once.
    var title = item._title;
    var haystack = item._search;

    var i;
    for (i = 0; i < ctx.terms.length; i += 1) {
      if (!fieldHits(haystack, ctx, i)) return -1;
    }

    var score = 0;
    if (ctx.query) {
      var queryHit = ctx.queryMatcher
        ? ctx.queryMatcher.test(title)
        : title.indexOf(ctx.query) >= 0;
      if (queryHit) score += 20;
    }

    for (i = 0; i < ctx.terms.length; i += 1) {
      if (fieldHits(title, ctx, i)) score += 6;
      if (fieldHits(haystack, ctx, i)) score += 1;
    }
    return score;
  }

  function renderResults(results, statsNode, listNode, rawQuery, ctx, loadingFulltext) {
    if (!rawQuery) {
      statsNode.innerHTML = "输入关键词后开始搜索（支持中文和英文）";
      listNode.innerHTML = "";
      return;
    }

    var stats = "关键词 “" + escapeHtml(rawQuery) + "” ，找到 " + results.length + " 篇文章";
    if (loadingFulltext) stats += "（全文索引加载中，结果可能继续增加）";
    statsNode.innerHTML = stats;

    if (!results.length) {
      listNode.innerHTML = "<p>没有找到结果，试试更短的关键词或同义词。</p>";
      return;
    }

    listNode.innerHTML = results.map(function(item) {
      var snippet = makeSnippet(item.content || item.excerpt, ctx);
      var highlightedSnippet = highlightText(snippet, ctx);
      var highlightedTitle = highlightText(item.title || "", ctx);
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

  // Pre-compute the lowered haystack once per item so per-keystroke search
  // never re-lowercases megabytes of text (the root cause of earlier jank).
  // Only title and content are searchable; tags are display-only.
  function buildSearchFields(item) {
    item._title = (item.title || "").toLowerCase();
    item._search = item._title + " " +
      (item.content ? item.content.toLowerCase() : (item.excerpt || "").toLowerCase());
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

  // Load the fulltext NDJSON exactly once at page load. Contents are merged
  // into the in-memory index; afterwards every search is pure in-memory.
  function loadFulltextOnce(url, indexByUrl, onProgress, onDone) {
    fetch(url + "?t=" + Date.now())
      .then(function(resp) {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        return resp.text();
      })
      .then(function(text) {
        var lines = text.split("\n");
        var i = 0;

        // Merge in chunks on idle time so the main thread stays responsive.
        function mergeChunk() {
          var deadline = Date.now() + 12;
          while (i < lines.length && Date.now() < deadline) {
            processNDJSONLine(lines[i], function(row) {
              var meta = indexByUrl[row.url];
              if (meta && row.content) {
                meta.content = row.content;
                buildSearchFields(meta);
              }
            });
            i += 1;
          }
          if (i < lines.length) {
            setTimeout(mergeChunk, 0);
          } else {
            onDone(null);
          }
          onProgress();
        }
        mergeChunk();
      })
      .catch(function(err) { onDone(err); });
  }

  function doSearch(state, inputNode, statsNode, listNode) {
    var rawQuery = (inputNode.value || "").trim();

    if (!rawQuery) {
      renderResults([], statsNode, listNode, "", null, false);
      return;
    }

    var ctx = buildQueryContext(rawQuery);
    var results = state.indexData
      .map(function(item) {
        return { item: item, score: scoreResult(item, ctx) };
      })
      .filter(function(result) { return result.score >= 0; })
      .sort(function(a, b) {
        if (b.score !== a.score) return b.score - a.score;
        return (b.item.date || "").localeCompare(a.item.date || "");
      })
      .map(function(result) { return result.item; });

    renderResults(results, statsNode, listNode, rawQuery, ctx, !state.fulltextReady);
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
          buildSearchFields(item);
          indexByUrl[item.url] = item;
        });

        var state = {
          indexData: data,
          fulltextReady: !options.contentUrl
        };

        var debounceTimer = null;
        var runSearch = function() {
          doSearch(state, inputNode, statsNode, listNode);
        };
        var runSearchDebounced = function() {
          if (debounceTimer) clearTimeout(debounceTimer);
          debounceTimer = setTimeout(runSearch, 120);
        };

        runSearch();
        inputNode.addEventListener("input", runSearchDebounced);

        if (options.contentUrl) {
          loadFulltextOnce(options.contentUrl, indexByUrl, function() {}, function(err) {
            state.fulltextReady = true;
            if (err) return;
            // Refresh current query with fulltext-aware results.
            if ((inputNode.value || "").trim()) runSearch();
          });
        }
      })
      .catch(function() {
        statsNode.innerHTML = "搜索索引加载失败，请稍后重试。";
      });
  }

  window.BLOG_SEARCH = {
    init: init
  };
})(window);
