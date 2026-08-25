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

  function scoreResult(item, query, terms) {
    var title = (item.title || "").toLowerCase();
    var tags = getTagsText(item).toLowerCase();
    var content = (item.content || "").toLowerCase();
    var full = title + " " + tags + " " + content;

    var i;
    for (i = 0; i < terms.length; i += 1) {
      if (full.indexOf(terms[i]) < 0) return -1;
    }

    var score = 0;
    if (query && title.indexOf(query) >= 0) score += 20;
    if (query && tags.indexOf(query) >= 0) score += 10;

    for (i = 0; i < terms.length; i += 1) {
      if (title.indexOf(terms[i]) >= 0) score += 6;
      if (tags.indexOf(terms[i]) >= 0) score += 3;
      if (content.indexOf(terms[i]) >= 0) score += 1;
    }
    return score;
  }

  function renderResults(results, statsNode, listNode, rawQuery, terms) {
    if (!rawQuery) {
      statsNode.innerHTML = "输入关键词后开始搜索（支持中文和英文）";
      listNode.innerHTML = "";
      return;
    }

    statsNode.innerHTML = "关键词 “" + escapeHtml(rawQuery) + "” ，找到 " + results.length + " 篇文章";

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

  function doSearch(data, inputNode, statsNode, listNode) {
    var rawQuery = (inputNode.value || "").trim();
    var query = rawQuery.toLowerCase();
    var terms = tokenize(rawQuery);

    if (!query) {
      renderResults([], statsNode, listNode, "", []);
      return;
    }

    var results = data
      .map(function(item) {
        return {
          item: item,
          score: scoreResult(item, query, terms)
        };
      })
      .filter(function(result) { return result.score >= 0; })
      .sort(function(a, b) {
        if (b.score !== a.score) return b.score - a.score;
        return (b.item.date || "").localeCompare(a.item.date || "");
      })
      .slice(0, 50)
      .map(function(result) { return result.item; });

    renderResults(results, statsNode, listNode, rawQuery, terms);
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

        var runSearch = function() {
          doSearch(data, inputNode, statsNode, listNode);
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
