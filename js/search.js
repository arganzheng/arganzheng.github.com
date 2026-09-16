// Site search client. The index is built by _plugins/search_index.rb:
//   <base>/meta.json      one [url, title, date, tags] per post (id = index)
//   <base>/idx/<n>.json   inverted index bucket n: key -> posting list (id deltas)
//   <base>/doc/<slug>.txt plain text of one post
// A query fetches only the buckets its keys hash into (a few KB each) and the
// text of the posts it is about to show; nothing is downloaded up front except
// meta.json. Matching semantics are those of the old all-in-memory version:
// ASCII words match whole words, CJK terms match as substrings, every term must
// hit, title hits rank first, ties go to the newer post.
(function(window) {
  var BUCKETS = 256;
  var PAGE = 10;
  var CJK_RUN = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]+/g;

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

  // English/number terms match whole words only, so "AI" does not hit
  // "WAIT" or "RAID". CJK terms keep substring matching.
  function isWordTerm(term) {
    return /^[a-z0-9]+$/i.test(term);
  }

  // Index keys a term needs: a list of alternatives-groups, ANDed across
  // groups and ORed within one. `exact` is false when the keys only
  // approximate the term (bigram chain of 3+ CJK chars, ASCII with
  // punctuation), in which case the real text is checked before a post is shown.
  function keysOf(term) {
    if (isWordTerm(term)) return { keys: [[term]], exact: true };
    // Substring semantics: `state` must also hit inside `load_state_dict`.
    var keys = (term.match(/[a-z0-9]+/g) || []).map(function(w) { return [w, "~" + w]; });
    var runs = term.match(CJK_RUN) || [];
    runs.forEach(function(run) {
      if (run.length === 1) keys.push([run]);
      for (var i = 0; i + 1 < run.length; i += 1) keys.push([run.slice(i, i + 2)]);
    });
    // Exact only for a lone CJK character or bigram (a key of its own).
    return { keys: keys, exact: runs.length === 1 && keys.length === 1 && term === runs[0] };
  }

  // Same arithmetic as SearchIndex.bucket_of in _plugins/search_index.rb.
  function bucketOf(key) {
    var h = 0;
    for (var i = 0; i < key.length; i += 1) h = (h * 31 + key.charCodeAt(i)) >>> 0;
    return h % BUCKETS;
  }

  function buildQueryContext(rawQuery) {
    var query = rawQuery.toLowerCase();
    var terms = tokenize(rawQuery);
    var plans = terms.map(keysOf);
    return {
      query: query,
      terms: terms,
      plans: plans,
      exact: plans.every(function(p) { return p.exact; }),
      matchers: terms.map(function(term) {
        return isWordTerm(term) ? new RegExp("\\b" + escapeRegExp(term) + "\\b") : null;
      }),
      snippetMatchers: terms.map(function(term) {
        return isWordTerm(term)
          ? new RegExp("\\b" + escapeRegExp(term) + "\\b", "i")
          : new RegExp(escapeRegExp(term), "i");
      }),
      highlightMatchers: terms
        .slice()
        .sort(function(a, b) { return b.length - a.length; })
        .map(function(term) {
          return isWordTerm(term)
            ? new RegExp("\\b(" + escapeRegExp(term) + ")\\b", "ig")
            : new RegExp("(" + escapeRegExp(term) + ")", "ig");
        }),
      queryMatcher: isWordTerm(query) ? new RegExp("\\b" + escapeRegExp(query) + "\\b") : null
    };
  }

  function fieldHits(field, ctx, i) {
    var matcher = ctx.matchers[i];
    return matcher ? matcher.test(field) : field.indexOf(ctx.terms[i]) >= 0;
  }

  function makeSnippet(text, ctx) {
    var cleanText = (text || "").replace(/\s+/g, " ").trim();
    if (!cleanText) return "";
    var pos = -1, matchLength = 0;
    for (var i = 0; i < ctx.snippetMatchers.length && pos < 0; i += 1) {
      var m = ctx.snippetMatchers[i].exec(cleanText);
      if (m) { pos = m.index; matchLength = m[0].length; }
    }
    if (pos < 0) return cleanText.slice(0, 320);
    var start = Math.max(0, pos - 90);
    var end = Math.min(cleanText.length, pos + matchLength + 220);
    return (start > 0 ? "..." : "") + cleanText.slice(start, end) + (end < cleanText.length ? "..." : "");
  }

  function highlightText(text, ctx) {
    var output = escapeHtml(text || "");
    ctx.highlightMatchers.forEach(function(re) {
      output = output.replace(re, "<mark>$1</mark>");
    });
    return output;
  }

  // Title-only scoring: every candidate already hits the body via the index,
  // so the old "+1 per term in body" is a constant and was dropped.
  function scoreTitle(item, ctx) {
    var title = item._title, score = 0;
    if (ctx.queryMatcher ? ctx.queryMatcher.test(title) : title.indexOf(ctx.query) >= 0) score += 20;
    for (var i = 0; i < ctx.terms.length; i += 1) if (fieldHits(title, ctx, i)) score += 6;
    return score;
  }

  function fullMatch(item, text, ctx) {
    var haystack = item._title + " " + text.toLowerCase();
    for (var i = 0; i < ctx.terms.length; i += 1) if (!fieldHits(haystack, ctx, i)) return false;
    return true;
  }

  function Index(options) {
    this.base = options.baseUrl.replace(/\/+$/, "");
    this.bust = options.version ? "?v=" + options.version : "";
    this.buckets = {};
    this.docs = {};
  }

  Index.prototype.fetchText = function(url) {
    return fetch(url + this.bust).then(function(resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status + " " + url);
      return resp.text();
    });
  };

  Index.prototype.load = function() {
    var self = this;
    return this.fetchText(this.base + "/meta.json").then(function(text) {
      self.items = JSON.parse(text).map(function(row, id) {
        return { id: id, url: row[0], title: row[1], date: row[2], tags: row[3] || [], _title: (row[1] || "").toLowerCase() };
      });
      return self.items;
    });
  };

  Index.prototype.bucket = function(n) {
    if (!this.buckets[n]) {
      this.buckets[n] = this.fetchText(this.base + "/idx/" + n + ".json").then(JSON.parse);
    }
    return this.buckets[n];
  };

  // Posting list (sorted ids) for one key, [] when the key occurs nowhere.
  Index.prototype.postings = function(key) {
    return this.bucket(bucketOf(key)).then(function(bucket) {
      var deltas = bucket[key] || [], ids = [], prev = -1;
      for (var i = 0; i < deltas.length; i += 1) { prev += deltas[i] + 1; ids.push(prev); }
      return ids;
    });
  };

  Index.prototype.doc = function(item) {
    if (!this.docs[item.id]) {
      var slug = item.url.replace(/^.*\//, "").replace(/\.html$/, "");
      this.docs[item.id] = this.fetchText(this.base + "/doc/" + slug + ".txt");
    }
    return this.docs[item.id];
  };

  // Sorted union of several posting lists.
  function union(lists) {
    var seen = {}, out = [];
    lists.forEach(function(l) { l.forEach(function(id) { if (!seen[id]) { seen[id] = 1; out.push(id); } }); });
    return out.sort(function(a, b) { return a - b; });
  }

  // Candidate ids: intersection over groups of the union within each group.
  Index.prototype.candidates = function(ctx) {
    var groups = [];
    ctx.plans.forEach(function(p) { groups = groups.concat(p.keys); });
    if (!groups.length) return Promise.resolve(null);
    var self = this;
    return Promise.all(groups.map(function(alts) {
      return Promise.all(alts.map(function(k) { return self.postings(k); })).then(union);
    })).then(function(lists) {
      lists.sort(function(a, b) { return a.length - b.length; });
      var set = lists[0];
      for (var i = 1; i < lists.length && set.length; i += 1) {
        var other = lists[i], out = [], a = 0, b = 0;
        while (a < set.length && b < other.length) {
          if (set[a] === other[b]) { out.push(set[a]); a += 1; b += 1; }
          else if (set[a] < other[b]) a += 1;
          else b += 1;
        }
        set = out;
      }
      return set;
    });
  };

  function renderItem(item) {
    var tagsText = (item.tags || []).join(" ");
    return (
      '<article class="search-result-item" data-id="' + item.id + '">' +
        '<h3><a href="' + escapeHtml(item.url) + '">' + item._titleHtml + "</a></h3>" +
        '<div class="search-result-meta">' + escapeHtml(item.date || "") + (tagsText ? " · " + escapeHtml(tagsText) : "") + "</div>" +
        '<p class="search-result-snippet">&nbsp;</p>' +
      "</article>"
    );
  }

  function init(options) {
    var inputNode = document.getElementById(options.inputId);
    var statsNode = document.getElementById(options.statsId);
    var listNode = document.getElementById(options.resultsId);
    if (!inputNode || !statsNode || !listNode) return;

    var index = new Index(options);
    var state = { seq: 0, results: [], shown: 0, ctx: null, dropped: 0 };

    function statsLine() {
      var n = state.results.length - state.dropped;
      return "关键词 “" + escapeHtml(inputNode.value.trim()) + "” ，找到 " + (state.ctx.exact ? "" : "约 ") + n + " 篇文章";
    }

    // Fill in the snippet of one rendered item once its text arrives; drop it
    // if the real text does not contain every term (index approximation).
    function fillSnippet(item, seq) {
      var ctx = state.ctx;
      index.doc(item).then(function(text) {
        if (seq !== state.seq) return;
        var node = listNode.querySelector('[data-id="' + item.id + '"]');
        if (!node) return;
        if (!fullMatch(item, text, ctx)) {
          node.remove();
          state.dropped += 1;
          statsNode.innerHTML = statsLine();
          return;
        }
        node.querySelector(".search-result-snippet").innerHTML = highlightText(makeSnippet(text, ctx), ctx);
      }).catch(function() {
        if (seq !== state.seq) return;
        var node = listNode.querySelector('[data-id="' + item.id + '"] .search-result-snippet');
        if (node) node.innerHTML = "<em>摘要加载失败</em>";
      });
    }

    function showMore() {
      var seq = state.seq;
      var slice = state.results.slice(state.shown, state.shown + PAGE);
      state.shown += slice.length;
      var more = listNode.querySelector(".search-more");
      if (more) more.remove();
      listNode.insertAdjacentHTML("beforeend", slice.map(renderItem).join(""));
      if (state.shown < state.results.length) {
        listNode.insertAdjacentHTML("beforeend",
          '<p class="search-more"><a href="#" role="button">显示更多（还有 ' + (state.results.length - state.shown) + " 篇）</a></p>");
      }
      slice.forEach(function(item) { fillSnippet(item, seq); });
    }

    function runSearch() {
      var rawQuery = (inputNode.value || "").trim();
      var seq = ++state.seq;
      state.results = []; state.shown = 0; state.dropped = 0;

      if (!rawQuery) {
        statsNode.innerHTML = "输入关键词后开始搜索（支持中文和英文）";
        listNode.innerHTML = "";
        return;
      }
      var ctx = buildQueryContext(rawQuery);
      state.ctx = ctx;
      index.candidates(ctx).then(function(ids) {
        if (seq !== state.seq) return;
        if (ids === null) {
          statsNode.innerHTML = "关键词里需要有汉字、字母或数字。";
          listNode.innerHTML = "";
          return;
        }
        state.results = ids.map(function(id) { return index.items[id]; })
          .map(function(item) { return { item: item, score: scoreTitle(item, ctx) }; })
          .sort(function(a, b) {
            if (b.score !== a.score) return b.score - a.score;
            return (b.item.date || "").localeCompare(a.item.date || "");
          })
          .map(function(r) { r.item._titleHtml = highlightText(r.item.title, ctx); return r.item; });
        statsNode.innerHTML = statsLine();
        listNode.innerHTML = state.results.length ? "" : "<p>没有找到结果，试试更短的关键词或同义词。</p>";
        if (state.results.length) showMore();
      }).catch(function() {
        if (seq !== state.seq) return;
        statsNode.innerHTML = "搜索索引加载失败，请稍后重试。";
      });
    }

    listNode.addEventListener("click", function(event) {
      var link = event.target.closest(".search-more a");
      if (!link) return;
      event.preventDefault();
      showMore();
    });
    // Also page in automatically when the list is scrolled near its end.
    listNode.addEventListener("scroll", function() {
      if (!listNode.querySelector(".search-more")) return;
      if (listNode.scrollTop + listNode.clientHeight >= listNode.scrollHeight - 200) showMore();
    });

    index.load().then(function() {
      var initialQuery = getParam("q");
      if (initialQuery) inputNode.value = initialQuery;
      var debounceTimer = null;
      inputNode.addEventListener("input", function() {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(runSearch, 150);
      });
      runSearch();
    }).catch(function() {
      statsNode.innerHTML = "搜索索引加载失败，请稍后重试。";
    });
  }

  window.BLOG_SEARCH = { init: init };
})(window);
