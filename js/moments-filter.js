/*! moments-filter.js — client-side filters for every Moments stream and the heatmap toggle. */
(function () {
  var heat = document.querySelector('.ms-heat');
  if (heat) {
    var views = Array.prototype.slice.call(heat.querySelectorAll('[data-heat-view]'));
    var buttons = Array.prototype.slice.call(heat.querySelectorAll('[data-heat-mode]'));
    var year = heat.querySelector('[data-heat-view="year"]');

    function setHeatMode(mode) {
      views.forEach(function (view) { view.hidden = view.getAttribute('data-heat-view') !== mode; });
      buttons.forEach(function (button) {
        button.setAttribute('aria-pressed', button.getAttribute('data-heat-mode') === mode ? 'true' : 'false');
      });
      try { localStorage.setItem('moments-heat', mode); } catch (e) { /* storage may be disabled */ }
      if (mode === 'year' && year) {
        var scroll = year.querySelector('.ms-heat-scroll');
        if (scroll) window.requestAnimationFrame(function () { scroll.scrollLeft = scroll.scrollWidth; });
      }
    }

    buttons.forEach(function (button) {
      button.addEventListener('click', function () { setHeatMode(button.getAttribute('data-heat-mode')); });
    });
    var initialMode = 'recent';
    try { if (localStorage.getItem('moments-heat') === 'year') initialMode = 'year'; } catch (e) { /* storage may be disabled */ }
    setHeatMode(initialMode);
  }

  var filter = document.querySelector('.moments-filter');
  var stream = document.querySelector('.post-container.moments');
  if (!filter || !stream) return;

  var input = filter.querySelector('.mf-search input');
  var clear = filter.querySelector('.mf-clear');
  var chips = Array.prototype.slice.call(filter.querySelectorAll('.mf-chip'));
  var dateToggle = filter.querySelector('.mf-date-toggle');
  var dateRange = filter.querySelector('.mf-date-range');
  var dates = Array.prototype.slice.call(filter.querySelectorAll('[data-filter-date]'));
  var resultsWrap = filter.querySelector('.mf-results-wrap');
  var count = filter.querySelector('.mf-count');
  var results = filter.querySelector('.mf-results');
  var none = filter.querySelector('.mf-none');
  var more = filter.querySelector('.mf-more');
  var src = filter.getAttribute('data-src');
  var base = filter.getAttribute('data-base') || '';
  var indexPromise = window.MomentsIndex && window.MomentsIndex.load ?
    window.MomentsIndex.load(src) :
    fetch(src).then(function (r) { return r.ok ? r.json() : []; });
  var visibleCount = 50;
  var timer;

  function esc(text) {
    return String(text || '').replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function regexEsc(text) { return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
  function state() {
    var selected = {};
    chips.forEach(function (button) { selected[button.getAttribute('data-filter')] = button.getAttribute('aria-pressed') === 'true'; });
    var from = filter.querySelector('[data-filter-date="from"]').value;
    var to = filter.querySelector('[data-filter-date="to"]').value;
    var query = input.value.trim();
    return {
      q: query,
      terms: query.split(/\s+/).filter(Boolean),
      img: selected.img,
      quote: selected.quote,
      refs: selected.refs,
      from: from,
      to: to,
      active: !!(query || selected.img || selected.quote || selected.refs || from || to)
    };
  }
  function highlighted(value, terms) {
    var text = esc(value);
    if (!terms.length) return text;
    var alternatives = terms.slice().sort(function (a, b) { return b.length - a.length; }).map(function (term) {
      return regexEsc(esc(term));
    });
    return text.replace(new RegExp('(' + alternatives.join('|') + ')', 'gi'), '<mark>$1</mark>');
  }
  function snippet(value, terms) {
    var text = String(value || ''), limit = 140;
    if (text.length <= limit) return text;
    var lower = text.toLowerCase(), first = -1, firstLength = 0;
    terms.forEach(function (term) {
      var index = lower.indexOf(term.toLowerCase());
      if (index >= 0 && (first < 0 || index < first || (index === first && term.length > firstLength))) {
        first = index;
        firstLength = term.length;
      }
    });
    var start = first >= 0 && first + firstLength > limit - 1 ? Math.max(0, first - 30) : 0;
    var leading = start > 0;
    var end = Math.min(text.length, start + limit - (leading ? 1 : 0));
    if (end < text.length) end -= 1;
    return (leading ? '…' : '') + text.slice(start, end) + (end < text.length ? '…' : '');
  }
  function matches(entry, current) {
    if (current.img && !entry.img) return false;
    if (current.quote && !entry.quote) return false;
    if (current.refs && !entry.refs) return false;
    if (current.from && entry.month < current.from) return false;
    if (current.to && entry.month > current.to) return false;
    var searchable = [entry.text || '', (entry.tags || []).join(' '), entry.place || ''];
    return current.terms.every(function (term) {
      var needle = term.toLowerCase();
      return searchable.some(function (field) { return field.toLowerCase().indexOf(needle) !== -1; });
    });
  }
  function resultCard(entry, terms) {
    var href = base + entry.url;
    var date = entry.date + (entry.time ? ' ' + entry.time : '');
    var text = snippet(entry.text, terms);
    var place = entry.place ? '<span class="moment-place">· ' + highlighted(entry.place, terms) + '</span>' : '';
    var tags = (entry.tags || []).map(function (tag) {
      return '<a class="moment-tag" href="' + esc(base + '/moments/tag/' + tag + '.html') + '">' + highlighted('#' + tag, terms) + '</a>';
    }).join('');
    var image = entry.img ? '<img class="mf-result-thumb" src="' + esc(base + entry.img) + '" alt="" loading="lazy">' : '';
    return '<li class="moment mf-result" data-entry-id="' + esc(entry.id) + '">' +
      '<div class="moment-head"><a class="moment-when" href="' + esc(href) + '"><time>' + highlighted(date, terms) + '</time></a>' + place + '</div>' +
      '<div class="moment-body mf-result-body"><div class="mf-result-copy">' +
      (text ? '<p>' + highlighted(text, terms) + '</p>' : '') +
      (tags ? '<div class="mf-result-tags">' + tags + '</div>' : '') +
      '</div>' + image + '</div></li>';
  }
  function draw(current) {
    return indexPromise.then(function (entries) {
      var found = (entries || []).filter(function (entry) { return matches(entry, current); });
      count.textContent = '找到 ' + found.length + ' 条';
      results.innerHTML = found.slice(0, visibleCount).map(function (entry) {
        return resultCard(entry, current.terms);
      }).join('');
      none.hidden = found.length > 0;
      more.hidden = found.length <= visibleCount;
    }).catch(function () {
      count.textContent = '找到 0 条';
      results.innerHTML = '';
      none.hidden = false;
      more.hidden = true;
    });
  }
  function writeUrl(current) {
    var url = new URL(window.location.href);
    ['q', 'img', 'quote', 'refs', 'from', 'to'].forEach(function (key) { url.searchParams.delete(key); });
    if (current.q) url.searchParams.set('q', current.q);
    if (current.img) url.searchParams.set('img', '1');
    if (current.quote) url.searchParams.set('quote', '1');
    if (current.refs) url.searchParams.set('refs', '1');
    if (current.from) url.searchParams.set('from', current.from);
    if (current.to) url.searchParams.set('to', current.to);
    window.history.replaceState(window.history.state, '', url.pathname + url.search + url.hash);
  }
  function render(updateUrl) {
    var current = state();
    stream.classList.toggle('is-filtering', current.active);
    resultsWrap.hidden = !current.active;
    clear.hidden = !current.active;
    dateToggle.classList.toggle('is-active', !!(current.from || current.to));
    if (current.active) {
      visibleCount = 50;
      draw(current);
    } else {
      results.innerHTML = '';
      none.hidden = true;
      more.hidden = true;
    }
    if (updateUrl) writeUrl(current);
  }
  function clearAll() {
    input.value = '';
    chips.forEach(function (button) { button.setAttribute('aria-pressed', 'false'); button.classList.remove('is-active'); });
    dates.forEach(function (date) { date.value = ''; });
    render(true);
  }
  function scheduleRender() {
    clearTimeout(timer);
    timer = setTimeout(function () { render(true); }, 150);
  }

  var params = new URLSearchParams(window.location.search);
  input.value = params.get('q') || '';
  chips.forEach(function (button) {
    var active = params.get(button.getAttribute('data-filter')) === '1';
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
    button.classList.toggle('is-active', active);
    button.addEventListener('click', function () {
      var next = button.getAttribute('aria-pressed') !== 'true';
      button.setAttribute('aria-pressed', next ? 'true' : 'false');
      button.classList.toggle('is-active', next);
      render(true);
    });
  });
  dates.forEach(function (date) {
    date.value = params.get(date.getAttribute('data-filter-date')) || '';
    date.addEventListener('change', function () { render(true); });
  });
  input.addEventListener('input', scheduleRender);
  input.addEventListener('search', function () { if (!input.value) clearAll(); else render(true); });
  input.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') { event.preventDefault(); clearAll(); }
  });
  clear.addEventListener('click', clearAll);
  dateToggle.addEventListener('click', function () {
    var open = dateRange.classList.toggle('is-open');
    dateToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  filter.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && state().active) { event.preventDefault(); clearAll(); }
  });
  more.addEventListener('click', function () {
    visibleCount += 50;
    draw(state());
  });
  render(false);
})();
