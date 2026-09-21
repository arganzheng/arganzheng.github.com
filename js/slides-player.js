/*!
 * slides-player.js — the web-PowerPoint viewer on a deck's landing page
 * (_layouts/slides.html): thumbnail rail on the left, the reveal.js deck in a
 * same-origin iframe (`/slides/foo/play.html?embed`) on the right, driven
 * through the iframe's `Reveal` API.
 *
 *   - bar / edge buttons: prev, next, fullscreen; position `cur / total`
 *   - ← → / Space / F on the page work when the pointer was last on the player
 *     (the iframe grabs keys itself once it has focus)
 *   - thumbnails: the slide HTML in a 1280×720 box, scaled to the rail width
 *     (--thumb-scale); the current page is `.is-current` and kept in view;
 *     click = jump. `data-background*` from `<!-- .slide: … -->` is applied,
 *     dark colours flip the text light like reveal does
 *   - `#/N` in the landing URL = page N (replaceState, so stepping through a
 *     deck leaves no history trail)
 */
(function () {
  'use strict';

  function init() {
    var studio = document.querySelector('.deck-studio');
    if (!studio) return;
    var player = studio.querySelector('.deck-player');
    var frame = player.querySelector('.deck-player-frame');
    var cur = player.querySelector('.dp-cur');
    var total = player.querySelector('.dp-total');
    var rail = studio.querySelector('.deck-rail');
    var thumbs = Array.prototype.slice.call(studio.querySelectorAll('.deck-thumb'));
    var R = null;
    var armed = false;   // pointer was last over the player: keys steer it

    // ------------------------------------------------------------ thumbnails

    function scaleThumbs() {
      var f = studio.querySelector('.deck-thumb-frame');
      if (!f) return;
      var w = f.clientWidth;
      if (w) studio.style.setProperty('--thumb-scale', (w / 1280).toFixed(4));
      // the rail is as tall as the player, so both scroll together
      if (window.matchMedia('(min-width: 992px)').matches) rail.style.setProperty('--rail-h', player.offsetHeight + 'px');
      else rail.style.removeProperty('--rail-h');
    }
    thumbs.forEach(function (t) {
      var inner = t.querySelector('.deck-thumb-inner');
      var bg = inner.getAttribute('data-background-color') || inner.getAttribute('data-background');
      var img = inner.getAttribute('data-background-image');
      if (bg && /^(#|rgb|hsl|[a-z]+$)/i.test(bg)) {
        inner.style.background = bg;
        if (isDark(bg)) inner.classList.add('is-dark');
      }
      if (img) {
        inner.style.backgroundImage = 'url(' + img + ')';
        inner.style.backgroundSize = inner.getAttribute('data-background-size') || 'cover';
        inner.style.backgroundPosition = 'center';
        inner.classList.add('is-dark');
      }
    });
    function isDark(c) {
      var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c.trim());
      if (!m) return false;
      var h = m[1].length === 3 ? m[1].replace(/./g, '$&$&') : m[1];
      var r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16), b = parseInt(h.slice(4, 6), 16);
      return (r * 299 + g * 587 + b * 114) / 1000 < 128;
    }
    if (window.ResizeObserver) new ResizeObserver(scaleThumbs).observe(player);
    window.addEventListener('resize', scaleThumbs);
    scaleThumbs();

    function markCurrent(n) {
      thumbs.forEach(function (t) {
        var on = +t.getAttribute('data-page') === n;
        t.classList.toggle('is-current', on);
        if (on) t.setAttribute('aria-current', 'true'); else t.removeAttribute('aria-current');
        if (on) keepInView(t);
      });
    }
    function keepInView(t) {
      var horizontal = rail.scrollWidth > rail.clientWidth + 1;
      if (horizontal) {
        var l = t.offsetLeft, r = l + t.offsetWidth;
        if (l < rail.scrollLeft) rail.scrollLeft = l - 8;
        else if (r > rail.scrollLeft + rail.clientWidth) rail.scrollLeft = r - rail.clientWidth + 8;
      } else {
        var top = t.offsetTop - rail.offsetTop, bot = top + t.offsetHeight;
        if (top < rail.scrollTop) rail.scrollTop = top - 8;
        else if (bot > rail.scrollTop + rail.clientHeight) rail.scrollTop = bot - rail.clientHeight + 8;
      }
    }
    thumbs.forEach(function (t) {
      t.querySelector('.deck-thumb-link').addEventListener('click', function (e) {
        if (!R) return;   // no Reveal yet: the #/N anchor is picked up by fromHash later
        e.preventDefault();
        goTo(+t.getAttribute('data-page'));
        armed = true;
      });
    });

    // ---------------------------------------------------------------- player

    function slides() { return R.getSlides(); }
    function index() { return slides().indexOf(R.getCurrentSlide()); }

    function goTo(n) {   // 1-based flat page number
      if (!R) return;
      var s = slides()[n - 1];
      if (!s) return;
      var idx = R.getIndices(s);
      R.slide(idx.h, idx.v);
    }

    function update() {
      var n = index() + 1, t = slides().length;
      cur.textContent = n;
      total.textContent = t;
      player.classList.toggle('at-start', n <= 1);
      player.classList.toggle('at-end', n >= t);
      markCurrent(n);
      var h = '#/' + n;
      if (location.hash !== h && (n > 1 || location.hash)) {
        try { history.replaceState(null, '', location.pathname + location.search + (n > 1 ? h : '')); } catch (e) { /* file:// */ }
      }
    }

    function act(what) {
      if (!R) return;
      if (what === 'prev') R.prev();
      else if (what === 'next') R.next();
      else if (what === 'fullscreen') toggleFullscreen();
    }

    function toggleFullscreen() {
      var fs = document.fullscreenElement || document.webkitFullscreenElement;
      if (fs) { (document.exitFullscreen || document.webkitExitFullscreen).call(document); return; }
      (player.requestFullscreen || player.webkitRequestFullscreen).call(player);
    }
    function onFullscreen() {
      var on = (document.fullscreenElement || document.webkitFullscreenElement) === player;
      player.classList.toggle('is-fullscreen', on);
      var ic = player.querySelector('[data-act="fullscreen"] .fa');
      if (ic) { ic.classList.toggle('fa-expand', !on); ic.classList.toggle('fa-compress', on); }
      if (R) setTimeout(function () { R.layout(); }, 50);
    }
    document.addEventListener('fullscreenchange', onFullscreen);
    document.addEventListener('webkitfullscreenchange', onFullscreen);

    player.addEventListener('click', function (e) {
      var b = e.target.closest('[data-act]');
      if (!b) return;
      e.preventDefault();
      act(b.getAttribute('data-act'));
      armed = true;
    });
    studio.addEventListener('mouseenter', function () { armed = true; });
    document.addEventListener('mousedown', function (e) { armed = !!e.target.closest('.deck-studio'); });
    document.addEventListener('keydown', function (e) {
      if (!armed || !R || e.altKey || e.ctrlKey || e.metaKey) return;
      var t = e.target;
      if (t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable)) return;
      if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); R.prev(); }
      else if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); R.next(); }
      else if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFullscreen(); }
    });

    function fromHash() {
      var m = /^#\/(\d+)$/.exec(location.hash);
      if (m) goTo(+m[1]);
    }
    window.addEventListener('hashchange', fromHash);

    // Reveal inside the iframe: same origin, so talk to it directly. It may
    // not exist yet when the iframe's `load` fires (its script is deferred
    // by the CDN load), hence the short poll.
    function attach() {
      var w = frame.contentWindow;
      var r = w && w.Reveal;
      if (!r || !r.isReady || !r.isReady()) { setTimeout(attach, 60); return; }
      R = r;
      R.on('slidechanged', update);
      fromHash();
      update();
      player.classList.add('is-ready');
    }
    if (frame.contentDocument && frame.contentDocument.readyState === 'complete' && frame.contentWindow.Reveal) attach();
    else frame.addEventListener('load', attach);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
