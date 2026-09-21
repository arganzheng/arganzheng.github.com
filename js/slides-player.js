/*!
 * slides-player.js — the SlideShare-style player on a deck's landing page
 * (_layouts/slides.html). The reveal.js deck runs in a same-origin iframe
 * (`/slides/foo/play.html?embed`); this script drives it through the iframe's
 * `Reveal` API:
 *
 *   - bar / edge buttons: prev, next, fullscreen; position `cur / total`
 *   - ← → / Space / F on the page work when the pointer was last on the player
 *     (the iframe grabs keys itself once it has focus)
 *   - `#/N` in the landing URL = page N in the player (replaceState, so
 *     stepping through a deck leaves no history trail); the flat copies below
 *     are `#pN` — clicking a page number there plays that page
 *   - the flat page currently shown in the player is marked `.is-current`
 */
(function () {
  'use strict';

  function init() {
    var player = document.querySelector('.deck-player');
    if (!player) return;
    var frame = player.querySelector('.deck-player-frame');
    var cur = player.querySelector('.dp-cur');
    var total = player.querySelector('.dp-total');
    var pages = Array.prototype.slice.call(document.querySelectorAll('.deck-pages .deck-page'));
    var R = null;
    var armed = false;   // pointer was last over the player: keys steer it

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
      pages.forEach(function (p) { p.classList.toggle('is-current', +p.getAttribute('data-page') === n); });
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
      var el = player;
      (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
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
    player.addEventListener('mouseenter', function () { armed = true; });
    document.addEventListener('mousedown', function (e) { armed = !!e.target.closest('.deck-player'); });
    document.addEventListener('keydown', function (e) {
      if (!armed || !R || e.altKey || e.ctrlKey || e.metaKey) return;
      var t = e.target;
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || (t && t.isContentEditable)) return;
      if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); R.prev(); }
      else if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { e.preventDefault(); R.next(); }
      else if (e.key === 'f' || e.key === 'F') { e.preventDefault(); toggleFullscreen(); }
    });

    // flat page numbers: play that page (the anchor still works without JS)
    pages.forEach(function (p) {
      var a = p.querySelector('.deck-page-no');
      if (!a) return;
      a.addEventListener('click', function (e) {
        if (!R) return;
        e.preventDefault();
        goTo(+p.getAttribute('data-page'));
        player.scrollIntoView({ behavior: 'smooth', block: 'start' });
        armed = true;
      });
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
