/*!
 * Clean Blog v1.0.0 (http://startbootstrap.com) — Copyright 2015 Start Bootstrap, Apache 2.0
 * Hux Blog v1.6.0 — Copyright 2016 @huxpro, Apache 2.0
 * Rewritten without jQuery (the blog loads no jQuery / Bootstrap JS any more).
 */
(function () {
  'use strict';

  function ready(fn) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn); else fn(); }
  function wrap(el, className) {
    var box = document.createElement('div');
    box.className = className;
    el.parentNode.insertBefore(box, el);
    box.appendChild(el);
  }

  ready(function () {
    // responsive tables (Bootstrap CSS classes; the CSS is still Bootstrap 3)
    Array.prototype.forEach.call(document.querySelectorAll('table'), function (t) {
      if (!t.parentNode.classList.contains('table-responsive')) wrap(t, 'table-responsive');
      t.classList.add('table');
    });

    // responsive embedded videos
    Array.prototype.forEach.call(document.querySelectorAll('iframe[src*="youtube.com"], iframe[src*="vimeo.com"]'), function (f) {
      if (!f.parentNode.classList.contains('embed-responsive')) wrap(f, 'embed-responsive embed-responsive-16by9');
      f.classList.add('embed-responsive-item');
    });

    // Navigation: hide the navbar on scroll-down, slide it back in on scroll-up
    // (desktop only); pin the side catalog once the header has scrolled away.
    var MQL = 1170;
    var nav = document.querySelector('.navbar-custom');
    if (document.documentElement.clientWidth <= MQL) return;
    var headerHeight = nav ? nav.offsetHeight : 0;
    var banner = document.querySelector('.intro-header .container');
    var bannerHeight = banner ? banner.offsetHeight : 0;
    var previousTop = 0;
    window.addEventListener('scroll', function () {
      var currentTop = window.pageYOffset || document.documentElement.scrollTop || 0;
      if (nav) {
        if (currentTop < previousTop) {
          // scrolling up
          if (currentTop > 0 && nav.classList.contains('is-fixed')) nav.classList.add('is-visible');
          else nav.classList.remove('is-visible', 'is-fixed');
        } else {
          // scrolling down
          nav.classList.remove('is-visible');
          if (currentTop > headerHeight && !nav.classList.contains('is-fixed')) nav.classList.add('is-fixed');
        }
      }
      previousTop = currentTop;
      var catalog = document.querySelector('.side-catalog');
      if (catalog) catalog.classList.toggle('fixed', currentTop > bannerHeight + 41);
    }, { passive: true });
  });
})();
