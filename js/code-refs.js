/*!
 * code-refs.js — prose ↔ code-line links (the Code Hike "code mentions" idea).
 *
 * _plugins/code_lines.rb turns `# !ref name` in a fence into
 * `span.line.ref-line[data-ref=name]` (the directive line itself is dropped)
 * and `[text](#name)` in the prose into `a.code-ref[data-ref=name]`.
 *
 *   prose → code   hover a ref: its lines light up; click: scroll there if
 *                  needed and flash (without JS it is a plain anchor jump)
 *   code → prose   the gutter number of a referenced line is the marker:
 *                  hovering it (mouse) or tapping it (touch) opens the
 *                  explaining paragraph in the shared InlinePopover, and the
 *                  prose refs light up
 *
 * The gutter is a ::before pseudo-element, so "over the marker" is decided by
 * x-coordinate against the pre's left edge.
 */
(function () {
  'use strict';

  var HOVER = window.matchMedia && window.matchMedia('(hover: hover)').matches;

  function init() {
    var container = document.querySelector('.post-container');
    if (!container || !window.InlinePopover) return;
    var refs = container.querySelectorAll('a.code-ref[data-ref]');
    if (!refs.length) return;
    var Popover = window.InlinePopover;

    function lines(name) {
      // hidden code-tabs panels stay out of it
      return Array.prototype.filter.call(container.querySelectorAll('.line[data-ref="' + name + '"]'), function (l) {
        return !l.closest('[hidden]');
      });
    }
    function anchors(name) { return container.querySelectorAll('a.code-ref[data-ref="' + name + '"]'); }
    function setActive(name, on) {
      lines(name).forEach(function (l) { l.classList.toggle('is-active', on); });
      Array.prototype.forEach.call(anchors(name), function (a) { a.classList.toggle('is-active', on); });
    }
    function flash(el) {
      el.classList.remove('code-ref-flash');
      void el.offsetWidth;
      el.classList.add('code-ref-flash');
    }
    function inView(el) {
      var r = el.getBoundingClientRect();
      var nav = document.querySelector('nav.navbar-fixed-top');
      var top = nav ? nav.offsetHeight : 0;
      return r.top >= top && r.bottom <= window.innerHeight;
    }
    function goTo(el) {
      if (!inView(el)) Popover.scrollToTargetWithOffset(el);
      flash(el);
    }
    function blockOf(el) {
      return el.closest('li, p, td, th, dd, blockquote, h2, h3, h4') || el.parentElement;
    }

    // --- prose → code
    Array.prototype.forEach.call(refs, function (a) {
      var name = a.getAttribute('data-ref');
      if (HOVER) {
        a.addEventListener('mouseenter', function () { setActive(name, true); });
        a.addEventListener('mouseleave', function () { setActive(name, false); });
      }
      a.addEventListener('focus', function () { setActive(name, true); });
      a.addEventListener('blur', function () { setActive(name, false); });
      a.addEventListener('click', function (e) {
        var target = lines(name);
        if (!target.length) return;
        e.preventDefault();
        e.stopPropagation();
        history.pushState(null, null, '#' + name);
        goTo(target[0]);
        target.forEach(function (l) { if (l !== target[0]) flash(l); });
      });
    });

    // --- code → prose
    function explanation(name) {
      var a = anchors(name)[0];
      if (!a) return '';
      var block = blockOf(a);
      var clone = block.cloneNode(true);
      // the ref that is being explained, emphasised; other refs stay plain text
      Array.prototype.forEach.call(clone.querySelectorAll('a.code-ref'), function (x) {
        var span = document.createElement(x.getAttribute('data-ref') === name ? 'mark' : 'span');
        span.innerHTML = x.innerHTML;
        x.replaceWith(span);
      });
      Array.prototype.forEach.call(clone.querySelectorAll('.annotation-marker, .sec-react, sup[id^="fnref"]'), function (x) { x.remove(); });
      var wrap = document.createElement('div');
      wrap.className = 'popover-code-ref';
      wrap.appendChild(clone);
      var jump = document.createElement('div');
      jump.className = 'popover-jump-footnote';
      jump.innerHTML = '<a href="#" data-jump>查看说明 <i class="fa fa-angle-double-down"></i></a>';
      jump.querySelector('a').addEventListener('click', function (e) {
        e.preventDefault(); e.stopPropagation();
        Popover.hide();
        goTo(block);
      });
      wrap.appendChild(jump);
      return wrap;
    }

    // a stand-in trigger so the card is anchored to the gutter number, not the whole line
    function gutterTrigger(line, gutter) {
      return {
        line: line,
        getBoundingClientRect: function () {
          var r = line.getBoundingClientRect();
          return { top: r.top, bottom: r.bottom, height: r.height, left: gutter.left, right: gutter.right, width: gutter.right - gutter.left };
        },
        contains: function (el) { return line.contains(el); }
      };
    }

    Array.prototype.forEach.call(container.querySelectorAll('pre[data-refs]'), function (pre) {
      var hovered = null;
      var gutterWidth = 0;
      function measure() {
        var first = pre.querySelector('.line');
        if (!first) return;
        var s = window.getComputedStyle(first, '::before');
        gutterWidth = ['width', 'padding-left', 'padding-right'].reduce(function (w, p) { return w + (parseFloat(s.getPropertyValue(p)) || 0); }, 0);
      }
      function gutterOf(e) {
        var line = e.target.closest && e.target.closest('.line.ref-line');
        if (!line) return null;
        if (!gutterWidth) measure();
        var left = pre.getBoundingClientRect().left;
        return e.clientX >= left && e.clientX <= left + gutterWidth ? { line: line, rect: { left: left, right: left + gutterWidth } } : null;
      }
      function leave() {
        if (!hovered) return;
        setActive(hovered.getAttribute('data-ref'), false);
        hovered.classList.remove('is-gutter-hover');
        hovered = null;
        Popover.scheduleHide();
      }
      function enter(hit) {
        if (hovered === hit.line) return;
        leave();
        hovered = hit.line;
        var name = hit.line.getAttribute('data-ref');
        hit.line.classList.add('is-gutter-hover');
        setActive(name, true);
        Popover.scheduleShow(gutterTrigger(hit.line, hit.rect), function () { return explanation(name); }, { delay: 150 });
      }

      if (HOVER) {
        pre.addEventListener('mousemove', function (e) {
          var hit = gutterOf(e);
          if (hit) enter(hit); else leave();
        });
        pre.addEventListener('mouseleave', leave);
      }
      pre.addEventListener('click', function (e) {
        var hit = gutterOf(e);
        if (!hit) return;
        e.preventDefault();
        e.stopPropagation();
        var name = hit.line.getAttribute('data-ref');
        var current = Popover.currentTrigger();
        if (current && current.line === hit.line && Popover.isActive()) {
          Popover.hide();
          setActive(name, false);
        } else {
          setActive(name, true);
          Popover.show(gutterTrigger(hit.line, hit.rect), explanation(name));
        }
      });
    });

    // a `#name` in the URL on load: the id line is CSS (:target); flash the
    // whole range and the prose refs too
    if (location.hash) {
      var hit = container.querySelector('.line[id="' + location.hash.slice(1).replace(/"/g, '') + '"]');
      if (hit) {
        var name = hit.getAttribute('data-ref');
        lines(name).forEach(flash);
        Array.prototype.forEach.call(anchors(name), flash);
      }
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
