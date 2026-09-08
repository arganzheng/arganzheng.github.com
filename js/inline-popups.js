/**
 * inline-popups.js
 * 
 * 1. External Links: automatically decorates external links in .post-container
 *    with dashed underline, target="_blank", rel="noopener noreferrer", and ↗ icon.
 * 2. Popup Footnotes: transforms standard Kramdown footnotes ([^1]) into
 *    interactive popup cards on hover/click without jumping to the bottom.
 * 3. Inline Tips: renders dashed underlines with an icon on terms annotated with
 *    inline tips ([term](# "tip:..."), .tip, .inline-tip, or {% include tip.html %})
 *    and displays a floating rich popover on hover.
 */

(function () {
  'use strict';

  function initInlinePopups() {
    var container = document.querySelector('.post-container');
    if (!container) return;

    var popover = createPopover();
    var currentTrigger = null;
    var showTimer = null;
    var hideTimer = null;

    // 1. Process External Links
    processExternalLinks(container);

    // 2. Process Standard Kramdown Footnotes
    processFootnotes(container, popover);

    // 3. Process Reverse Footnote Back-links
    processReverseFootnotes();

    // 4. Process Inline Tips
    processInlineTips(container, popover);

    // --- Helpers ---

    function scrollToTargetWithOffset(targetEl) {
      if (!targetEl) return;
      var navbar = document.querySelector('nav.navbar-fixed-top');
      var navHeight = navbar ? navbar.offsetHeight : 65;
      var targetRect = targetEl.getBoundingClientRect();
      var targetTop = targetRect.top + (window.pageYOffset || document.documentElement.scrollTop) - navHeight - 16;

      window.scrollTo({
        top: Math.max(0, targetTop),
        behavior: 'smooth'
      });

      targetEl.classList.remove('footnote-highlight-flash');
      void targetEl.offsetWidth; // trigger reflow
      targetEl.classList.add('footnote-highlight-flash');
    }

    function createPopover() {
      var existing = document.getElementById('inline-popover-card');
      if (existing) return existing;

      var card = document.createElement('div');
      card.id = 'inline-popover-card';
      card.className = 'inline-popover-card';
      card.setAttribute('role', 'tooltip');
      card.setAttribute('aria-hidden', 'true');

      var arrow = document.createElement('div');
      arrow.className = 'popover-arrow';
      card.appendChild(arrow);

      var content = document.createElement('div');
      content.className = 'popover-content';
      card.appendChild(content);

      document.body.appendChild(card);

      // Mouse events on popover to keep it open when hovering inside
      card.addEventListener('mouseenter', function () {
        clearTimeout(hideTimer);
      });

      card.addEventListener('mouseleave', function () {
        scheduleHide();
      });

      // Close on Esc
      document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' || e.keyCode === 27) {
          hidePopover();
        }
      });

      // Close on outside click
      document.addEventListener('click', function (e) {
        if (!card.contains(e.target) && (!currentTrigger || !currentTrigger.contains(e.target))) {
          hidePopover();
        }
      });

      // Jump to bottom footnote from inside popover
      card.addEventListener('click', function (e) {
        var jumpLink = e.target.closest('.popover-jump-footnote a');
        if (jumpLink) {
          e.preventDefault();
          var hash = jumpLink.hash ? jumpLink.hash.slice(1) : '';
          var targetLi = document.getElementById(hash);
          hidePopover();
          if (targetLi) {
            history.pushState(null, null, '#' + hash);
            scrollToTargetWithOffset(targetLi);
          }
        }
      });

      // Re-render math when KaTeX finishes loading
      document.addEventListener('richcontent:rendered', function () {
        if (popover && popover.classList.contains('is-active')) {
          var contentEl = popover.querySelector('.popover-content');
          if (contentEl) renderMathIfPresent(contentEl);
        }
      });

      return card;
    }

    function scheduleShow(trigger, contentProvider) {
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      showTimer = setTimeout(function () {
        showPopover(trigger, contentProvider);
      }, 100);
    }

    function scheduleHide() {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
      hideTimer = setTimeout(function () {
        hidePopover();
      }, 200);
    }

    function showPopover(trigger, contentProvider) {
      currentTrigger = trigger;
      var contentEl = popover.querySelector('.popover-content');
      var htmlContent = typeof contentProvider === 'function' ? contentProvider() : contentProvider;
      contentEl.innerHTML = htmlContent;
      renderMathIfPresent(contentEl);

      popover.style.display = 'block';
      popover.style.visibility = 'hidden';
      popover.setAttribute('aria-hidden', 'false');

      // Calculate position
      var triggerRect = trigger.getBoundingClientRect();
      var popoverRect = popover.getBoundingClientRect();
      var arrow = popover.querySelector('.popover-arrow');

      var scrollX = window.pageXOffset || document.documentElement.scrollLeft || 0;
      var scrollY = window.pageYOffset || document.documentElement.scrollTop || 0;

      var gap = 9;
      var placement = 'top';

      // Check if space on top is sufficient
      if (triggerRect.top - popoverRect.height - gap < 10) {
        placement = 'bottom';
      }

      var top;
      if (placement === 'top') {
        top = triggerRect.top + scrollY - popoverRect.height - gap;
        popover.classList.remove('placement-bottom');
        popover.classList.add('placement-top');
      } else {
        top = triggerRect.bottom + scrollY + gap;
        popover.classList.remove('placement-top');
        popover.classList.add('placement-bottom');
      }

      // Horizontal centering
      var left = triggerRect.left + scrollX + (triggerRect.width / 2) - (popoverRect.width / 2);
      var minLeft = scrollX + 12;
      var maxLeft = scrollX + document.documentElement.clientWidth - popoverRect.width - 12;
      left = Math.max(minLeft, Math.min(left, maxLeft));

      popover.style.top = top + 'px';
      popover.style.left = left + 'px';

      // Position the arrow directly above/below the center of trigger
      var triggerCenterX = triggerRect.left + scrollX + (triggerRect.width / 2);
      var arrowLeft = triggerCenterX - left;
      arrow.style.left = Math.max(12, Math.min(arrowLeft, popoverRect.width - 12)) + 'px';

      popover.style.visibility = 'visible';
      popover.classList.add('is-active');
    }

    function hidePopover() {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
      if (popover) {
        popover.classList.remove('is-active');
        popover.setAttribute('aria-hidden', 'true');
        setTimeout(function () {
          if (!popover.classList.contains('is-active')) {
            popover.style.display = 'none';
          }
        }, 150);
      }
      currentTrigger = null;
    }

    function simpleMarkdownToHtml(text) {
      if (!text) return '';
      var hasHtml = /<[a-z][\s\S]*>/i.test(text);
      var safe = text;
      if (!hasHtml) {
        var div = document.createElement('div');
        div.textContent = text;
        safe = div.innerHTML;
      }

      // Inline code `code`
      safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
      // Bold **text** or __text__
      safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      safe = safe.replace(/__([^_]+)__/g, '<strong>$1</strong>');
      // Italic *text* or _text_
      safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
      // Markdown links [text](url)
      safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      // Line breaks
      safe = safe.replace(/\n\n+/g, '<p></p>');
      safe = safe.replace(/\n/g, '<br/>');

      return safe;
    }

    // --- 1. Process External Links ---
    function processExternalLinks(root) {
      var links = root.querySelectorAll('a[href]');

      for (var i = 0; i < links.length; i++) {
        var a = links[i];
        var href = a.getAttribute('href') || '';

        // Skip internal, anchors, mailto, footnotes, reversefootnotes
        if (!/^https?:\/\//i.test(href)) continue;
        if (a.classList.contains('footnote') || a.classList.contains('reversefootnote')) continue;
        if (a.querySelector('img')) continue; // skip image links
        if (!a.textContent.trim()) continue; // skip empty links (e.g. icon widgets)

        // Exclude pagination, related-posts, share widgets, and disqus comments
        if (a.closest('.pager, .related-posts, .share, .comment, .jiathis_style, .side-catalog')) continue;

        try {
          var url = new URL(href, window.location.origin);
          // Check if external
          if (url.hostname && !isInternalHost(url.hostname)) {
            a.classList.add('external-link');
            a.setAttribute('target', '_blank');
            a.setAttribute('rel', 'noopener noreferrer');
          }
        } catch (e) {
          // ignore malformed URLs
        }
      }
    }

    function renderMathIfPresent(el) {
      if (typeof renderMathInElement === 'function') {
        renderMathInElement(el, {
          delimiters: [
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false },
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false }
          ],
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
          ignoredClasses: ['mermaid', 'highlight', 'highlighter-rouge'],
          throwOnError: false,
          errorColor: '#cf222e'
        });
      }
    }

    function isInternalHost(host) {
      if (!host) return true;
      if (host === window.location.hostname) return true;
      if (host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0') return true;
      if (host === 'arganzheng.life' || host === 'arganzheng.github.io' || host === 'arganzheng.github.com') return true;
      return false;
    }

    function getFootnoteContent(hash) {
      var targetLi = document.getElementById(hash);
      if (!targetLi) return '';

      // Clone target LI content and clean up reversefootnote link
      var clone = targetLi.cloneNode(true);
      var backLinks = clone.querySelectorAll('.reversefootnote');
      for (var j = 0; j < backLinks.length; j++) {
        backLinks[j].parentNode.removeChild(backLinks[j]);
      }
      var html = clone.innerHTML.trim();
      html += '<div class="popover-jump-footnote"><a href="#' + hash + '">查看文末完整脚注 <i class="fa fa-angle-double-down"></i></a></div>';
      return html;
    }

    // --- 2. Process Standard Kramdown Footnotes ---
    function processFootnotes(root, popover) {
      var footnoteLinks = root.querySelectorAll('sup a.footnote');
      if (!footnoteLinks.length) return;

      for (var i = 0; i < footnoteLinks.length; i++) {
        (function () {
          var a = footnoteLinks[i];
          var sup = a.closest('sup');
          var href = a.getAttribute('href') || '';
          var hash = a.hash ? a.hash.slice(1) : href.replace(/^#/, '');
          if (!hash) return;

          var targetTrigger = sup || a;
          targetTrigger.classList.add('has-popup-footnote');

          targetTrigger.addEventListener('mouseenter', function () {
            scheduleShow(targetTrigger, function () {
              return getFootnoteContent(hash);
            });
          });

          targetTrigger.addEventListener('mouseleave', function () {
            scheduleHide();
          });

          targetTrigger.addEventListener('click', function (e) {
            var isTouch = e.pointerType === 'touch' || (e.sourceCapabilities && e.sourceCapabilities.firesTouchEvents);
            if (isTouch) {
              // On touch device, tapping toggles the popup
              if (currentTrigger === targetTrigger && popover.classList.contains('is-active')) {
                hidePopover();
                e.preventDefault();
              } else {
                e.preventDefault();
                showPopover(targetTrigger, function () {
                  return getFootnoteContent(hash);
                });
              }
            } else {
              // On desktop with mouse, click smoothly jumps to the footnote at the bottom!
              hidePopover();
              var targetLi = document.getElementById(hash);
              if (targetLi) {
                e.preventDefault();
                history.pushState(null, null, '#' + hash);
                scrollToTargetWithOffset(targetLi);
              }
            }
          });
        })();
      }
    }

    // --- 3. Process Reverse Footnotes (Jump back to reference) ---
    function processReverseFootnotes() {
      var backLinks = document.querySelectorAll('.footnotes a.reversefootnote');
      for (var i = 0; i < backLinks.length; i++) {
        (function (link) {
          link.addEventListener('click', function (e) {
            var href = link.getAttribute('href') || '';
            var hash = link.hash ? link.hash.slice(1) : href.replace(/^#/, '');
            var target = document.getElementById(hash);
            if (target) {
              e.preventDefault();
              history.pushState(null, null, '#' + hash);
              scrollToTargetWithOffset(target);
            }
          });
        })(backLinks[i]);
      }
    }

    // --- 4. Process Inline Tips ---
    function processInlineTips(root, popover) {
      // 3.1 Elements with .inline-tip
      var tipEls = root.querySelectorAll('.inline-tip');
      for (var i = 0; i < tipEls.length; i++) {
        bindTipElement(tipEls[i]);
      }

      // 3.2 Anchors with title="tip: ..." or class="tip"
      var anchors = root.querySelectorAll('a[title], a.tip');
      for (var j = 0; j < anchors.length; j++) {
        var a = anchors[j];
        var title = a.getAttribute('title') || '';
        var dataTip = a.getAttribute('data-tip') || '';

        var tipText = '';
        if (/^tips?:\s*/i.test(title)) {
          tipText = title.replace(/^tips?:\s*/i, '').trim();
          a.removeAttribute('title'); // Prevent native browser tooltip
        } else if (a.classList.contains('tip') && title) {
          tipText = title.trim();
          a.removeAttribute('title');
        } else if (dataTip) {
          tipText = dataTip.trim();
        }

        if (tipText) {
          a.classList.add('inline-tip');
          a.setAttribute('data-tip', tipText);
          bindTipElement(a);
        }
      }

      function bindTipElement(el) {
        if (el.getAttribute('data-bound-tip')) return;
        el.setAttribute('data-bound-tip', 'true');

        var rawTip = el.getAttribute('data-tip') || '';
        var url = el.getAttribute('data-url');
        var htmlContent = simpleMarkdownToHtml(rawTip);

        if (url) {
          htmlContent += '<div class="popover-footer-link"><a href="' + url + '" target="_blank" rel="noopener noreferrer">了解更多 <i class="fa fa-external-link"></i></a></div>';
        }

        el.addEventListener('mouseenter', function () {
          scheduleShow(el, htmlContent);
        });

        el.addEventListener('mouseleave', function () {
          scheduleHide();
        });

        el.addEventListener('click', function (e) {
          var href = el.getAttribute('href');
          if (!href || href === '#' || href === 'javascript:void(0)') {
            e.preventDefault();
          }
          if (currentTrigger === el && popover.classList.contains('is-active')) {
            hidePopover();
          } else {
            showPopover(el, htmlContent);
          }
        });
      }
    }
  }

  window.initInlinePopups = initInlinePopups;

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initInlinePopups);
  } else {
    initInlinePopups();
  }
})();
