/*
 * Internal post link previews. The preview index is fetched on the first
 * eligible hover and shares InlinePopover with footnotes and inline tips.
 */
(function () {
    'use strict';

    var container = document.querySelector('.post-container');
    if (!container || !window.matchMedia || !window.matchMedia('(hover: hover)').matches) return;

    var overlay = document.getElementById('search-overlay');
    var baseUrl = overlay ? overlay.getAttribute('data-search-base-url') : '/search';
    var basePath = overlay ? (overlay.getAttribute('data-site-baseurl') || '').replace(/\/+$/, '') : '';
    var siteHost = overlay ? overlay.getAttribute('data-site-host') : '';
    var version = overlay ? overlay.getAttribute('data-search-version') : '';
    var indexPromise = null;
    var activeLink = null;

    function decodePath(path) {
        try { return decodeURIComponent(path); } catch (e) { return path; }
    }

    function normalizedPath(path) {
        path = decodePath(path);
        if (basePath && (path === basePath || path.indexOf(basePath + '/') === 0)) {
            path = path.slice(basePath.length);
        }
        return path.charAt(0) === '/' ? path : '/' + path;
    }

    function candidateFrom(target) {
        if (!target || !target.closest) return null;
        var link = target.closest('a[href]');
        if (!link || !container.contains(link)) return null;
        if (link.classList.contains('heading-anchor') ||
                link.classList.contains('inline-tip') ||
                link.classList.contains('pa-item') ||
                link.classList.contains('footnote') ||
                link.classList.contains('reversefootnote') ||
                link.querySelector('img') ||
                link.closest('.post-actions, sup[id^="fnref"], [data-no-preview]')) return null;

        var destination;
        try { destination = new URL(link.href, window.location.href); } catch (e) { return null; }
        var hostname = destination.hostname.replace(/^www\./, '');
        var allowedHost = hostname === window.location.hostname.replace(/^www\./, '') ||
            (siteHost && hostname === siteHost.replace(/^www\./, ''));
        if (!allowedHost || normalizedPath(destination.pathname) === normalizedPath(window.location.pathname)) return null;
        return link;
    }

    function fetchIndex() {
        if (!indexPromise) {
            var url = (baseUrl || '/search').replace(/\/+$/, '') + '/preview.json';
            if (version) url += '?v=' + encodeURIComponent(version);
            indexPromise = fetch(url).then(function (response) {
                if (!response.ok) throw new Error('preview index request failed');
                return response.json();
            }).catch(function () { return {}; });
        }
        return indexPromise;
    }

    function escapeHtml(value) {
        return String(value == null ? '' : value).replace(/[&<>"']/g, function (char) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char];
        });
    }

    function previewHtml(link, entry) {
        var destination = new URL(link.href, window.location.href);
        var title = entry[0];
        var date = entry[1];
        var summary = entry[2];
        var series = entry[3];
        var html = '<div class="lp-title">' + escapeHtml(title) + '</div>' +
            '<div class="lp-meta">' + escapeHtml(date) +
            (series ? ' · ' + escapeHtml(series) : '') + '</div>' +
            '<div class="lp-summary">' + escapeHtml(summary) + '</div>';
        if (destination.hash) {
            var fragment = destination.hash.slice(1);
            try { fragment = decodeURIComponent(fragment); } catch (e) {}
            html += '<div class="lp-anchor">§ ' + escapeHtml(fragment) + '</div>';
        }
        return html;
    }

    function isStillActive(link) {
        return activeLink === link && (link.matches(':hover') || document.activeElement === link);
    }

    function showPreview(link) {
        activeLink = link;
        InlinePopover.scheduleHide();
        fetchIndex().then(function (index) {
            if (!isStillActive(link)) return;
            var path = normalizedPath(new URL(link.href, window.location.href).pathname);
            var entry = index[path];
            if (!entry) return;
            InlinePopover.scheduleShow(link, previewHtml(link, entry), {
                className: 'link-preview-popover',
                delay: 300
            });
        });
    }

    function leaveLink(link) {
        if (activeLink !== link) return;
        activeLink = null;
        InlinePopover.scheduleHide();
    }

    function onMouseOver(event) {
        var link = candidateFrom(event.target);
        if (!link || (event.relatedTarget && link.contains(event.relatedTarget))) return;
        showPreview(link);
    }

    function onMouseOut(event) {
        var link = candidateFrom(event.target);
        if (!link || (event.relatedTarget && link.contains(event.relatedTarget))) return;
        leaveLink(link);
    }

    function onFocusIn(event) {
        var link = candidateFrom(event.target);
        if (link) showPreview(link);
    }

    function onFocusOut(event) {
        var link = candidateFrom(event.target);
        if (!link || (event.relatedTarget && link.contains(event.relatedTarget))) return;
        leaveLink(link);
    }

    container.addEventListener('mouseover', onMouseOver);
    container.addEventListener('mouseout', onMouseOut);
    container.addEventListener('focusin', onFocusIn);
    container.addEventListener('focusout', onFocusOut);
})();
