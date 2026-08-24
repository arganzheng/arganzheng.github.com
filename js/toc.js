/*!
 * toc.js
 * Table of contents for posts, in two flavors sharing one heading scan:
 *
 *   1. inline  : replaces the `[TOC]` marker written by MWeb/Typora, which
 *                kramdown leaves as a plain paragraph.
 *   2. floating: fills `ul.catalog-body` in the sticky side column, and keeps
 *                the entry of the section being read highlighted (scroll spy).
 *
 * Heading ids come from kramdown (auto_ids); we only slug them ourselves when
 * an id is missing.
 */
(function () {
    'use strict';

    var TOC_MARKER = /^\[{1,2}\s*toc\s*\]{1,2}$/i;
    var HEADINGS = 'h1, h2, h3, h4, h5, h6';
    // Post footer areas living inside `.post-container` must stay out of the TOC.
    var EXCLUDED = '.pager, .related-posts, .share, .comment, .markdown-toc';
    // Height of the fixed navbar, used both as scroll offset and spy threshold.
    var NAV_OFFSET = 80;
    var COLLAPSED_CLASS = 'outline-collapsed';
    // Keep the catalog readable on long posts: show at most 3 outline levels.
    var MAX_OUTLINE_DEPTH = 3;

    // Github-flavored anchor: lowercased, punctuation dropped, spaces dashed.
    function slugify(text) {
        return text.trim().toLowerCase()
            .replace(/[\s\u3000]+/g, '-')
            .replace(/[!-\/:-@\[-`{-~\u2000-\u206f\u3001-\u303f\uff01-\uff65]/g, '')
            .replace(/^-+|-+$/g, '') || 'section';
    }

    function uniqueId(base, taken) {
        var id = base, i = 1;
        while (taken[id] || document.getElementById(id)) {
            id = base + '-' + i++;
        }
        taken[id] = true;
        return id;
    }

    function findMarkers(container) {
        var markers = [];
        var paragraphs = container.querySelectorAll('p');
        for (var i = 0; i < paragraphs.length; i++) {
            if (TOC_MARKER.test(paragraphs[i].textContent)) markers.push(paragraphs[i]);
        }
        return markers;
    }

    function collectHeadings(container) {
        var nodes = container.querySelectorAll(HEADINGS);
        var headings = [], taken = {};
        for (var i = 0; i < nodes.length; i++) {
            var node = nodes[i];
            if (node.closest && node.closest(EXCLUDED)) continue;
            var text = node.textContent.replace(/\s+/g, ' ').trim();
            if (!text) continue;
            if (!node.id) node.id = uniqueId(slugify(text), taken);
            headings.push({ level: parseInt(node.tagName.charAt(1), 10), text: text, node: node });
        }
        return headings;
    }

    function clampOutlineDepth(headings) {
        if (!headings.length) return headings;
        var topLevel = Math.min.apply(null, headings.map(function (h) { return h.level; }));
        var maxLevel = topLevel + MAX_OUTLINE_DEPTH - 1;
        return headings.filter(function (h) { return h.level <= maxLevel; });
    }

    function scrollToHeading(node) {
        var top = node.getBoundingClientRect().top + window.pageYOffset - NAV_OFFSET;
        window.scrollTo({ top: Math.max(top, 0), behavior: 'smooth' });
    }

    function createLink(heading, entries) {
        var link = document.createElement('a');
        link.href = '#' + heading.node.id;
        link.textContent = heading.text;
        link.title = heading.text;
        link.addEventListener('click', function (e) {
            if (e.metaKey || e.ctrlKey || e.shiftKey) return;
            e.preventDefault();
            scrollToHeading(heading.node);
            if (history.replaceState) history.replaceState(null, '', '#' + heading.node.id);
        });
        if (entries) entries.push({ link: link, node: heading.node });
        return link;
    }

    /*
     * Build a nested list of headings into `root` (a <ul>).
     * `entries` collects link/heading pairs for the scroll spy.
     */
    function buildList(root, headings, entries) {
        var topLevel = Math.min.apply(null, headings.map(function (h) { return h.level; }));
        // Each frame holds the list currently collecting items of `level`.
        var stack = [{ level: topLevel, list: root }];

        headings.forEach(function (heading) {
            while (stack.length > 1 && heading.level < stack[stack.length - 1].level) stack.pop();

            if (heading.level > stack[stack.length - 1].level) {
                var parent = stack[stack.length - 1].list;
                var nested = document.createElement('ul');
                // A skipped level (e.g. h2 -> h4) nests under the previous item.
                (parent.lastChild || parent).appendChild(nested);
                stack.push({ level: heading.level, list: nested });
            }

            var item = document.createElement('li');
            item.className = 'h' + heading.level + '_nav';
            item.appendChild(createLink(heading, entries));
            stack[stack.length - 1].list.appendChild(item);
        });

        return root;
    }

    function makeToggle(target, foldedClass) {
        var toggle = document.createElement('a');
        toggle.className = 'markdown-toc-toggle';
        toggle.href = '#';
        toggle.textContent = '隐藏';
        toggle.addEventListener('click', function (e) {
            e.preventDefault();
            var folded = target.classList.toggle(foldedClass);
            toggle.textContent = folded ? '显示' : '隐藏';
        });
        return toggle;
    }

    function buildInlineToc(headings) {
        var toc = document.createElement('div');
        toc.className = 'markdown-toc';

        var title = document.createElement('div');
        title.className = 'markdown-toc-title';
        title.textContent = '目录';
        title.appendChild(makeToggle(toc, 'fold'));

        toc.appendChild(title);
        toc.appendChild(buildList(document.createElement('ul'), headings, null));
        return toc;
    }

    /*
     * Highlight the entry of the heading currently being read and keep it
     * scrolled into view inside the (possibly overflowing) list `panel`.
     */
    function initScrollSpy(entries, panel) {
        var active = null;

        function activate(entry) {
            if (entry === active) return;
            if (active) {
                active.link.parentNode.classList.remove('active');
                ancestors(active.link).forEach(function (li) { li.classList.remove('active-parent'); });
            }
            active = entry;
            if (!active) return;
            active.link.parentNode.classList.add('active');
            ancestors(active.link).forEach(function (li) { li.classList.add('active-parent'); });
            keepVisible(active.link);
        }

        function ancestors(link) {
            var list = [], node = link.parentNode.parentNode;
            while (node && node !== panel) {
                if (node.tagName === 'LI') list.push(node);
                node = node.parentNode;
            }
            return list;
        }

        function keepVisible(link) {
            if (panel.scrollHeight <= panel.clientHeight) return;
            var top = link.getBoundingClientRect().top - panel.getBoundingClientRect().top + panel.scrollTop;
            if (top < panel.scrollTop || top > panel.scrollTop + panel.clientHeight - link.offsetHeight) {
                panel.scrollTop = top - panel.clientHeight / 3;
            }
        }

        function update() {
            var current = null;
            for (var i = 0; i < entries.length; i++) {
                if (entries[i].node.getBoundingClientRect().top - NAV_OFFSET - 10 > 0) break;
                current = entries[i];
            }
            // Near the bottom of the page the last section is the one being read.
            if (window.innerHeight + window.pageYOffset >= document.body.scrollHeight - 5) {
                current = entries[entries.length - 1];
            }
            activate(current);
        }

        var scheduled = false;
        function onScroll() {
            if (scheduled) return;
            scheduled = true;
            window.requestAnimationFrame(function () {
                scheduled = false;
                update();
            });
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll);
        update();
    }

    /*
     * Show/hide the whole outline. The flag lives on <body> because hiding the
     * outline has to give its grid column back to the article, not just empty
     * the column out.
     */
    function initCollapse(panel) {
        // The outline always starts open: collapsing is a per-page-view choice,
        // not a preference that should follow the reader around.
        function set(value) {
            document.body.classList.toggle(COLLAPSED_CLASS, value);
        }

        set(false);

        var close = panel.querySelector('.catalog-close');
        var open = document.querySelector('.outline-reopen');
        if (close) close.addEventListener('click', function () { set(true); });
        if (open) open.addEventListener('click', function () { set(false); });

        // Keep the legacy fold toggle working if a layout still ships it.
        var toggle = panel.querySelector('.catalog-toggle');
        if (toggle) {
            toggle.addEventListener('click', function (e) {
                e.preventDefault();
                panel.classList.toggle('fold');
            });
        }
    }

    /*
     * Live "Filter headings" box: an item stays visible when it matches itself
     * or when one of its children does, so the hierarchy is kept.
     */
    function initFilter(panel, body) {
        var input = panel.querySelector('.catalog-filter-input');
        var empty = panel.querySelector('.catalog-empty');
        if (!input) return;

        function matches(item, query) {
            var link = item.querySelector(':scope > a');
            var own = link && link.textContent.toLowerCase().indexOf(query) !== -1;
            var children = item.querySelectorAll(':scope > ul > li');
            var hit = false;
            Array.prototype.forEach.call(children, function (child) {
                if (matches(child, query)) hit = true;
            });
            var visible = own || hit;
            item.classList.toggle('filtered-out', !visible);
            return visible;
        }

        input.addEventListener('input', function () {
            var query = input.value.trim().toLowerCase();
            var items = body.querySelectorAll(':scope > li');
            var any = false;
            Array.prototype.forEach.call(items, function (item) {
                if (!query) {
                    item.classList.remove('filtered-out');
                    Array.prototype.forEach.call(item.querySelectorAll('li'), function (child) {
                        child.classList.remove('filtered-out');
                    });
                    any = true;
                    return;
                }
                if (matches(item, query)) any = true;
            });
            panel.classList.toggle('filtering', !!query);
            if (empty) empty.style.display = any ? 'none' : 'block';
        });

        input.addEventListener('keydown', function (e) {
            if (e.key !== 'Escape') return;
            input.value = '';
            input.dispatchEvent(new Event('input'));
        });
    }

    function renderFloatingToc(headings) {
        var panel = document.querySelector('.side-catalog');
        var body = panel && panel.querySelector('.catalog-body');
        if (!body || !headings.length) return;

        var entries = [];
        buildList(body, headings, entries);

        initCollapse(panel);
        initFilter(panel, body);
        initScrollSpy(entries, body);
    }

    function render() {
        var container = document.querySelector('.post-container') || document.querySelector('article');
        if (!container) return;

        var markers = findMarkers(container);
        var panel = document.querySelector('.side-catalog');
        if (!markers.length && !panel) return;

        var headings = clampOutlineDepth(collectHeadings(container));
        markers.forEach(function (marker) {
            if (!headings.length) {
                marker.parentNode.removeChild(marker);
                return;
            }
            marker.parentNode.replaceChild(buildInlineToc(headings), marker);
        });

        renderFloatingToc(headings);

        // Re-apply the anchor offset for links opened with a hash.
        if (window.location.hash) {
            var target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
            if (target) window.setTimeout(function () { scrollToHeading(target); }, 0);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', render);
    } else {
        render();
    }
})();
