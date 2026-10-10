(function () {
    'use strict';
    function init() {
    var article = document.querySelector('.post-container');
    if (!article || article.classList.contains('moments')) return;
    var excluded = '.pager, .related-posts, .share, .comment, .markdown-toc, .backlinks';
    var direct = Array.prototype.slice.call(article.children);
    var headings = direct.filter(function (el) { return /^H[1-6]$/.test(el.tagName) && !el.closest(excluded); });
    if (!headings.length) return;
    var topLevel = Math.min.apply(null, headings.map(function (h) { return +h.tagName.slice(1); }));
    var sections = [];
    function sectionNodes(heading) {
        var level = +heading.tagName.slice(1), nodes = [], node = heading.nextElementSibling;
        while (node) {
            if (node.matches(excluded)) break;
            if (/^H[1-6]$/.test(node.tagName) && +node.tagName.slice(1) <= level) break;
            nodes.push(node); node = node.nextElementSibling;
        }
        return nodes;
    }
    function sectionFor(heading) {
        return { heading: heading, level: +heading.tagName.slice(1) };
    }
    function setFold(section, folded) {
        section.button.setAttribute('aria-expanded', String(!folded));
        section.button.setAttribute('aria-label', folded ? '展开本节' : '折叠本节');
        sectionNodes(section.heading).forEach(function (node) {
            if (folded) {
                if (node.hasAttribute('hidden')) return;
                node.setAttribute('hidden', 'until-found');
                node.setAttribute('data-fold-owner', section.heading.id);
            } else if (node.getAttribute('data-fold-owner') === section.heading.id) {
                node.removeAttribute('hidden');
                node.removeAttribute('data-fold-owner');
            }
        });
    }
    headings.forEach(function (heading) {
        var level = +heading.tagName.slice(1);
        if (level !== topLevel && level !== topLevel + 1) return;
        if (!heading.id) return;
        var button = document.createElement('button');
        button.type = 'button'; button.className = 'heading-fold';
        button.setAttribute('aria-expanded', 'true'); button.setAttribute('aria-label', '折叠本节');
        heading.insertBefore(button, heading.firstChild);
        var section = sectionFor(heading); section.button = button; sections.push(section);
        button.addEventListener('click', function (event) {
            var fold = button.getAttribute('aria-expanded') === 'true';
            if (event.altKey) {
                sections.filter(function (other) { return other.level === level; }).forEach(function (other) { setFold(other, fold); });
            } else setFold(section, fold);
        });
    });
    function expandAll() { sections.forEach(function (section) { setFold(section, false); }); }
    function reveal(el) {
        var owners = [], node = el;
        while (node && article.contains(node)) {
            if (node.getAttribute && node.hasAttribute('data-fold-owner')) owners.push(node.getAttribute('data-fold-owner'));
            node = node.parentElement;
        }
        owners.reverse().forEach(function (id) {
            var section = sections.find(function (entry) { return entry.heading.id === id; });
            if (section) setFold(section, false);
        });
    }
    window.SectionFold = { reveal: reveal, expandAll: expandAll };
    article.addEventListener('beforematch', function (event) { reveal(event.target); }, true);
    window.addEventListener('hashchange', function () {
        var target;
        try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (e) { target = null; }
        if (target) reveal(target);
    });
    var printFolds = [];
    window.addEventListener('beforeprint', function () {
        printFolds = sections.filter(function (section) { return section.button.getAttribute('aria-expanded') === 'false'; });
        expandAll();
    });
    window.addEventListener('afterprint', function () { printFolds.slice().reverse().forEach(function (section) { setFold(section, true); }); printFolds = []; });
    if (location.hash) {
        var target;
        try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (e) { target = null; }
        if (target) reveal(target);
    }
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
