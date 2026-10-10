(function () {
    'use strict';

    var meta = document.querySelector('.post-container .post-length, .post-length');
    var article = meta && document.querySelector('.post-container');
    if (!article || article.classList.contains('moments')) return;

    var key = 'readpos:' + location.pathname;
    var endSelector = '.post-license, .pager, .backlinks, .related-posts, .comment';
    var start = 0, end = 0, bar, pill, frame = 0, saveTimer = 0, savedAt = 0, pillTimer = 0, pillStartY = 0;
    function y() { return window.pageYOffset || document.documentElement.scrollTop || 0; }
    function absTop(el) { return el.getBoundingClientRect().top + y(); }
    function measure() {
        var marker = article.querySelector(endSelector);
        start = absTop(article);
        end = marker ? absTop(marker) : start + article.offsetHeight;
    }
    function fraction() {
        var span = end - start;
        return span > 0 ? Math.max(0, Math.min(1, (y() + innerHeight * .3 - start) / span)) : 0;
    }
    function update() {
        frame = 0; measure();
        var p = fraction();
        bar.style.transform = 'scaleX(' + p + ')';
        bar.classList.toggle('is-visible', p > 0);
    }
    function queueUpdate() { if (!frame) frame = requestAnimationFrame(update); }
    function position() {
        var headings = article.querySelectorAll('h1,h2,h3,h4'), threshold = y() + 90, found = null, ht = -Infinity;
        for (var i = 0; i < headings.length; i++) {
            var t = absTop(headings[i]);
            if (t < end && t <= threshold && t > ht) { found = headings[i]; ht = t; }
        }
        return found ? { id: found.id || null, dy: y() - ht } : { id: null, dy: y() };
    }
    function prune() {
        try {
            var now = Date.now(), keys = [];
            for (var i = localStorage.length - 1; i >= 0; i--) {
                var k = localStorage.key(i);
                if (!k || k.indexOf('readpos:') !== 0) continue;
                var v;
                try { v = JSON.parse(localStorage.getItem(k)); } catch (e) { v = null; }
                if (!v || !v.t || now - v.t > 90 * 86400000) localStorage.removeItem(k);
                else keys.push({ key: k, t: v.t });
            }
            keys.sort(function (a, b) { return b.t - a.t; });
            for (var j = 200; j < keys.length; j++) localStorage.removeItem(keys[j].key);
        } catch (e) { /* storage may be unavailable */ }
    }
    function save() {
        saveTimer = 0; savedAt = Date.now();
        var p = fraction();
        try {
            if (p > .95) { localStorage.removeItem(key); prune(); return; }
            if (p < .05) return;
            var pos = position();
            localStorage.setItem(key, JSON.stringify({ id: pos.id, dy: pos.dy, p: p, t: Date.now() }));
            prune();
        } catch (e) { /* storage may be unavailable */ }
    }
    function queueSave() {
        if (!saveTimer) saveTimer = setTimeout(save, Math.max(0, 1000 - (Date.now() - savedAt)));
    }
    function cleanText(el) {
        var clone = el.cloneNode(true), junk = clone.querySelectorAll('.heading-anchor,.sec-react,.annotation-marker');
        for (var i = 0; i < junk.length; i++) junk[i].remove();
        return clone.textContent.replace(/\s+/g, ' ').trim();
    }
    function dismiss() {
        clearTimeout(pillTimer); pillTimer = 0;
        if (pill) pill.remove();
        pill = null;
    }
    function restore() {
        if (location.hash || y() >= 200) return;
        try {
            var value = JSON.parse(localStorage.getItem(key));
            if (!value || value.p < .05 || value.p > .95 || !value.t || Date.now() - value.t > 90 * 86400000) return;
            var heading = value.id && document.getElementById(value.id);
            pill = document.createElement('div'); pill.className = 'reading-position-prompt';
            var label = document.createElement('span');
            label.textContent = heading ? '上次读到「' + cleanText(heading) + '」 · ' : '上次读到 ' + Math.round(value.p * 100) + '% · ';
            var go = document.createElement('button'); go.type = 'button'; go.className = 'reading-position-resume'; go.textContent = '继续阅读';
            go.addEventListener('click', function () {
                if (heading && window.SectionFold) window.SectionFold.reveal(heading);
                window.scrollTo({ top: Math.max(0, (heading ? absTop(heading) : 0) + (+value.dy || 0)), behavior: 'smooth' });
                dismiss();
            });
            var close = document.createElement('button'); close.type = 'button'; close.className = 'reading-position-close';
            close.setAttribute('aria-label', '关闭'); close.textContent = '×'; close.addEventListener('click', dismiss);
            pill.appendChild(label); pill.appendChild(go); pill.appendChild(close); document.body.appendChild(pill);
            pillStartY = y();
            pillTimer = setTimeout(dismiss, 12000);
        } catch (e) { /* storage may be unavailable */ }
    }
    function init() {
        bar = document.createElement('div'); bar.className = 'reading-progress'; bar.setAttribute('aria-hidden', 'true');
        document.body.appendChild(bar); measure(); update(); restore();
        window.addEventListener('scroll', function () {
            queueUpdate(); queueSave();
            if (pill && Math.abs(y() - pillStartY) > 600) dismiss();
        }, { passive: true });
        window.addEventListener('resize', queueUpdate);
        window.addEventListener('pagehide', save);
        document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'hidden') save(); });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
