/*!
 * code-tabs.js — one code block, several languages, switched by tabs.
 *
 * Markdown (kramdown keeps highlighting the fences inside a markdown="1" div):
 *
 *   <div class="code-tabs" markdown="1">
 *   ```python
 *   ...
 *   ```
 *   ```java
 *   ...
 *   ```
 *   </div>
 *
 * Every direct `.highlighter-rouge` child becomes a panel (label from its
 * `language-xxx` class); other children stay visible above the panels. The
 * choice is page-wide and remembered (localStorage "code-tab-lang"): picking
 * Java once switches every group on the page and the next post opens on Java.
 * Groups without the preferred language show their first panel. Without JS the
 * CSS stacks the panels and labels them (see less/extras.less).
 */
(function () {
    'use strict';

    var KEY = 'code-tab-lang';
    var LABELS = {
        python: 'Python', py: 'Python', java: 'Java', cpp: 'C++', 'c++': 'C++', c: 'C',
        javascript: 'JavaScript', js: 'JavaScript', typescript: 'TypeScript', go: 'Go', rust: 'Rust',
        bash: 'Shell', sh: 'Shell', shell: 'Shell', text: '文本', plaintext: '文本', yaml: 'YAML', json: 'JSON', sql: 'SQL'
    };

    function langOf(panel) {
        var m = /(?:^|\s)language-([\w+#-]+)/.exec(panel.className);
        return m ? m[1].toLowerCase() : 'text';
    }

    function readPref() {
        try { return window.localStorage.getItem(KEY) || ''; } catch (e) { return ''; }
    }

    function savePref(lang) {
        try { window.localStorage.setItem(KEY, lang); } catch (e) { /* private mode */ }
    }

    var groups = [];

    function build(root) {
        var panels = Array.prototype.filter.call(root.children, function (el) {
            return el.classList.contains('highlighter-rouge');
        });
        if (panels.length < 2) return null;

        var bar = document.createElement('div');
        bar.className = 'code-tabs-bar';
        bar.setAttribute('role', 'tablist');

        var tabs = panels.map(function (panel) {
            var lang = langOf(panel);
            panel.setAttribute('data-lang', lang);
            panel.setAttribute('role', 'tabpanel');
            var button = document.createElement('button');
            button.type = 'button';
            button.className = 'code-tab';
            button.setAttribute('role', 'tab');
            button.setAttribute('data-lang', lang);
            button.textContent = LABELS[lang] || lang;
            button.addEventListener('click', function () {
                savePref(lang);
                groups.forEach(function (g) { g.show(lang); });
            });
            bar.appendChild(button);
            return { lang: lang, panel: panel, button: button };
        });

        root.insertBefore(bar, panels[0]);
        root.classList.add('code-tabs-ready');

        var group = {
            show: function (lang) {
                var hit = tabs.some(function (t) { return t.lang === lang; });
                tabs.forEach(function (t, i) {
                    var on = hit ? t.lang === lang : i === 0;
                    t.panel.classList.toggle('is-active', on);
                    t.panel.hidden = !on;
                    t.button.classList.toggle('is-active', on);
                    t.button.setAttribute('aria-selected', on ? 'true' : 'false');
                });
            }
        };
        return group;
    }

    function init() {
        var roots = document.querySelectorAll('.post-container .code-tabs:not(.code-tabs-ready)');
        Array.prototype.forEach.call(roots, function (root) {
            var g = build(root);
            if (g) groups.push(g);
        });
        var pref = readPref();
        groups.forEach(function (g) { g.show(pref); });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
