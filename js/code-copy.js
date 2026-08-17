/*!
 * code-copy.js
 * Github-style "copy" button on every code block and Mermaid diagram.
 * Mermaid containers appear asynchronously (the diagram replaces the fenced
 * block after the CDN script loads), so they are picked up by an observer.
 */
(function () {
    'use strict';

    var COPY_ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">' +
        '<path fill="currentColor" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>' +
        '<path fill="currentColor" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path></svg>';

    var DONE_ICON = '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">' +
        '<path fill="currentColor" d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path></svg>';

    function copy(text) {
        // The async API needs a secure context *and* a focused document, otherwise
        // it can reject (or hang), so fall back to the old textarea trick.
        if (navigator.clipboard && navigator.clipboard.writeText && window.isSecureContext && document.hasFocus()) {
            return navigator.clipboard.writeText(text).catch(function () { return legacyCopy(text); });
        }
        return legacyCopy(text);
    }

    function legacyCopy(text) {
        var area = document.createElement('textarea');
        area.value = text;
        area.setAttribute('readonly', '');
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        var ok = document.execCommand('copy');
        document.body.removeChild(area);
        return ok ? Promise.resolve() : Promise.reject(new Error('copy failed'));
    }

    function addButton(anchor, getText) {
        if (anchor.querySelector(':scope > .code-copy')) return;

        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'code-copy';
        button.title = '复制';
        button.setAttribute('aria-label', '复制');
        button.innerHTML = COPY_ICON;

        button.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            copy(getText()).then(function () {
                button.classList.add('copied');
                button.innerHTML = DONE_ICON;
                button.title = '已复制';
                window.setTimeout(function () {
                    button.classList.remove('copied');
                    button.innerHTML = COPY_ICON;
                    button.title = '复制';
                }, 1600);
            }).catch(function () {
                button.title = '复制失败';
            });
        });

        anchor.classList.add('code-copy-anchor');
        anchor.appendChild(button);
    }

    function decorateCodeBlocks(root) {
        var blocks = root.querySelectorAll('pre');
        Array.prototype.forEach.call(blocks, function (pre) {
            if (pre.closest('.mermaid')) return;
            var code = pre.querySelector('code') || pre;
            // Anchor on the rouge wrapper when there is one, so the button sits
            // outside the scrollable <pre>.
            var anchor = pre.closest('.highlighter-rouge') || pre;
            addButton(anchor, function () { return code.innerText.replace(/\n$/, ''); });
        });
    }

    function decorateDiagrams(root) {
        var diagrams = root.querySelectorAll('.mermaid[data-mermaid-source]');
        Array.prototype.forEach.call(diagrams, function (diagram) {
            // Diagrams that failed to render keep their source as text; a copy
            // button is still useful there.
            if (!diagram.querySelector('svg') && !diagram.classList.contains('mermaid-error')) return;
            addButton(diagram, function () { return diagram.getAttribute('data-mermaid-source'); });
        });
    }

    function init() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        decorateCodeBlocks(container);
        decorateDiagrams(container);

        // Diagrams render one after another, well after DOMContentLoaded.
        if (window.MutationObserver) {
            var observer = new MutationObserver(function () { decorateDiagrams(container); });
            observer.observe(container, { childList: true, subtree: true });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
