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
        // js/figures.js may have moved our button into its corner strip (.fig-tools)
        if (anchor.querySelector(':scope > .code-copy, :scope > .fig-tools > .code-copy, :scope > .fig-media > .fig-tools > .code-copy')) return;

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

    function tableCellText(cell) {
        return (cell.innerText || cell.textContent || '').replace(/\s+/g, ' ').trim();
    }

    function tableRows(table) {
        return Array.prototype.map.call(table.querySelectorAll('tr'), function (row) {
            return Array.prototype.map.call(row.children, tableCellText);
        }).filter(function (row) { return row.length; });
    }

    function escapeMarkdownCell(value) {
        return String(value || '').replace(/\|/g, '\\|').replace(/\r?\n/g, '<br>');
    }

    function tableMarkdown(table) {
        var rows = tableRows(table);
        if (!rows.length) return '';
        var head = rows[0], width = head.length;
        var out = ['| ' + head.map(escapeMarkdownCell).join(' | ') + ' |',
            '| ' + head.map(function () { return '---'; }).join(' | ') + ' |'];
        rows.slice(1).forEach(function (row) {
            while (row.length < width) row.push('');
            out.push('| ' + row.slice(0, width).map(escapeMarkdownCell).join(' | ') + ' |');
        });
        return out.join('\n');
    }

    function tableTsv(table) {
        return tableRows(table).map(function (row) { return row.join('\t'); }).join('\n');
    }

    function tableHtml(table) {
        var clone = table.cloneNode(true);
        Array.prototype.forEach.call(clone.querySelectorAll('.table-tools, .table-copy, .table-feedback'), function (el) {
            if (el.parentNode) el.parentNode.removeChild(el);
        });
        clone.removeAttribute('class');
        clone.removeAttribute('style');
        return clone.outerHTML;
    }

    function tableCopy(table, format) {
        if (format === 'tsv') return tableTsv(table);
        if (format === 'markdown') return tableMarkdown(table);
        return tableHtml(table);
    }

    function tableCopyMenu(anchor, table, button) {
        var menu = document.createElement('div');
        menu.className = 'table-copy-menu';
        menu.hidden = true;
        menu.setAttribute('role', 'menu');
        menu.innerHTML = '<button type="button" data-format="tsv">TSV（表格软件）</button>' +
            '<button type="button" data-format="markdown">Markdown</button>' +
            '<button type="button" data-format="html">HTML（富文本）</button>';
        anchor.appendChild(menu);

        function close() { menu.hidden = true; button.setAttribute('aria-expanded', 'false'); }
        button.addEventListener('click', function (e) {
            e.preventDefault(); e.stopPropagation();
            var open = menu.hidden;
            document.querySelectorAll('.table-copy-menu:not([hidden])').forEach(function (m) { m.hidden = true; });
            menu.hidden = !open;
            button.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        menu.addEventListener('click', function (e) {
            var item = e.target.closest('[data-format]');
            if (!item) return;
            e.preventDefault(); e.stopPropagation();
            copy(tableCopy(table, item.getAttribute('data-format'))).then(function () {
                button.classList.add('copied');
                button.title = '已复制 ' + item.textContent.trim();
                window.setTimeout(function () { button.classList.remove('copied'); button.title = '复制表格'; }, 1600);
            }).catch(function () { button.title = '复制失败'; });
            close();
        });
        document.addEventListener('click', function (e) { if (!anchor.contains(e.target)) close(); });
        document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
    }

    function addTableButton(table) {
        if (table.closest('.comment, .annotation-panel, .series-toc, .related-posts')) return;
        var anchor = table.parentNode.classList.contains('table-responsive') ? table.parentNode : table;
        anchor.classList.add('table-tools-anchor');
        if (anchor.querySelector(':scope > .table-tools')) return;
        var tools = document.createElement('div');
        tools.className = 'table-tools';
        tools.addEventListener('click', function (e) { e.stopPropagation(); });
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'code-copy table-copy';
        button.title = '复制表格';
        button.setAttribute('aria-label', '复制表格');
        button.setAttribute('aria-haspopup', 'menu');
        button.setAttribute('aria-expanded', 'false');
        button.innerHTML = COPY_ICON;
        tools.appendChild(button);
        anchor.appendChild(tools);
        tableCopyMenu(anchor, table, button);
    }

    function decorateTables(root) {
        Array.prototype.forEach.call(root.querySelectorAll('table'), addTableButton);
    }

    function init() {
        var container = document.querySelector('.post-container');
        if (!container) return;

        decorateCodeBlocks(container);
        decorateDiagrams(container);
        decorateTables(container);

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
