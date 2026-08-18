/*!
 * code-tokens.js
 * Rouge tags an identifier the same way wherever it appears, while github.com
 * highlights with tree-sitter and knows about member access. That is the one
 * remaining difference from Github's rendering, so we recover it here for the
 * two languages this blog is full of (checked token by token against the way
 * github.com renders these very posts):
 *
 *   python  obj.attr / obj.method()  -> constant blue   (#0550ae)
 *   java    Obj.method()             -> entity purple   (#6639ba)
 *
 * This is a heuristic over rouge's token stream, not a parser: it only looks at
 * whether an identifier directly follows a "." and whether a "(" follows it.
 */
(function () {
    'use strict';

    function language(block) {
        var wrapper = block.closest('[class*="language-"]');
        var match = wrapper && wrapper.className.match(/language-([\w+-]+)/);
        return match ? match[1] : '';
    }

    function previousToken(node) {
        var prev = node.previousSibling;
        while (prev && prev.nodeType === 3 && !prev.nodeValue.trim()) prev = prev.previousSibling;
        return prev;
    }

    function nextText(node) {
        var next = node.nextSibling;
        while (next && next.nodeType === 3 && !next.nodeValue.trim()) next = next.nextSibling;
        if (!next) return '';
        return (next.nodeType === 3 ? next.nodeValue : next.textContent).trim();
    }

    function followsDot(span) {
        var prev = previousToken(span);
        if (!prev) return false;
        var text = prev.nodeType === 3 ? prev.nodeValue : prev.textContent;
        return /\.$/.test(text.replace(/\s+$/, ''));
    }

    /*
     * Rules per language, each verified against the colours github.com gives the
     * same snippets. `after: dot` = the token directly follows a ".",
     * `before: paren` = a "(" directly follows it.
     */
    var RULES = {
        python: [
            // attributes (`.n`) and dotted calls (`.nf`) are both blue on Github;
            // bare calls keep rouge's purple, which is what Github does as well
            { selector: 'span.n, span.nf', after: 'dot', className: 'tok-member' }
        ],
        java: [
            // Github leaves every type name plain, rouge paints them like variables
            { selector: 'span.nc', className: 'tok-plain' },
            // ... and paints anything called, dotted or not, as an entity
            { selector: 'span.n, span.na, span.nf', before: 'paren', className: 'tok-call' }
        ]
    };

    function decorate(block) {
        var rules = RULES[language(block)];
        if (!rules) return;
        rules.forEach(function (rule) {
            Array.prototype.forEach.call(block.querySelectorAll(rule.selector), function (span) {
                if (rule.after === 'dot' && !followsDot(span)) return;
                if (rule.before === 'paren' && nextText(span).charAt(0) !== '(') return;
                span.classList.add(rule.className);
            });
        });
    }

    function init() {
        var container = document.querySelector('.post-container');
        if (!container) return;
        Array.prototype.forEach.call(container.querySelectorAll('.highlight'), decorate);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
