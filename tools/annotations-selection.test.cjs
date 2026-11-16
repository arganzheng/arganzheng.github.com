'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../js/annotations.js'), 'utf8');
const selector = source.match(/  var EXCLUDE_SELECTOR = [\s\S]*?;/)[0];
const functions = source.slice(source.indexOf('  function isExcluded('), source.indexOf('  function markHost('));
const selection = source.slice(source.indexOf('  function currentRange('), source.indexOf('  function updateToolbar('));

function element(tag, classes = [], parent = null) {
  return {
    nodeType: 1, parentNode: parent, parentElement: parent,
    classList: { contains: name => classes.includes(name) },
    matches(selectors) {
      return selectors.split(',').some(s => {
        s = s.trim();
        return s.startsWith('.') ? classes.includes(s.slice(1)) : s === tag;
      });
    },
    closest(selectors) {
      return this.matches(selectors) ? this : parent ? parent.closest(selectors) : null;
    }
  };
}

function fixture() {
  const container = element('article');
  container.contains = node => {
    for (; node; node = node.parentNode) if (node === container) return true;
    return false;
  };
  const box = element('div', ['callout'], container);
  const title = element('p', ['callout-title'], box);
  const icon = element('svg', ['callout-icon'], title);
  const body = element('p', [], box);
  const context = { container, window: {}, katexOf: () => null };
  vm.createContext(context);
  vm.runInContext(selector + functions + selection, context);
  return { context, container, title, icon, body };
}

const textNode = parent => ({ nodeType: 3, parentNode: parent });

test('callout title, nested title text and body are indexed', () => {
  const { context, title, body } = fixture();
  for (const parent of [title, element('strong', [], title), body]) {
    assert.equal(context.isExcluded(textNode(parent)), false);
    assert.equal(context.excludedAncestor(textNode(parent)), null);
  }
});

test('decorative icon and comment controls stay excluded', () => {
  const { context, container, icon } = fixture();
  for (const parent of [icon, element('button', [], container), element('span', ['annotation-marker'], container)]) {
    const node = textNode(parent);
    assert.equal(context.isExcluded(node), true);
    assert.equal(context.excludedAncestor(node), parent);
  }
});

for (const crossBody of [false, true]) {
  test(crossBody ? 'selection can cross from callout title into body' : 'title-only selection keeps its boundaries', () => {
    const { context, container, title, body } = fixture();
    const range = {
      startContainer: textNode(title), endContainer: textNode(crossBody ? body : title),
      commonAncestorContainer: container, collapsed: false,
      cloneRange() { return this; },
      toString() { return crossBody ? '与 Java 对照 正文' : '与 Java 对照'; },
      setStartAfter() { assert.fail('title start must not be excluded'); },
      setEndBefore() { assert.fail('title end must not be excluded'); }
    };
    context.window.getSelection = () => ({ isCollapsed: false, rangeCount: 1, getRangeAt: () => range });
    assert.equal(context.currentRange(), range);
  });
}
