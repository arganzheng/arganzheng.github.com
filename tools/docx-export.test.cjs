'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const DocxExport = require('../js/docx-export.js');

test('escapes XML text and removes XML 1.0 control characters', () => {
  assert.equal(DocxExport.xmlEscape(`A&B <x> "'\u0001`), 'A&amp;B &lt;x&gt; &quot;&apos;');
});

test('scales oversized images to 6.27 inches without changing their aspect ratio', () => {
  const extent = DocxExport.imageExtent(1200, 600);
  assert.equal(extent.cx, DocxExport.maxImageWidthEmu);
  assert.equal(extent.cy, Math.round(DocxExport.maxImageWidthEmu / 2));

  const small = DocxExport.imageExtent(200, 100);
  assert.equal(small.cx, 200 * 9525);
  assert.equal(small.cy, 100 * 9525);
});

test('gives every ordered list its own restart override', () => {
  const xml = DocxExport.numberingXml([
    { numId: 2, start: 3 },
    { numId: 3, start: 1 },
    { numId: 4, start: 5, ilvl: 2 }
  ]);

  assert.equal((xml.match(/<w:abstractNum /g) || []).length, 2);
  assert.match(xml, /<w:num w:numId="2"><w:abstractNumId w:val="1"\/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="3"\/>/);
  assert.match(xml, /<w:num w:numId="3"><w:abstractNumId w:val="1"\/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"\/>/);
  assert.match(xml, /<w:num w:numId="4"><w:abstractNumId w:val="1"\/><w:lvlOverride w:ilvl="2"><w:startOverride w:val="5"\/>/);
  assert.equal((xml.match(/<w:lvl w:ilvl="/g) || []).length, 18);
});
