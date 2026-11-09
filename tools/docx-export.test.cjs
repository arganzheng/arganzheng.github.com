'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const DocxExport = require('../js/docx-export.js');

function tableFixture(rows) {
  const grid = DocxExport.tableGrid(rows);
  const xml = DocxExport.tableRowsXml(grid, [], 1000);
  return { grid, xml, rows: xml.match(/<w:tr>[\s\S]*?<\/w:tr>/g) || [] };
}

function cells(row) {
  return row.match(/<w:tc>[\s\S]*?<\/w:tc>/g) || [];
}

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

test('keeps cells aligned after a row-spanned middle cell', () => {
  const result = tableFixture([
    [{}, { rowspan: 2 }, {}],
    [{}, {}],
    [{}, {}, {}]
  ]);
  const firstRow = cells(result.rows[0]);
  const secondRow = cells(result.rows[1]);

  assert.equal(result.grid.columnCount, 3);
  assert.equal(firstRow.length, 3);
  assert.match(firstRow[1], /<w:vMerge w:val="restart"\/>/);
  assert.equal(secondRow.length, 3);
  assert.match(secondRow[1], /<w:vMerge\/>/);
  assert.doesNotMatch(secondRow[0], /<w:vMerge/);
  assert.doesNotMatch(secondRow[2], /<w:vMerge/);
});

test('combines row and column spans on the same cell', () => {
  const result = tableFixture([
    [{ rowspan: 2, colspan: 2 }, {}],
    [{}]
  ]);
  const firstRow = cells(result.rows[0]);
  const secondRow = cells(result.rows[1]);

  assert.equal(result.grid.columnCount, 3);
  assert.equal(firstRow.length, 2);
  assert.match(firstRow[0], /<w:gridSpan w:val="2"\/><w:vMerge w:val="restart"\/>/);
  assert.equal(secondRow.length, 2);
  assert.match(secondRow[0], /<w:gridSpan w:val="2"\/><w:vMerge\/>/);
});

test('clamps row spans at the last table row', () => {
  const result = tableFixture([
    [{ rowspan: 8 }, {}],
    [{}, {}]
  ]);

  assert.equal(result.grid.rows.length, 2);
  assert.equal(result.grid.rows[0].cells[0].rowspan, 2);
  assert.equal((result.xml.match(/<w:vMerge w:val="restart"\/>/g) || []).length, 1);
  assert.equal((result.xml.match(/<w:vMerge\/>/g) || []).length, 1);
});
