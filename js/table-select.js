/*!
 * table-select.js
 * Spreadsheet-style mouse selection and copy for article table cells.
 */
(function () {
    'use strict';

    var EXCLUDED = '.comment, .annotation-panel, .series-toc, .related-posts';
    var container = document.querySelector('.post-container');
    if (!container) return;

    var tracking = false;
    var rectMode = false;
    var startTable = null;
    var startRecord = null;
    var grid = null;
    var activeSelection = null;

    function eligibleTable(table) {
        return table && container.contains(table) && !table.closest(EXCLUDED);
    }

    function targetCell(target) {
        var element = target && target.nodeType === 1 ? target : target && target.parentElement;
        return element && element.closest ? element.closest('td, th') : null;
    }

    function buildGrid(table) {
        var rows = table.rows, matrix = [], records = [], cells = new WeakMap();

        for (var r = 0; r < rows.length; r++) {
            matrix[r] = matrix[r] || [];
            var row = rows[r];
            for (var i = 0; i < row.cells.length; i++) {
                var cell = row.cells[i];
                var colSpan = Math.max(1, cell.colSpan || 1);
                var col = 0, conflict;
                do {
                    conflict = false;
                    for (var offset = 0; offset < colSpan; offset++) {
                        if (matrix[r][col + offset]) {
                            conflict = true;
                            col++;
                            break;
                        }
                    }
                } while (conflict);

                var rowSpan = cell.rowSpan === 0 ? rows.length - r : Math.max(1, cell.rowSpan || 1);
                var record = {
                    cell: cell,
                    r0: r,
                    c0: col,
                    r1: Math.min(rows.length - 1, r + rowSpan - 1),
                    c1: col + colSpan - 1
                };
                records.push(record);
                cells.set(cell, record);

                for (var rr = record.r0; rr <= record.r1; rr++) {
                    matrix[rr] = matrix[rr] || [];
                    for (var cc = record.c0; cc <= record.c1; cc++) matrix[rr][cc] = record;
                }
            }
        }
        return { rows: matrix, records: records, cells: cells };
    }

    function intersects(record, bounds) {
        return record.r0 <= bounds.r1 && record.r1 >= bounds.r0 &&
            record.c0 <= bounds.c1 && record.c1 >= bounds.c0;
    }

    function expandedBounds(start, end, records) {
        var bounds = {
            r0: Math.min(start.r0, end.r0),
            c0: Math.min(start.c0, end.c0),
            r1: Math.max(start.r1, end.r1),
            c1: Math.max(start.c1, end.c1)
        };
        var changed;
        do {
            changed = false;
            records.forEach(function (record) {
                if (!intersects(record, bounds)) return;
                var next = {
                    r0: Math.min(bounds.r0, record.r0),
                    c0: Math.min(bounds.c0, record.c0),
                    r1: Math.max(bounds.r1, record.r1),
                    c1: Math.max(bounds.c1, record.c1)
                };
                if (next.r0 !== bounds.r0 || next.c0 !== bounds.c0 ||
                    next.r1 !== bounds.r1 || next.c1 !== bounds.c1) {
                    bounds = next;
                    changed = true;
                }
            });
        } while (changed);
        return bounds;
    }

    function clearCellClasses() {
        document.querySelectorAll('.post-container td.ts-selected, .post-container th.ts-selected')
            .forEach(function (cell) { cell.classList.remove('ts-selected'); });
        document.querySelectorAll('.post-container table.ts-dragging')
            .forEach(function (table) { table.classList.remove('ts-dragging'); });
    }

    function stopTracking() {
        if (startTable) startTable.classList.remove('ts-dragging');
        tracking = false;
        rectMode = false;
        startTable = null;
        startRecord = null;
        grid = null;
    }

    function clearSelection() {
        clearCellClasses();
        stopTracking();
        activeSelection = null;
    }

    function tableCellText(cell) {
        return (cell.innerText || cell.textContent || '').replace(/\s+/g, ' ').trim();
    }

    function updateSelection(endRecord) {
        var bounds = expandedBounds(startRecord, endRecord, grid.records);
        grid.records.forEach(function (record) {
            record.cell.classList.toggle('ts-selected', intersects(record, bounds));
        });
        activeSelection = { table: startTable, grid: grid, bounds: bounds };
    }

    function selectedText(selection) {
        var lines = [];
        for (var r = selection.bounds.r0; r <= selection.bounds.r1; r++) {
            var row = [];
            for (var c = selection.bounds.c0; c <= selection.bounds.c1; c++) {
                var record = selection.grid.rows[r] && selection.grid.rows[r][c];
                row.push(record && record.r0 === r && record.c0 === c ? tableCellText(record.cell) : '');
            }
            lines.push(row.join('\t'));
        }
        return lines.join('\n');
    }

    function escapeMarkdownCell(value) {
        return String(value || '').replace(/\|/g, '\\|').replace(/\r?\n/g, '<br>');
    }

    function rowValues(selection, rowIndex) {
        var values = [];
        for (var c = selection.bounds.c0; c <= selection.bounds.c1; c++) {
            var record = selection.grid.rows[rowIndex] && selection.grid.rows[rowIndex][c];
            values.push(record && record.r0 === rowIndex && record.c0 === c ?
                tableCellText(record.cell) : '');
        }
        return values;
    }

    function isHeaderRow(selection, rowIndex) {
        var records = selection.grid.records.filter(function (record) {
            return record.r0 === rowIndex;
        });
        return records.length > 0 && records.every(function (record) {
            return record.cell.tagName === 'TH';
        });
    }

    function markdownRow(values) {
        return '| ' + values.map(escapeMarkdownCell).join(' | ') + ' |';
    }

    function selectedMarkdown(selection) {
        var bounds = selection.bounds;
        var rows = [];
        var dataStart = bounds.r0 + 1;
        if (!isHeaderRow(selection, bounds.r0) && isHeaderRow(selection, 0)) {
            var header = rowValues(selection, 0);
            rows.push(markdownRow(header));
            rows.push(markdownRow(header.map(function () { return '---'; })));
            dataStart = bounds.r0;
        } else {
            var first = rowValues(selection, bounds.r0);
            rows.push(markdownRow(first));
            rows.push(markdownRow(first.map(function () { return '---'; })));
        }
        for (var r = dataStart; r <= bounds.r1; r++) {
            rows.push(markdownRow(rowValues(selection, r)));
        }
        return rows.join('\n');
    }

    function current() {
        if (tracking || !activeSelection) return null;
        var selection = activeSelection;
        var bounds = selection.bounds;
        var anchor = null;
        for (var i = 0; i < selection.grid.records.length; i++) {
            var record = selection.grid.records[i];
            if (record.r0 < bounds.r0 || record.r0 > bounds.r1 ||
                record.c0 < bounds.c0 || record.c0 > bounds.c1) continue;
            if (tableCellText(record.cell)) {
                anchor = record;
                break;
            }
        }
        if (!anchor) return null;
        var range = document.createRange();
        range.selectNodeContents(anchor.cell);
        var rect = null;
        selection.grid.records.forEach(function (record) {
            if (!record.cell.classList.contains('ts-selected')) return;
            var cellRect = record.cell.getBoundingClientRect();
            if (!rect) {
                rect = {
                    top: cellRect.top,
                    left: cellRect.left,
                    bottom: cellRect.bottom,
                    right: cellRect.right
                };
            } else {
                rect.top = Math.min(rect.top, cellRect.top);
                rect.left = Math.min(rect.left, cellRect.left);
                rect.bottom = Math.max(rect.bottom, cellRect.bottom);
                rect.right = Math.max(rect.right, cellRect.right);
            }
        });
        if (!rect) return null;
        rect.width = rect.right - rect.left;
        rect.height = rect.bottom - rect.top;
        return {
            range: range,
            rect: rect,
            text: selectedText(selection),
            markdown: selectedMarkdown(selection),
            cells: {
                rows: bounds.r1 - bounds.r0 + 1,
                cols: bounds.c1 - bounds.c0 + 1,
                dr: anchor.r0 - bounds.r0,
                dc: anchor.c0 - bounds.c0
            }
        };
    }

    function cellsFrom(cell, cells) {
        if (!cells) return [];
        var table = cell && cell.closest('table');
        if (!eligibleTable(table)) return [];
        var selectionGrid = buildGrid(table);
        var anchor = selectionGrid.cells.get(cell);
        if (!anchor) return [];
        var rowCount = table.rows.length;
        var colCount = selectionGrid.rows.reduce(function (max, row) {
            return Math.max(max, row.length);
        }, 0);
        var rowSpan = Math.floor(Number(cells && cells.rows));
        var colSpan = Math.floor(Number(cells && cells.cols));
        if (rowCount < 1 || colCount < 1 || !isFinite(rowSpan) || !isFinite(colSpan) ||
            rowSpan < 1 || colSpan < 1) return [];
        var dr = Math.floor(Number(cells.dr) || 0);
        var dc = Math.floor(Number(cells.dc) || 0);
        var r0 = Math.max(0, Math.min(rowCount - 1, anchor.r0 - dr));
        var c0 = Math.max(0, Math.min(colCount - 1, anchor.c0 - dc));
        var r1 = Math.max(r0, Math.min(rowCount - 1, r0 + rowSpan - 1));
        var c1 = Math.max(c0, Math.min(colCount - 1, c0 + colSpan - 1));
        var bounds = expandedBounds(
            { r0: r0, c0: c0, r1: r0, c1: c0 },
            { r0: r1, c0: c1, r1: r1, c1: c1 },
            selectionGrid.records
        );
        return selectionGrid.records.filter(function (record) {
            return intersects(record, bounds);
        }).map(function (record) {
            return {
                cell: record.cell,
                top: record.r0 === bounds.r0,
                bottom: record.r1 === bounds.r1,
                left: record.c0 === bounds.c0,
                right: record.c1 === bounds.c1
            };
        });
    }

    function selectedHtml(selection) {
        var table = document.createElement('table');
        var tbody = document.createElement('tbody');
        table.appendChild(tbody);
        for (var r = selection.bounds.r0; r <= selection.bounds.r1; r++) {
            var row = document.createElement('tr');
            selection.grid.records.forEach(function (record) {
                if (record.r0 !== r || record.c0 < selection.bounds.c0 ||
                    record.c0 > selection.bounds.c1) return;
                var cell = document.createElement('td');
                cell.innerHTML = record.cell.innerHTML;
                row.appendChild(cell);
            });
            tbody.appendChild(row);
        }
        return table.outerHTML;
    }

    document.addEventListener('mousedown', function (event) {
        var target = event.target && (event.target.nodeType === 1 ? event.target : event.target.parentElement);
        if (target && target.closest && target.closest('.annotation-toolbar, .pa-share-pop')) return;
        clearSelection();
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return;
        var cell = targetCell(event.target);
        var table = cell && cell.closest('table');
        if (!eligibleTable(table)) return;

        grid = buildGrid(table);
        startRecord = grid.cells.get(cell);
        if (!startRecord) {
            grid = null;
            return;
        }
        startTable = table;
        tracking = true;
    }, true);

    document.addEventListener('mousemove', function (event) {
        if (!tracking || !(event.buttons & 1)) return;
        var cell = targetCell(event.target);
        if (!cell || cell.closest('table') !== startTable) return;
        var endRecord = grid.cells.get(cell);
        if (!endRecord) return;
        if (!rectMode && endRecord.cell === startRecord.cell) return;
        if (!rectMode) {
            rectMode = true;
            startTable.classList.add('ts-dragging');
            var selection = window.getSelection();
            if (selection) selection.removeAllRanges();
        }
        updateSelection(endRecord);
    }, true);

    document.addEventListener('selectstart', function (event) {
        if (rectMode) event.preventDefault();
    }, true);

    document.addEventListener('mouseup', function () {
        if (!tracking) return;
        if (rectMode && activeSelection && activeSelection.bounds.r0 === activeSelection.bounds.r1) {
            var selection = activeSelection;
            var bounds = selection.bounds;
            if (startTable) startTable.classList.remove('ts-dragging');
            selection.grid.records.forEach(function (record) {
                record.cell.classList.remove('ts-selected');
            });
            activeSelection = null;
            var rowRecords = selection.grid.records.filter(function (record) {
                return record.r0 === bounds.r0 && record.c0 >= bounds.c0 && record.c0 <= bounds.c1;
            });
            if (rowRecords.length) {
                var range = document.createRange();
                range.setStart(rowRecords[0].cell, 0);
                var lastRecord = rowRecords[rowRecords.length - 1];
                range.setEnd(lastRecord.cell, lastRecord.cell.childNodes.length);
                var nativeSelection = window.getSelection();
                if (nativeSelection) {
                    nativeSelection.removeAllRanges();
                    nativeSelection.addRange(range);
                }
            }
        } else if (startTable) {
            startTable.classList.remove('ts-dragging');
        }
        stopTracking();
    }, true);

    document.addEventListener('dragstart', function () {
        if (tracking) clearSelection();
    }, true);

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && (activeSelection || (tracking && rectMode))) clearSelection();
    });

    document.addEventListener('copy', function (event) {
        if (!activeSelection || !event.clipboardData) return;
        var selection = window.getSelection();
        if (selection && !selection.isCollapsed) return;
        event.preventDefault();
        event.clipboardData.setData('text/plain', selectedText(activeSelection));
        event.clipboardData.setData('text/html', selectedHtml(activeSelection));
    });

    window.BlogTableSelect = { current: current, clear: clearSelection, cellsFrom: cellsFrom };

    if (window.MutationObserver) {
        var observer = new MutationObserver(function () {
            var table = activeSelection ? activeSelection.table : (tracking ? startTable : null);
            if (table && !document.documentElement.contains(table)) clearSelection();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
