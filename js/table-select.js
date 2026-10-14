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
        if (startTable) startTable.classList.remove('ts-dragging');
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

    if (window.MutationObserver) {
        var observer = new MutationObserver(function () {
            var table = activeSelection ? activeSelection.table : (tracking ? startTable : null);
            if (table && !document.documentElement.contains(table)) clearSelection();
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
