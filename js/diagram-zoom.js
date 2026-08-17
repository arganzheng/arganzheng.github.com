/*!
 * diagram-zoom.js
 * Click a Mermaid diagram to open it full screen, then zoom (wheel / buttons /
 * pinch) and pan (drag). Rendered diagrams are capped at the column width, which
 * makes the big flowcharts in long posts unreadable without this.
 */
(function () {
    'use strict';

    var SELECTOR = '.post-container .mermaid';
    var MIN_SCALE = 0.1;
    var MAX_SCALE = 12;
    var ZOOM_STEP = 1.25;

    var overlay, stage, content;
    var scale = 1, tx = 0, ty = 0;
    var natural = { width: 0, height: 0 };
    var dragging = false, lastX = 0, lastY = 0, pinchDistance = 0;
    var bodyOverflow = '';

    function clamp(value) {
        return Math.min(MAX_SCALE, Math.max(MIN_SCALE, value));
    }

    function apply() {
        content.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
    }

    function zoomAt(x, y, factor) {
        var next = clamp(scale * factor);
        if (next === scale) return;
        // Keep the point under the cursor fixed while scaling.
        tx = x - (x - tx) * (next / scale);
        ty = y - (y - ty) * (next / scale);
        scale = next;
        apply();
    }

    function stagePoint(clientX, clientY) {
        var rect = stage.getBoundingClientRect();
        return { x: clientX - rect.left, y: clientY - rect.top };
    }

    function fit() {
        var rect = stage.getBoundingClientRect();
        var padding = 32;
        scale = clamp(Math.min(
            (rect.width - padding) / natural.width,
            (rect.height - padding) / natural.height
        ));
        tx = (rect.width - natural.width * scale) / 2;
        ty = (rect.height - natural.height * scale) / 2;
        apply();
    }

    function build() {
        overlay = document.createElement('div');
        overlay.className = 'diagram-zoom';
        overlay.setAttribute('role', 'dialog');
        overlay.innerHTML =
            '<div class="diagram-zoom-stage"><div class="diagram-zoom-content"></div></div>' +
            '<div class="diagram-zoom-toolbar">' +
                '<button type="button" data-action="out" title="缩小">&minus;</button>' +
                '<button type="button" data-action="reset" title="适应窗口">重置</button>' +
                '<button type="button" data-action="in" title="放大">+</button>' +
                '<button type="button" data-action="close" title="关闭 (Esc)">&times;</button>' +
            '</div>' +
            '<div class="diagram-zoom-hint">滚轮缩放 · 拖动平移 · 双击重置 · Esc 关闭</div>';
        stage = overlay.querySelector('.diagram-zoom-stage');
        content = overlay.querySelector('.diagram-zoom-content');
        document.body.appendChild(overlay);
        bindOverlay();
    }

    function bindOverlay() {
        overlay.querySelector('.diagram-zoom-toolbar').addEventListener('click', function (e) {
            var button = e.target.closest('button');
            if (!button) return;
            var center = { x: stage.clientWidth / 2, y: stage.clientHeight / 2 };
            if (button.dataset.action === 'in') zoomAt(center.x, center.y, ZOOM_STEP);
            if (button.dataset.action === 'out') zoomAt(center.x, center.y, 1 / ZOOM_STEP);
            if (button.dataset.action === 'reset') fit();
            if (button.dataset.action === 'close') close();
        });

        stage.addEventListener('wheel', function (e) {
            e.preventDefault();
            var point = stagePoint(e.clientX, e.clientY);
            zoomAt(point.x, point.y, e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
        }, { passive: false });

        stage.addEventListener('mousedown', function (e) {
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            overlay.classList.add('dragging');
            e.preventDefault();
        });

        document.addEventListener('mousemove', function (e) {
            if (!dragging) return;
            tx += e.clientX - lastX;
            ty += e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            apply();
        });

        document.addEventListener('mouseup', function () {
            dragging = false;
            if (overlay) overlay.classList.remove('dragging');
        });

        stage.addEventListener('touchstart', function (e) {
            if (e.touches.length === 1) {
                lastX = e.touches[0].clientX;
                lastY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                pinchDistance = touchDistance(e.touches);
            }
        }, { passive: true });

        stage.addEventListener('touchmove', function (e) {
            if (e.touches.length === 1) {
                tx += e.touches[0].clientX - lastX;
                ty += e.touches[0].clientY - lastY;
                lastX = e.touches[0].clientX;
                lastY = e.touches[0].clientY;
                apply();
            } else if (e.touches.length === 2 && pinchDistance) {
                var distance = touchDistance(e.touches);
                var mid = stagePoint(
                    (e.touches[0].clientX + e.touches[1].clientX) / 2,
                    (e.touches[0].clientY + e.touches[1].clientY) / 2
                );
                zoomAt(mid.x, mid.y, distance / pinchDistance);
                pinchDistance = distance;
            }
            e.preventDefault();
        }, { passive: false });

        stage.addEventListener('dblclick', fit);

        overlay.addEventListener('click', function (e) {
            if (e.target === overlay || e.target === stage) close();
        });

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && overlay.classList.contains('open')) close();
        });
    }

    function touchDistance(touches) {
        var dx = touches[0].clientX - touches[1].clientX;
        var dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function open(svg) {
        if (!overlay) build();

        var clone = svg.cloneNode(true);
        var box = svg.viewBox && svg.viewBox.baseVal;
        var rect = svg.getBoundingClientRect();
        natural.width = (box && box.width) || rect.width || 800;
        natural.height = (box && box.height) || rect.height || 600;

        // Mermaid injects a <style> inside the svg whose rules are all scoped by the
        // svg's id, so the clone needs its own id wired into those rules - otherwise
        // it renders with default (black) fills.
        var sourceId = svg.getAttribute('id');
        if (sourceId) {
            var cloneId = sourceId + '-zoom';
            clone.setAttribute('id', cloneId);
            Array.prototype.forEach.call(clone.querySelectorAll('style'), function (style) {
                style.textContent = style.textContent.split('#' + sourceId).join('#' + cloneId);
            });
        }

        clone.style.maxWidth = 'none';
        clone.style.width = natural.width + 'px';
        clone.style.height = natural.height + 'px';

        content.innerHTML = '';
        content.appendChild(clone);

        bodyOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        overlay.classList.add('open');
        fit();
    }

    function close() {
        overlay.classList.remove('open');
        content.innerHTML = '';
        document.body.style.overflow = bodyOverflow;
    }

    document.addEventListener('click', function (e) {
        if (overlay && overlay.contains(e.target)) return;
        var diagram = e.target.closest && e.target.closest(SELECTOR);
        var svg = diagram && diagram.querySelector('svg');
        if (svg) open(svg);
    });

    window.addEventListener('resize', function () {
        if (overlay && overlay.classList.contains('open')) fit();
    });
})();
