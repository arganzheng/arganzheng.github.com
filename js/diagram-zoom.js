/*!
 * diagram-zoom.js
 * Lightbox for Mermaid diagrams and content images: full screen, then zoom
 * (wheel / buttons / pinch) and pan (drag); Moments galleries also navigate
 * between original images with buttons, arrow keys and fit-scale swipes.
 * Exposed as window.DiagramZoom.open(el, opts); js/figures.js puts the 放大
 * button in each figure's corner strip. Tiny images (icons, QR codes < 200 px) get none.
 */
(function () {
    'use strict';

    var IMG_MIN = 200;
    var MIN_SCALE = 0.1;
    var MAX_SCALE = 12;
    var ZOOM_STEP = 1.25;

    var overlay, stage, content;
    var scale = 1, tx = 0, ty = 0;
    var fitScale = 1;
    var natural = { width: 0, height: 0 };
    var dragging = false, lastX = 0, lastY = 0, pinchDistance = 0;
    var bodyOverflow = '';
    var gallery = null, galleryIndex = 0;
    var galleryControls, galleryPrevious, galleryNext, galleryCounter;
    var swipeStartX = 0, swipeStartY = 0, swiping = false;

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
        fitScale = clamp(Math.min(
            (rect.width - padding) / natural.width,
            (rect.height - padding) / natural.height
        ));
        scale = fitScale;
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
            '<div class="diagram-zoom-gallery" hidden>' +
                '<button type="button" class="diagram-zoom-previous" aria-label="上一张">‹</button>' +
                '<span class="diagram-zoom-counter" aria-live="polite"></span>' +
                '<button type="button" class="diagram-zoom-next" aria-label="下一张">›</button>' +
            '</div>' +
            '<div class="diagram-zoom-hint">滚轮缩放 · 拖动平移 · 双击重置 · Esc 关闭</div>';
        stage = overlay.querySelector('.diagram-zoom-stage');
        content = overlay.querySelector('.diagram-zoom-content');
        galleryControls = overlay.querySelector('.diagram-zoom-gallery');
        galleryPrevious = overlay.querySelector('.diagram-zoom-previous');
        galleryNext = overlay.querySelector('.diagram-zoom-next');
        galleryCounter = overlay.querySelector('.diagram-zoom-counter');
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

        galleryControls.addEventListener('click', function (e) {
            var button = e.target.closest('button');
            if (!button || !gallery) return;
            if (button === galleryPrevious) showGalleryImage(galleryIndex - 1);
            if (button === galleryNext) showGalleryImage(galleryIndex + 1);
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
                swipeStartX = lastX;
                swipeStartY = lastY;
                swiping = false;
            } else if (e.touches.length === 2) {
                pinchDistance = touchDistance(e.touches);
                swiping = false;
            }
        }, { passive: true });

        stage.addEventListener('touchmove', function (e) {
            if (e.touches.length === 1) {
                var x = e.touches[0].clientX;
                var y = e.touches[0].clientY;
                var dx = x - swipeStartX;
                var dy = y - swipeStartY;
                if (gallery && scale <= fitScale + 0.001 && Math.abs(dx) > 8 && Math.abs(dx) > Math.abs(dy) * 1.5) {
                    swiping = true;
                    e.preventDefault();
                    return;
                }
                if (swiping) {
                    e.preventDefault();
                    return;
                }
                tx += x - lastX;
                ty += y - lastY;
                lastX = x;
                lastY = y;
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

        stage.addEventListener('touchend', function (e) {
            if (swiping && gallery && e.changedTouches.length) {
                var dx = e.changedTouches[0].clientX - swipeStartX;
                var dy = e.changedTouches[0].clientY - swipeStartY;
                if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) {
                    showGalleryImage(galleryIndex + (dx < 0 ? 1 : -1));
                }
            }
            swiping = false;
            pinchDistance = 0;
        });

        stage.addEventListener('dblclick', fit);

        overlay.addEventListener('click', function (e) {
            if (e.target === overlay || e.target === stage) close();
        });

        document.addEventListener('keydown', function (e) {
            if (!overlay.classList.contains('open')) return;
            if (e.key === 'Escape') close();
            if (gallery && e.key === 'ArrowLeft') {
                e.preventDefault();
                showGalleryImage(galleryIndex - 1);
            }
            if (gallery && e.key === 'ArrowRight') {
                e.preventDefault();
                showGalleryImage(galleryIndex + 1);
            }
        });
    }

    function touchDistance(touches) {
        var dx = touches[0].clientX - touches[1].clientX;
        var dy = touches[0].clientY - touches[1].clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function updateGalleryControls() {
        galleryControls.hidden = !gallery;
        if (!gallery) return;
        galleryCounter.textContent = (galleryIndex + 1) + ' / ' + gallery.length;
        galleryPrevious.disabled = galleryIndex === 0;
        galleryNext.disabled = galleryIndex === gallery.length - 1;
    }

    function showGalleryImage(index, placeholder) {
        if (!gallery || index < 0 || index >= gallery.length) return;
        galleryIndex = index;
        updateGalleryControls();

        var item = gallery[galleryIndex];
        var image = document.createElement('img');
        image.alt = placeholder ? placeholder.alt : '';
        image.style.margin = '0';
        image.style.maxWidth = 'none';
        image.style.width = (natural.width || 800) + 'px';
        image.style.height = (natural.height || 600) + 'px';
        image.onload = function () {
            if (content.firstChild !== image) return;
            natural.width = image.naturalWidth || natural.width;
            natural.height = image.naturalHeight || natural.height;
            image.style.width = natural.width + 'px';
            image.style.height = natural.height + 'px';
            fit();
        };
        content.innerHTML = '';
        content.appendChild(image);
        if (placeholder) {
            natural.width = placeholder.naturalWidth || placeholder.getBoundingClientRect().width || 800;
            natural.height = placeholder.naturalHeight || placeholder.getBoundingClientRect().height || 600;
            image.src = placeholder.currentSrc || placeholder.src;
            image.style.width = natural.width + 'px';
            image.style.height = natural.height + 'px';
        }
        fit();
        image.src = item.src;
    }

    function open(svg, opts) {
        if (!overlay) build();

        gallery = opts && opts.list && opts.list.length > 1 ? opts.list : null;
        galleryIndex = gallery ? Math.max(0, Math.min(gallery.length - 1, Number(opts.index) || 0)) : 0;
        var singleImageSource = opts && opts.list && opts.list.length === 1 && svg.tagName === 'IMG' ?
            opts.list[0].src : null;
        updateGalleryControls();
        bodyOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';
        overlay.classList.add('open');
        if (gallery) {
            showGalleryImage(galleryIndex, svg);
            return;
        }

        var clone = svg.cloneNode(true);
        var box = svg.viewBox && svg.viewBox.baseVal;
        var rect = svg.getBoundingClientRect();
        natural.width = (box && box.width) || rect.width || 800;
        natural.height = (box && box.height) || rect.height || 600;
        if (svg.tagName === 'IMG') {
            natural.width = svg.naturalWidth || rect.width || 800;
            natural.height = svg.naturalHeight || rect.height || 600;
            clone.removeAttribute('loading');
            clone.style.margin = '0';
            if (singleImageSource) {
                clone.removeAttribute('srcset');
                clone.removeAttribute('sizes');
                clone.onload = function () {
                    if (content.firstChild !== clone) return;
                    natural.width = clone.naturalWidth || natural.width;
                    natural.height = clone.naturalHeight || natural.height;
                    clone.style.width = natural.width + 'px';
                    clone.style.height = natural.height + 'px';
                    fit();
                };
                clone.src = singleImageSource;
            }
        }

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
        fit();
    }

    function close() {
        overlay.classList.remove('open');
        content.innerHTML = '';
        gallery = null;
        updateGalleryControls();
        document.body.style.overflow = bodyOverflow;
    }

    // Icons and QR codes (< 200 px) are not worth a lightbox.
    function zoomable(img) {
        return !img.closest('a, .comment, .annotation-panel') && img.naturalWidth >= IMG_MIN;
    }

    // Opened from the corner button js/figures.js puts on every picture and
    // diagram — not by clicking the picture itself: a drag that selects a
    // caption or the text around a figure ends with a click on it, and the
    // lightbox used to swallow the 划线 toolbar (2026-09-16).
    window.DiagramZoom = { open: open, zoomable: zoomable };

    window.addEventListener('resize', function () {
        if (overlay && overlay.classList.contains('open')) fit();
    });
})();
