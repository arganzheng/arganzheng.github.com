/**
 * tagcloud.js — colour the tag pills on /tags/ by how many posts carry the tag
 * (`rel="<count>"` on each `#tag_cloud a`): background interpolated from
 * `start` (rare) to `end` (common). Replaces jquery.tagcloud.js; loaded by
 * _includes/footer.html only when the page has `#tag_cloud`.
 */
(function () {
  'use strict';
  var COLOR = { start: '#bbbbee', end: '#0085a1' };
  var links = document.querySelectorAll('#tag_cloud a[rel]');
  if (!links.length) return;
  function rgb(hex) {
    var h = hex.replace('#', '');
    if (h.length === 3) h = h.replace(/./g, function (c) { return c + c; });
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  var weights = Array.prototype.map.call(links, function (a) { return parseFloat(a.getAttribute('rel')) || 0; });
  var lowest = Math.min.apply(null, weights), range = Math.max.apply(null, weights) - lowest || 1;
  var from = rgb(COLOR.start), to = rgb(COLOR.end);
  Array.prototype.forEach.call(links, function (a, i) {
    var w = (weights[i] - lowest) / range;
    var c = from.map(function (n, k) { return Math.max(0, Math.min(255, Math.round(n + (to[k] - n) * w))); });
    a.style.backgroundColor = 'rgb(' + c.join(',') + ')';
  });
})();
