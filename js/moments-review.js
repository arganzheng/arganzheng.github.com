/*! moments-review.js — shared picker for the 随笔 每日回顾 sidebar and push. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.MomentsReview = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function pickReview(all, y, m, d) {
    var md = '-' + (m < 10 ? '0' : '') + m + '-' + (d < 10 ? '0' : '') + d;
    var out = [];
    all.forEach(function (e) {
      var ey = +e.date.slice(0, 4);
      if (e.date.slice(4) === md && ey < y) out.push({ e: e, label: (y - ey) + ' 年前的今天' });
    });
    if (!out.length) all.forEach(function (e) {
      var ey = +e.date.slice(0, 4), em = +e.date.slice(5, 7), ed = +e.date.slice(8, 10), ago = (y - ey) * 12 + (m - em);
      if (ed === d && ago > 0) out.push({ e: e, label: ago + ' 个月前的今天' });
    });
    return out.slice(0, 3);
  }

  return { pickReview: pickReview };
});
