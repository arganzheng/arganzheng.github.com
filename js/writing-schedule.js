(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WritingSchedule = api;
}(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  var DAY = 86400e3;

  function beijingToday(now) {
    var time = now == null ? Date.now() : (now instanceof Date ? now.getTime() : Number(now));
    return new Date(time + 8 * 3600e3).toISOString().slice(0, 10);
  }

  function upcoming(posts, now) {
    var today = beijingToday(now);
    return (posts || []).filter(function (post) { return post.date > today; }).slice().sort(function (a, b) {
      return a.date < b.date ? -1 : (a.date > b.date ? 1 : String(a.path).localeCompare(String(b.path)));
    });
  }

  function collisionDates(posts) {
    var byDate = Object.create(null);
    (posts || []).forEach(function (post) {
      (byDate[post.date] || (byDate[post.date] = [])).push(post);
    });
    return Object.keys(byDate).filter(function (date) {
      var day = byDate[date];
      if (day.length < 2) return false;
      if (day.length !== 2) return true;
      var a = day[0], b = day[1];
      return !(a.recap !== b.recap && a.series && a.series === b.series);
    }).sort();
  }

  function addDays(date, amount) {
    return new Date(Date.parse(date + 'T00:00:00Z') + amount * DAY).toISOString().slice(0, 10);
  }

  function gapDates(posts, today, lastDate) {
    if (!today || !lastDate || lastDate <= today) return [];
    var occupied = Object.create(null), gaps = [], run = [];
    (posts || []).forEach(function (post) { occupied[post.date] = true; });
    function finish() {
      if (run.length >= 3) gaps = gaps.concat(run);
      run = [];
    }
    for (var date = addDays(today, 1); date <= lastDate; date = addDays(date, 1)) {
      if (occupied[date]) finish();
      else run.push(date);
    }
    finish();
    return gaps;
  }

  function groupSchedule(posts) {
    var bySeries = Object.create(null), unassigned = [];
    (posts || []).forEach(function (post) {
      if (!post.series) { unassigned.push(post); return; }
      (bySeries[post.series] || (bySeries[post.series] = [])).push(post);
    });
    var series = Object.keys(bySeries).map(function (key) {
      var items = bySeries[key].slice().sort(function (a, b) {
        return a.date < b.date ? -1 : (a.date > b.date ? 1 : String(a.path).localeCompare(String(b.path)));
      });
      return { key: key, posts: items, nextDate: items[0].date };
    }).sort(function (a, b) {
      return a.nextDate < b.nextDate ? -1 : (a.nextDate > b.nextDate ? 1 : a.key.localeCompare(b.key));
    });
    series.forEach(function (group, index) {
      group.paletteIndex = paletteIndex(index);
    });
    unassigned.sort(function (a, b) { return a.date < b.date ? -1 : (a.date > b.date ? 1 : String(a.path).localeCompare(String(b.path))); });
    return { series: series, unassigned: unassigned };
  }

  function paletteIndex(position) {
    return position % 8;
  }

  return {
    beijingToday: beijingToday,
    upcoming: upcoming,
    collisionDates: collisionDates,
    addDays: addDays,
    gapDates: gapDates,
    groupSchedule: groupSchedule,
    paletteIndex: paletteIndex
  };
}));
