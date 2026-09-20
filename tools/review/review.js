/* Review page behaviour (tools/review.py): mode toggle, folds, PR comments. */
(function () {
  'use strict';
  var body = document.body;
  var mode = localStorage.getItem('rv:mode') || 'unified';
  function setMode(m) {
    mode = m; localStorage.setItem('rv:mode', m);
    body.classList.toggle('rv-mode-unified', m === 'unified');
    body.classList.toggle('rv-mode-split', m === 'split');
    var r = document.querySelector('input[name=mode][value="' + m + '"]'); if (r) r.checked = true;
  }
  setMode(mode);
  Array.prototype.forEach.call(document.querySelectorAll('input[name=mode]'), function (r) {
    r.addEventListener('change', function () { setMode(r.value); });
  });
  var expand = document.getElementById('rv-expand');
  if (expand) {
    expand.checked = localStorage.getItem('rv:expand') === '1';
    var apply = function () {
      localStorage.setItem('rv:expand', expand.checked ? '1' : '0');
      Array.prototype.forEach.call(document.querySelectorAll('.rv-fold'), function (d) { d.open = expand.checked; });
    };
    expand.addEventListener('change', apply); apply();
  }

  function api(path, data) {
    return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok || j.error) throw new Error(j.error || r.status); return j; }); });
  }
  function say(el, text, err) { if (!el) return; el.textContent = text; el.classList.toggle('is-err', !!err); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // section comment → new PR line comment
  Array.prototype.forEach.call(document.querySelectorAll('.rv-form'), function (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var ta = form.querySelector('textarea'), msg = form.querySelector('.rv-msg'), btn = form.querySelector('button');
      var text = ta.value.trim(); if (!text) return;
      btn.disabled = true; say(msg, '发送中…');
      api('/_api/comment', { path: form.dataset.path, line: form.dataset.line, side: form.dataset.side, body: text })
        .then(function (c) {
          var card = document.createElement('div');
          card.className = 'rv-thread'; card.dataset.first = c.id;
          card.innerHTML = '<div class="rv-c"><img src="' + esc(c.user.avatar_url) + '" alt=""><div><b>' + esc(c.user.login) + '</b> <time>刚刚</time><p>' + esc(text) + '</p></div></div>' +
            '<div class="rv-thread-foot"><a href="' + esc(c.html_url) + '" target="_blank" rel="noopener">GitHub ↗</a> <button type="button" class="rv-reply">回复</button></div>';
          form.parentNode.insertBefore(card, form); bindReply(card);
          ta.value = ''; say(msg, '已发到 PR'); btn.disabled = false;
        })
        .catch(function (err) { say(msg, '失败：' + err.message, true); btn.disabled = false; });
    });
  });

  // reply inside an existing thread
  function bindReply(card) {
    var b = card.querySelector('.rv-reply'); if (!b) return;
    b.addEventListener('click', function () {
      if (card.querySelector('.rv-reply-form')) return;
      var f = document.createElement('form'); f.className = 'rv-form rv-reply-form';
      f.innerHTML = '<textarea rows="2" placeholder="回复……"></textarea><div><button type="submit">发送</button><span class="rv-msg"></span></div>';
      card.appendChild(f);
      f.addEventListener('submit', function (e) {
        e.preventDefault();
        var ta = f.querySelector('textarea'), msg = f.querySelector('.rv-msg'), text = ta.value.trim(); if (!text) return;
        say(msg, '发送中…');
        api('/_api/comment', { reply_to: card.dataset.first, body: text })
          .then(function (c) {
            var d = document.createElement('div'); d.className = 'rv-c';
            d.innerHTML = '<img src="' + esc(c.user.avatar_url) + '" alt=""><div><b>' + esc(c.user.login) + '</b> <time>刚刚</time><p>' + esc(text) + '</p></div>';
            card.insertBefore(d, card.querySelector('.rv-thread-foot')); f.remove();
          })
          .catch(function (err) { say(msg, '失败：' + err.message, true); });
      });
    });
  }
  Array.prototype.forEach.call(document.querySelectorAll('.rv-thread'), bindReply);

  // approve / request changes (index page)
  Array.prototype.forEach.call(document.querySelectorAll('.rv-pr-actions button'), function (b) {
    b.addEventListener('click', function () {
      var msg = b.parentNode.querySelector('.rv-msg');
      var text = window.prompt(b.dataset.event === 'APPROVE' ? '批准说明（可空）' : '要求修改：说明', '');
      if (text === null) return;
      say(msg, '提交中…');
      api('/_api/review', { event: b.dataset.event, body: text })
        .then(function () { say(msg, b.dataset.event === 'APPROVE' ? '已批准' : '已提交「要求修改」'); })
        .catch(function (err) { say(msg, '失败：' + err.message, true); });
    });
  });
})();
