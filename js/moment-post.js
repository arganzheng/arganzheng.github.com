/*!
 * moment-post.js — the phone 发布页 for 随笔 (/moments/post.html; not part of
 * blog.min.js). Login = the comments' giscus OAuth (localStorage
 * "giscus-session" → worker POST /token → GitHub token); the worker's
 * POST /moments accepts the request only when GET /user is the repo owner.
 * Pictures are shrunk to ≤ 1600px wide in a canvas (WebP when the browser can
 * encode it, else JPEG) and sent base64 inside the JSON. The draft (text,
 * tags, place, quote, music) survives a reload in localStorage; pictures don't.
 */
(function () {
  var root = document.getElementById('mp');
  if (!root) return;
  var API = (root.getAttribute('data-api') || '').replace(/\/$/, '');
  var OWNER = (root.getAttribute('data-owner') || '').toLowerCase();
  var BASE = root.getAttribute('data-base') || '';
  var SESSION_KEY = 'giscus-session', DRAFT_KEY = 'moment-post-draft';
  var MAX_PICS = 9, MAX_W = 1600;
  // ?edit=YYYY-MM/<id> (the 编辑 link on a card): load that entry from the worker, PUT it back.
  var EDIT = /[?&]edit=(\d{4}-\d{2})\/(\d{8}(?:-\d{4})?(?:-\d+)?)/.exec(location.search);
  EDIT = EDIT ? { month: EDIT[1], id: EDIT[2] } : null;
  var dateOnly = false, timeTouched = false;
  var TAG = /(^|[^\p{L}\p{N}_\/&\\])#([\p{L}_][\p{L}\p{N}_\-·]*(?:\/[\p{L}\p{N}_\-·]+)*)/gu;

  function $(s) { return root.querySelector(s); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }

  var text = $('.mp-text'), file = $('.mp-file'), pics = $('.mp-pics'), tagsBox = $('.mp-tags'), tagNew = $('.mp-tag-new');
  var place = $('.mp-place'), quote = $('.mp-quote'), by = $('.mp-by'), music = $('.mp-music'), time = $('.mp-time');
  var submit = $('.mp-submit'), msg = $('.mp-msg'), loginBtn = $('.mp-login'), who = $('.mp-who');
  var done = $('.mp-done'), editor = $('.mp-editor');
  var token = null, user = null, images = [], busy = false;

  // ---- login (same flow as js/annotations.js)
  function takeSessionFromUrl() {
    var url = new URL(location.href), s = url.searchParams.get('giscus');
    if (!s) return;
    try { localStorage.setItem(SESSION_KEY, JSON.stringify(s)); } catch (e) { /* ignore */ }
    url.searchParams.delete('giscus');
    history.replaceState(history.state, '', url.toString());
  }
  function session() { try { var r = localStorage.getItem(SESSION_KEY); return r ? JSON.parse(r) : ''; } catch (e) { return ''; } }
  function login() {
    var url = new URL(location.href); url.hash = ''; url.searchParams.delete('giscus');
    location.href = 'https://giscus.app/api/oauth/authorize?redirect_uri=' + encodeURIComponent(url.toString());
  }
  function logout() { try { localStorage.removeItem(SESSION_KEY); } catch (e) { /* ignore */ } token = user = null; renderAuth(); }
  function api(path, opts) {
    opts = opts || {};
    var headers = { 'Content-Type': 'application/json' };
    if (opts.token) headers.Authorization = 'Bearer ' + opts.token;
    return fetch(API + path, { method: opts.method || 'GET', headers: headers, body: opts.body ? JSON.stringify(opts.body) : undefined })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (d) { if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status)); return d; }); });
  }
  function ensureToken() {
    if (token) return Promise.resolve(token);
    var s = session();
    if (!s) return Promise.reject(new Error('尚未登录'));
    return api('/token', { method: 'POST', body: { session: s } }).then(function (d) { if (!d.token) throw new Error('no token'); token = d.token; return token; });
  }
  function whoAmI() {
    return ensureToken().then(function (t) {
      return fetch('https://api.github.com/user', { headers: { Authorization: 'Bearer ' + t, Accept: 'application/vnd.github+json' } });
    }).then(function (r) { if (r.status === 401) { logout(); throw new Error('登录已过期'); } return r.json(); })
      .then(function (u) { user = u; renderAuth(); });
  }
  function isAuthor() { return !!(user && String(user.login || '').toLowerCase() === OWNER); }
  function renderAuth() {
    if (!user) {
      loginBtn.hidden = false; who.hidden = true; who.innerHTML = '';
      setMsg(session() ? '' : '先登录 GitHub（和评论用的是同一个登录）');
    } else {
      loginBtn.hidden = true; who.hidden = false;
      who.innerHTML = '<img src="' + esc(user.avatar_url || '') + '" alt=""> <b>@' + esc(user.login) + '</b> <button type="button" class="mp-logout">退出</button>';
      who.querySelector('.mp-logout').addEventListener('click', logout);
      setMsg(isAuthor() ? '' : '只有博客作者 @' + OWNER + ' 可以发布', !isAuthor());
    }
    updateSubmit();
  }

  // ---- draft
  function saveDraft() {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ text: text.value, tags: selectedTags(), place: place.value, quote: quote.value, by: by.value, music: music.value })); } catch (e) { /* ignore */ }
  }
  function loadDraft() {
    var d; try { d = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null'); } catch (e) { d = null; }
    if (!d) return;
    text.value = d.text || ''; place.value = d.place || ''; quote.value = d.quote || ''; by.value = d.by || ''; music.value = d.music || '';
    (d.tags || []).forEach(function (t) { toggleTag(t, true); });
    if (d.place || d.quote || d.music) $('.mp-more').open = true;
  }
  function clearDraft() { try { localStorage.removeItem(DRAFT_KEY); } catch (e) { /* ignore */ } }

  // ---- tags: chips of the existing tags + new ones typed in
  function selectedTags() { return Array.prototype.map.call(tagsBox.querySelectorAll('.mp-tag.is-on'), function (b) { return b.getAttribute('data-tag'); }); }
  function toggleTag(tag, on) {
    var btn = tagsBox.querySelector('.mp-tag[data-tag="' + tag.replace(/"/g, '\\"') + '"]');
    if (!btn) {
      btn = document.createElement('button'); btn.type = 'button'; btn.className = 'mp-tag'; btn.setAttribute('data-tag', tag); btn.textContent = '#' + tag;
      tagsBox.insertBefore(btn, tagNew);
    }
    btn.classList.toggle('is-on', on === undefined ? !btn.classList.contains('is-on') : on);
  }
  tagsBox.addEventListener('click', function (e) {
    var b = e.target.closest('.mp-tag'); if (!b) return;
    toggleTag(b.getAttribute('data-tag')); onChange();
  });
  function addNewTag() {
    var v = tagNew.value.trim().replace(/^#/, '').replace(/\s+/g, '');
    if (!v) return;
    toggleTag(v, true); tagNew.value = ''; onChange();
  }
  tagNew.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ' || e.key === ',') { e.preventDefault(); addNewTag(); } });
  tagNew.addEventListener('blur', addNewTag);

  // ---- pictures
  function shrink(f) {
    return new Promise(function (res, rej) {
      var url = URL.createObjectURL(f), img = new Image();
      img.onload = function () {
        var s = Math.min(1, MAX_W / img.naturalWidth);
        var c = document.createElement('canvas');
        c.width = Math.max(1, Math.round(img.naturalWidth * s)); c.height = Math.max(1, Math.round(img.naturalHeight * s));
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(url);
        c.toBlob(function (b) {
          if (b && b.type === 'image/webp') return res(b);
          c.toBlob(function (j) { j ? res(j) : rej(new Error('无法处理图片')); }, 'image/jpeg', 0.85);
        }, 'image/webp', 0.82);
      };
      img.onerror = function () { URL.revokeObjectURL(url); rej(new Error('无法读取图片 ' + (f.name || ''))); };
      img.src = url;
    });
  }
  function toBase64(blob) {
    return new Promise(function (res, rej) {
      var r = new FileReader();
      r.onload = function () { res(String(r.result).split(',')[1]); };
      r.onerror = function () { rej(new Error('读取失败')); };
      r.readAsDataURL(blob);
    });
  }
  file.addEventListener('change', function () {
    var list = Array.prototype.slice.call(file.files || []);
    file.value = '';
    if (!list.length) return;
    if (images.length + list.length > MAX_PICS) { setMsg('最多 ' + MAX_PICS + ' 张图', true); list = list.slice(0, MAX_PICS - images.length); }
    busy = true; updateSubmit(); setMsg('处理图片…');
    var chain = Promise.resolve();
    list.forEach(function (f) {
      chain = chain.then(function () { return shrink(f); }).then(function (b) {
        images.push({ blob: b, url: URL.createObjectURL(b) }); renderPics(); renderPreview();
      }).catch(function (e) { setMsg(e.message, true); });
    });
    chain.then(function () { busy = false; setMsg(''); updateSubmit(); });
  });
  // Order = gallery order. ‹ › on each thumbnail (the phone), drag on a mouse.
  function renderPics() {
    Array.prototype.forEach.call(pics.querySelectorAll('.mp-pic'), function (n) { n.remove(); });
    var add = pics.querySelector('.mp-add');
    images.forEach(function (im, i) {
      var d = document.createElement('div'); d.className = 'mp-pic'; d.draggable = true; d.setAttribute('data-i', i);
      d.innerHTML = '<img src="' + im.url + '" alt="">' +
        '<button type="button" class="mp-pic-x" data-i="' + i + '" title="移除">×</button>' +
        (images.length > 1 ? '<span class="mp-pic-mv">' +
          '<button type="button" class="mp-pic-l" data-i="' + i + '" title="前移"' + (i === 0 ? ' disabled' : '') + '>‹</button>' +
          '<button type="button" class="mp-pic-r" data-i="' + i + '" title="后移"' + (i === images.length - 1 ? ' disabled' : '') + '>›</button></span>' : '');
      pics.insertBefore(d, add);
    });
    add.hidden = images.length >= MAX_PICS;
  }
  function movePic(from, to) {
    if (to < 0 || to >= images.length || from === to) return;
    images.splice(to, 0, images.splice(from, 1)[0]);
    renderPics(); renderPreview();
  }
  pics.addEventListener('click', function (e) {
    var b = e.target.closest('.mp-pic-x, .mp-pic-l, .mp-pic-r'); if (!b || b.disabled) return;
    var i = +b.getAttribute('data-i');
    if (b.classList.contains('mp-pic-l')) return movePic(i, i - 1);
    if (b.classList.contains('mp-pic-r')) return movePic(i, i + 1);
    if (images[i].blob) URL.revokeObjectURL(images[i].url); images.splice(i, 1);
    renderPics(); renderPreview(); updateSubmit();
  });
  var dragFrom = -1;
  pics.addEventListener('dragstart', function (e) { var p = e.target.closest('.mp-pic'); if (!p) return; dragFrom = +p.getAttribute('data-i'); e.dataTransfer.effectAllowed = 'move'; p.classList.add('is-dragging'); });
  pics.addEventListener('dragover', function (e) { if (dragFrom < 0) return; e.preventDefault(); var p = e.target.closest('.mp-pic'); Array.prototype.forEach.call(pics.querySelectorAll('.mp-pic.is-over'), function (n) { n.classList.remove('is-over'); }); if (p) p.classList.add('is-over'); });
  pics.addEventListener('drop', function (e) { var p = e.target.closest('.mp-pic'); if (dragFrom < 0 || !p) return; e.preventDefault(); movePic(dragFrom, +p.getAttribute('data-i')); dragFrom = -1; });
  pics.addEventListener('dragend', function () { dragFrom = -1; Array.prototype.forEach.call(pics.querySelectorAll('.is-dragging, .is-over'), function (n) { n.classList.remove('is-dragging', 'is-over'); }); });

  // ---- preview card (same markup as _layouts/moments.html / _plugins/moments.rb)
  function hasContent() { return !!(text.value.trim() || selectedTags().length || quote.value.trim() || images.length || music.value.trim()); }
  function fmtTime(v) { return v ? v.replace('T', ' ') : ''; }
  function nowLocal() {
    var d = new Date(), p = function (n) { return (n < 10 ? '0' : '') + n; };
    return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + 'T' + p(d.getHours()) + ':' + p(d.getMinutes());
  }
  function renderText(s) {
    // paragraphs; `#标签` → chips; bare URLs → links. Not full Markdown — the site does that.
    return s.split(/\n{2,}/).map(function (para) {
      var h = esc(para).replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1">$1</a>').replace(/\n/g, '<br>');
      h = h.replace(TAG, function (m, pre, t) { return pre + '<a class="moment-tag">#' + t + '</a>'; });
      return '<p>' + h + '</p>';
    }).join('');
  }
  function renderPreview() {
    $('.mp-pv-time').textContent = fmtTime(time.value) || '现在';
    var pl = $('.mp-pv-place'); pl.hidden = !place.value.trim(); pl.querySelector('span').textContent = place.value.trim();
    var h = '';
    var t = text.value.trim(), tags = selectedTags().map(function (x) { return '#' + x; }).join(' ');
    if (t || tags) h += renderText((t + ' ' + tags).trim());
    if (quote.value.trim()) {
      h += '<blockquote class="moment-quote"><p>' + esc(quote.value.trim()).replace(/\n/g, '<br>') + '</p>' + (by.value.trim() ? '<div class="moment-cite">' + esc(by.value.trim()) + '</div>' : '') + '</blockquote>';
    }
    if (images.length) h += '<div class="moment-gallery n-' + images.length + '">' + images.map(function (im) { return '<span class="moment-pic"><img src="' + im.url + '" alt=""></span>'; }).join('') + '</div>';
    if (music.value.trim()) h += '<p class="mp-pv-music"><i class="fa fa-music"></i> ' + esc(music.value.trim()) + '</p>';
    $('.mp-pv-body').innerHTML = h || '<p class="mp-pv-empty">写点什么，这里会出现预览。</p>';
  }

  function setMsg(s, bad) { msg.textContent = s || ''; msg.classList.toggle('is-bad', !!bad); }
  function updateSubmit() { submit.disabled = busy || !isAuthor() || !hasContent(); }
  function onChange() { if (!EDIT) saveDraft(); renderPreview(); updateSubmit(); }
  [text, place, quote, by, music].forEach(function (el) { el.addEventListener('input', onChange); });
  time.addEventListener('input', function () { timeTouched = true; renderPreview(); });
  text.addEventListener('input', function () { text.style.height = 'auto'; text.style.height = Math.min(text.scrollHeight, 400) + 'px'; });

  // ---- publish
  submit.addEventListener('click', function () {
    if (submit.disabled) return;
    busy = true; updateSubmit(); setMsg('上传中…');
    var when = fmtTime(time.value);
    if (dateOnly && !timeTouched) when = when.slice(0, 10); // the entry had no time of day; keep it so
    var payload = { text: text.value.trim(), place: place.value.trim(), tags: selectedTags(), quote: quote.value.trim(), by: by.value.trim(), music: music.value.trim(), time: when, images: [] };
    if (EDIT) { payload.month = EDIT.month; payload.id = EDIT.id; }
    Promise.all(images.map(function (im) {
      if (!im.blob) return { url: im.keep };
      return toBase64(im.blob).then(function (d) { return { name: '', type: im.blob.type, data: d }; });
    }))
      .then(function (imgs) { payload.images = imgs; return ensureToken(); })
      .then(function (t) { return api('/moments', { method: EDIT ? 'PUT' : 'POST', token: t, body: payload }); })
      .then(function (r) {
        if (!EDIT) clearDraft();
        $('.mp-done-title').innerHTML = '<i class="fa fa-check"></i> ' + (EDIT ? '已更新' : '已提交');
        $('.mp-done-link').href = BASE + r.url;
        $('.mp-done-commit').href = 'https://github.com/' + (root.getAttribute('data-repo') || '') + '/commit/' + r.commit;
        $('.mp-done-commit').hidden = !root.getAttribute('data-repo');
        editor.hidden = true; done.hidden = false; $('.mp-preview-wrap').hidden = true;
        busy = false; setMsg('');
      })
      .catch(function (e) {
        busy = false; updateSubmit();
        if (/登录/.test(e.message)) { token = null; }
        setMsg('发布失败：' + e.message, true);
      });
  });
  // ---- edit mode: 删除 + loading the entry
  $('.mp-del').addEventListener('click', function () {
    if (!EDIT || busy || !isAuthor()) return;
    if (!window.confirm('删除这条随笔？\n会提交一个删除的 commit，图片一起删掉，1–2 分钟后生效。')) return;
    busy = true; updateSubmit(); setMsg('删除中…');
    ensureToken().then(function (t) { return api('/moments', { method: 'DELETE', token: t, body: { month: EDIT.month, id: EDIT.id } }); })
      .then(function (r) {
        $('.mp-done-title').innerHTML = '<i class="fa fa-check"></i> 已删除';
        $('.mp-done-link').href = BASE + r.month; $('.mp-done-link').textContent = '回到这个月的随笔 ›';
        $('.mp-done-commit').href = 'https://github.com/' + (root.getAttribute('data-repo') || '') + '/commit/' + r.commit;
        $('.mp-done-commit').hidden = !root.getAttribute('data-repo');
        editor.hidden = true; done.hidden = false; $('.mp-preview-wrap').hidden = true;
        busy = false; setMsg('');
      })
      .catch(function (e) { busy = false; updateSubmit(); setMsg('删除失败：' + e.message, true); });
  });
  function loadEntry() {
    busy = true; updateSubmit(); setMsg('读取随笔…');
    return ensureToken().then(function (t) { return api('/moments?month=' + encodeURIComponent(EDIT.month) + '&id=' + encodeURIComponent(EDIT.id), { token: t }); })
      .then(function (d) {
        var t = d.text || '', m = /(?:^|\s)((?:#[^\s#]+\s*)+)$/.exec(t);
        if (m && !d.raw) { t = t.slice(0, t.length - m[1].length).trim(); m[1].trim().split(/\s+/).forEach(function (x) { toggleTag(x.replace(/^#/, ''), true); }); }
        text.value = t; place.value = d.place || ''; quote.value = d.quote || ''; by.value = d.by || ''; music.value = d.music || '';
        if (d.place || d.quote || d.music) $('.mp-more').open = true;
        dateOnly = (d.time || '').length === 10;
        time.value = dateOnly ? d.time + 'T00:00' : (d.time || '').replace(' ', 'T');
        images = (d.images || []).map(function (u) { return { url: BASE + u, keep: u }; });
        var note = $('.mp-note'); note.hidden = !d.raw;
        text.dispatchEvent(new Event('input'));
        renderPics(); renderPreview();
        busy = false; setMsg(''); updateSubmit();
      })
      .catch(function (e) { busy = false; updateSubmit(); setMsg('读取失败：' + e.message, true); });
  }

  $('.mp-again').addEventListener('click', function () {
    if (EDIT) { location.href = location.pathname; return; }
    text.value = ''; quote.value = ''; by.value = ''; music.value = ''; place.value = '';
    images.forEach(function (im) { if (im.blob) URL.revokeObjectURL(im.url); }); images = [];
    Array.prototype.forEach.call(tagsBox.querySelectorAll('.mp-tag.is-on'), function (b) { b.classList.remove('is-on'); });
    time.value = nowLocal(); renderPics(); renderPreview(); updateSubmit();
    editor.hidden = false; done.hidden = true; $('.mp-preview-wrap').hidden = false; text.focus();
  });
  loginBtn.addEventListener('click', login);

  // ---- boot
  takeSessionFromUrl();
  time.value = nowLocal();
  if (EDIT) { root.classList.add('is-edit'); $('.mp-title').textContent = '编辑随笔'; submit.textContent = '保存'; $('.mp-del').hidden = false; }
  else loadDraft();
  renderPics(); renderPreview(); renderAuth();
  if (!API) setMsg('未配置 annotations.api，无法发布', true);
  else if (session()) whoAmI().then(function () { if (EDIT && isAuthor()) return loadEntry(); }).catch(function (e) { setMsg(e.message, true); });
  else if (EDIT) setMsg('先登录 GitHub 才能读取并修改这条随笔', true);
})();
