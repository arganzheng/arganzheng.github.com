/*!
 * moment-post.js — the phone 发布页 for 随笔 (/moments/post.html; not part of
 * blog.min.js). Login = the comments' giscus OAuth (localStorage
 * "giscus-session" → worker POST /token → GitHub token); the worker's
 * POST /moments accepts the request only when GET /user is the repo owner.
 * Pictures are shrunk to ≤ 1600px wide in a canvas and saved in IndexedDB with
 * the text draft; offline publishes are queued until the worker is reachable.
 */
(function () {
  var root = document.getElementById('mp');
  if (!root) return;
  var API = (root.getAttribute('data-api') || '').replace(/\/$/, '');
  var OWNER = (root.getAttribute('data-owner') || '').toLowerCase();
  var BASE = root.getAttribute('data-base') || '';
  var SESSION_KEY = 'giscus-session', DRAFT_KEY = 'moment-post-draft', QUEUE_DB = 'moment-queue';
  var PENDING_KEY = 'moments-pending';
  var MAX_PICS = 9, MAX_W = 1600;
  // ?edit=YYYY-MM/<id> (the 编辑 link on a card): load that entry from the worker, PUT it back.
  var EDIT = /[?&]edit=(\d{4}-\d{2})\/(\d{8}(?:-\d{4})?(?:-\d+)?)/.exec(location.search);
  EDIT = EDIT ? { month: EDIT[1], id: EDIT[2] } : null;
  var REF = new URLSearchParams(location.search).get('ref');
  REF = REF && /^\d{8}(?:-\d{4})?(?:-\d+)?$/.test(REF) ? REF : null;
  var dateOnly = false, timeTouched = false;
  var TAG = /(^|[^\p{L}\p{N}_\/&\\])#([\p{L}_][\p{L}\p{N}_\-·]*(?:\/[\p{L}\p{N}_\-·]+)*)/gu;

  function $(s) { return root.querySelector(s); }
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function dbOpen(name, store) {
    return new Promise(function (resolve, reject) {
      var request = indexedDB.open(name, 1);
      request.onupgradeneeded = function () { if (!request.result.objectStoreNames.contains(store)) request.result.createObjectStore(store); };
      request.onsuccess = function () { resolve(request.result); };
      request.onerror = function () { reject(request.error); };
    });
  }
  function dbAction(name, store, mode, method, key, value) {
    return dbOpen(name, store).then(function (db) {
      return new Promise(function (resolve, reject) {
        var tx = db.transaction(store, mode), objectStore = tx.objectStore(store);
        var request = method === 'put' ? objectStore.put(value, key) : objectStore[method](key);
        request.onsuccess = function () { resolve(request.result); };
        request.onerror = function () { reject(request.error); };
        tx.oncomplete = function () { db.close(); };
        tx.onerror = function () { db.close(); reject(tx.error); };
      });
    });
  }

  var text = $('.mp-text'), file = $('.mp-file'), pics = $('.mp-pics'), tagsBox = $('.mp-tags'), tagNew = $('.mp-tag-new');
  var place = $('.mp-place'), quote = $('.mp-quote'), by = $('.mp-by'), music = $('.mp-music'), time = $('.mp-time');
  var submit = $('.mp-submit'), msg = $('.mp-msg'), loginBtn = $('.mp-login'), who = $('.mp-who');
  var done = $('.mp-done'), editor = $('.mp-editor'), queueNote = $('.mp-queue');
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
      .catch(function (error) { error.network = true; throw error; })
      .then(function (r) { return r.json().catch(function () { return {}; }).then(function (d) { if (!r.ok) { var error = new Error(d.error || ('HTTP ' + r.status)); error.status = r.status; throw error; } return d; }); });
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
  function clearDraft() {
    try { localStorage.removeItem(DRAFT_KEY); } catch (e) { /* ignore */ }
    dbAction('moment-post', 'draft', 'readwrite', 'delete', 'pictures').catch(function () {});
  }
  function persistPictures() {
    return dbAction('moment-post', 'draft', 'readwrite', 'put', 'pictures', images.filter(function (image) { return image.blob; }).map(function (image) { return image.blob; })).catch(function () {});
  }
  function restorePictures() {
    return dbAction('moment-post', 'draft', 'readonly', 'get', 'pictures').then(function (blobs) {
      images = (blobs || []).slice(0, MAX_PICS).map(function (blob) { return { blob: blob, url: URL.createObjectURL(blob) }; });
    }).catch(function () {});
  }

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
  function addFiles(list) {
    list = Array.prototype.slice.call(list || []);
    if (!list.length) return Promise.resolve();
    if (images.length + list.length > MAX_PICS) { setMsg('最多 ' + MAX_PICS + ' 张图', true); list = list.slice(0, MAX_PICS - images.length); }
    if (!list.length) return Promise.resolve();
    busy = true; updateSubmit(); setMsg('处理图片…');
    var chain = Promise.resolve();
    list.forEach(function (f) {
      chain = chain.then(function () { return shrink(f); }).then(function (b) {
        images.push({ blob: b, url: URL.createObjectURL(b) }); renderPics(); renderPreview();
      }).catch(function (e) { setMsg(e.message, true); });
    });
    return chain.then(function () { busy = false; setMsg(''); persistPictures(); renderPics(); renderPreview(); updateSubmit(); });
  }
  file.addEventListener('change', function () {
    var list = Array.prototype.slice.call(file.files || []);
    file.value = '';
    addFiles(list);
  });
  function importShare() {
    if (!new URLSearchParams(location.search).has('shared')) return Promise.resolve();
    return dbAction('moments-share', 'share', 'readwrite', 'get', 'latest').then(function (shared) {
      if (!shared) return;
      var fields = [shared.title, shared.text].filter(Boolean);
      if (shared.url) fields.push(shared.url);
      if (fields.length) text.value = [text.value.trim(), fields.join('\n')].filter(Boolean).join('\n\n');
      saveDraft();
      var files = (shared.files || []).map(function (item) { return new File([item.blob], item.name || 'shared-image', { type: item.type || item.blob.type }); });
      return addFiles(files).then(function () { return dbAction('moments-share', 'share', 'readwrite', 'delete', 'latest'); });
    }).catch(function () {}).then(function () {
      var url = new URL(location.href); url.searchParams.delete('shared');
      history.replaceState(history.state, '', url.toString());
    });
  }
  // ---- picture grid (朋友圈): tap = full-screen preview, long-press (or mouse drag)
  // picks a picture up, the others make room live; dropping it on the red bar deletes it.
  var trash = document.createElement('div');
  trash.className = 'mp-trash'; trash.hidden = true;
  trash.innerHTML = '<i class="fa fa-trash"></i><span>拖到此处删除</span>';
  document.body.appendChild(trash);
  function removePic(i) {
    if (images[i].blob) URL.revokeObjectURL(images[i].url);
    images.splice(i, 1);
    persistPictures(); renderPics(); renderPreview(); updateSubmit();
  }
  function renderPics() {
    Array.prototype.forEach.call(pics.querySelectorAll('.mp-pic'), function (n) { n.remove(); });
    var add = pics.querySelector('.mp-add');
    images.forEach(function (im, i) {
      var d = document.createElement('div'); d.className = 'mp-pic'; d.setAttribute('data-i', i);
      d.setAttribute('role', 'button'); d.tabIndex = 0; d.title = '点按预览，长按拖动排序';
      d.innerHTML = '<img src="' + im.url + '" alt="" draggable="false">';
      pics.insertBefore(d, add);
    });
    add.hidden = images.length >= MAX_PICS;
  }
  function movePic(from, to) {
    if (to < 0 || to >= images.length || from === to) return;
    images.splice(to, 0, images.splice(from, 1)[0]);
    persistPictures(); renderPics(); renderPreview();
  }

  var drag = null;   // { from, to, el, ghost, x, y, ox, oy, touch, active, timer }
  function dragBegin(el, x, y, touch) {
    if (drag) return;
    drag = { from: +el.getAttribute('data-i'), el: el, x: x, y: y, touch: touch, active: false };
    if (touch) drag.timer = setTimeout(dragLift, 350);
  }
  function dragLift() {
    var d = drag, r = d.el.getBoundingClientRect();
    d.active = true; d.to = d.from;
    d.ox = d.x - r.left; d.oy = d.y - r.top;
    d.ghost = d.el.cloneNode(true); d.ghost.className = 'mp-ghost';
    d.ghost.style.width = r.width + 'px'; d.ghost.style.height = r.height + 'px';
    document.body.appendChild(d.ghost); ghostAt(d.x, d.y);
    d.el.classList.add('is-holding');
    trash.hidden = false; trash.getBoundingClientRect(); trash.classList.add('is-on');
    document.documentElement.classList.add('mp-dragging');
    if (navigator.vibrate) navigator.vibrate(15);
  }
  function ghostAt(x, y) { drag.ghost.style.transform = 'translate(' + (x - drag.ox) + 'px,' + (y - drag.oy) + 'px) scale(1.08)'; }
  function overTrash(y) { return y >= window.innerHeight - trash.offsetHeight; }
  // Hit-test with offsetLeft/Top (layout boxes, not the FLIP transforms in flight).
  function reflow(x, y) {
    var cells = Array.prototype.slice.call(pics.querySelectorAll('.mp-pic'));
    var box = pics.getBoundingClientRect(), px = x - box.left, py = y - box.top;
    var t = -1;
    cells.forEach(function (c, i) {
      if (px >= c.offsetLeft && px < c.offsetLeft + c.offsetWidth && py >= c.offsetTop && py < c.offsetTop + c.offsetHeight) t = i;
    });
    var cur = cells.indexOf(drag.el);
    if (t < 0 || t === cur) return;
    var before = cells.map(function (c) { return [c.offsetLeft, c.offsetTop]; });
    pics.insertBefore(drag.el, t > cur ? cells[t].nextSibling : cells[t]);
    cells.forEach(function (c, i) {
      var dx = before[i][0] - c.offsetLeft, dy = before[i][1] - c.offsetTop;
      if (!dx && !dy) return;
      c.style.transition = 'none'; c.style.transform = 'translate(' + dx + 'px,' + dy + 'px)';
      c.getBoundingClientRect();
      c.style.transition = ''; c.style.transform = '';
    });
    drag.to = t;
  }
  function dragMove(x, y, e) {
    if (!drag) return;
    if (!drag.active) {
      if (Math.abs(x - drag.x) + Math.abs(y - drag.y) < 8) return;
      if (drag.touch) { clearTimeout(drag.timer); drag = null; return; }   // a scroll, not a long-press
      dragLift();
    }
    if (e.cancelable) e.preventDefault();
    drag.y = y; ghostAt(x, y);
    var del = overTrash(y);
    trash.classList.toggle('is-hot', del);
    trash.querySelector('span').textContent = del ? '松手即可删除' : '拖到此处删除';
    if (!del) reflow(x, y);
  }
  function dragEnd(cancelled) {
    if (!drag) return;
    var d = drag; drag = null; clearTimeout(d.timer);
    if (!d.active) { if (!cancelled) openViewer(d.from); return; }
    d.ghost.remove();
    document.documentElement.classList.remove('mp-dragging');
    trash.classList.remove('is-on', 'is-hot');
    setTimeout(function () { if (!drag) trash.hidden = true; }, 200);
    if (!cancelled && overTrash(d.y)) return removePic(d.from);
    if (d.to !== d.from) movePic(d.from, d.to); else renderPics();
  }
  pics.addEventListener('contextmenu', function (e) { if (e.target.closest('.mp-pic')) e.preventDefault(); });
  pics.addEventListener('touchstart', function (e) {
    var p = e.target.closest('.mp-pic'); if (!p || e.touches.length > 1) return;
    dragBegin(p, e.touches[0].clientX, e.touches[0].clientY, true);
  }, { passive: true });
  document.addEventListener('touchmove', function (e) { if (drag && drag.touch) dragMove(e.touches[0].clientX, e.touches[0].clientY, e); }, { passive: false });
  document.addEventListener('touchend', function (e) { if (drag && drag.touch) { if (e.cancelable) e.preventDefault(); dragEnd(false); } });
  document.addEventListener('touchcancel', function () { if (drag && drag.touch) dragEnd(true); });
  pics.addEventListener('mousedown', function (e) {
    var p = e.target.closest('.mp-pic'); if (!p || e.button !== 0) return;
    e.preventDefault(); dragBegin(p, e.clientX, e.clientY, false);
  });
  document.addEventListener('mousemove', function (e) { if (drag && !drag.touch) dragMove(e.clientX, e.clientY, e); });
  document.addEventListener('mouseup', function () { if (drag && !drag.touch) dragEnd(false); });
  pics.addEventListener('keydown', function (e) {
    var p = e.target.closest('.mp-pic');
    if (p && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openViewer(+p.getAttribute('data-i')); }
  });

  // ---- full-screen preview with 删除 (朋友圈's 预览)
  var viewer = document.createElement('div'), vi = 0, vx = null;
  viewer.className = 'mp-viewer'; viewer.hidden = true;
  viewer.innerHTML = '<img alt=""><span class="mp-viewer-n"></span>' +
    '<button type="button" class="mp-viewer-del" title="删除这张"><i class="fa fa-trash"></i></button>';
  document.body.appendChild(viewer);
  function showViewer() {
    viewer.querySelector('img').src = images[vi].url;
    viewer.querySelector('.mp-viewer-n').textContent = images.length > 1 ? (vi + 1) + ' / ' + images.length : '';
  }
  function openViewer(i) {
    if (!images[i]) return;
    vi = i; showViewer(); viewer.hidden = false;
    document.documentElement.classList.add('mp-viewing');
  }
  function closeViewer() { viewer.hidden = true; document.documentElement.classList.remove('mp-viewing'); }
  function stepViewer(n) { if (vi + n >= 0 && vi + n < images.length) { vi += n; showViewer(); } }
  viewer.addEventListener('click', function (e) {
    if (vx === 'swiped') { vx = null; return; }
    if (!e.target.closest('.mp-viewer-del')) return closeViewer();
    if (!confirm('要删除这张照片吗？')) return;
    removePic(vi);
    if (!images.length) return closeViewer();
    vi = Math.min(vi, images.length - 1); showViewer();
  });
  viewer.addEventListener('touchstart', function (e) { vx = e.touches[0].clientX; }, { passive: true });
  viewer.addEventListener('touchend', function (e) {
    var dx = typeof vx === 'number' ? e.changedTouches[0].clientX - vx : 0;
    vx = Math.abs(dx) > 50 ? 'swiped' : null;
    if (vx) stepViewer(dx < 0 ? 1 : -1);
  });
  document.addEventListener('keydown', function (e) {
    if (viewer.hidden) return;
    if (e.key === 'Escape') closeViewer();
    else if (e.key === 'ArrowLeft') stepViewer(-1);
    else if (e.key === 'ArrowRight') stepViewer(1);
  });

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
      h = h.split(/(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`]*`)/g).map(function (part, i) {
        return i % 2 ? part : part.replace(/\[\[(\d{8}(?:-\d{4})?(?:-\d+)?)\]\]/g, '<a class="moment-ref"><span class="moment-ref-when">$1</span></a>');
      }).join('');
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

  function tinyThumb(blob) {
    return new Promise(function (resolve) {
      var url = URL.createObjectURL(blob), img = new Image();
      img.onload = function () {
        var scale = Math.min(1, 240 / img.naturalWidth), canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(img.naturalWidth * scale)); canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
        canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height); URL.revokeObjectURL(url);
        canvas.toBlob(function (small) {
          if (!small) return resolve('');
          var reader = new FileReader(); reader.onload = function () { resolve(String(reader.result)); }; reader.onerror = function () { resolve(''); }; reader.readAsDataURL(small);
        }, 'image/jpeg', 0.72);
      };
      img.onerror = function () { URL.revokeObjectURL(url); resolve(''); }; img.src = url;
    });
  }
  function savePending(payload, response) {
    var thumbs = (payload.images || []).filter(function (image) { return image.data; }).slice(0, 3).map(function (image) {
      var bytes = atob(image.data), array = new Uint8Array(bytes.length);
      for (var i = 0; i < bytes.length; i++) array[i] = bytes.charCodeAt(i);
      return tinyThumb(new Blob([array], { type: image.type || 'image/jpeg' }));
    });
    return Promise.all(thumbs).then(function (imagesData) {
      var pending = [];
      try { pending = JSON.parse(localStorage.getItem(PENDING_KEY) || '[]'); } catch (e) { pending = []; }
      var entry = {
        id: (response.url || '').split('#').pop(), month: response.month, url: response.url,
        title: payload.time, place: payload.place, text: payload.text, tags: payload.tags || [],
        thumbs: imagesData.filter(Boolean), at: Date.now()
      };
      pending = pending.filter(function (item) { return item.id !== entry.id; });
      pending.unshift(entry);
      try { localStorage.setItem(PENDING_KEY, JSON.stringify(pending.slice(0, 5))); } catch (e) { /* ignore */ }
    });
  }
  function completePublish(payload, response) {
    var stored = EDIT ? Promise.resolve() : savePending(payload, response).catch(function () {});
    return stored.then(function () {
      if (!EDIT) clearDraft();
      $('.mp-done-title').innerHTML = '<i class="fa fa-check"></i> ' + (EDIT ? '已更新' : '已提交');
      $('.mp-done-link').href = BASE + response.url;
      $('.mp-done-month').href = BASE + response.month;
      $('.mp-done-month').hidden = !response.month || !!EDIT;
      $('.mp-done-commit').href = 'https://github.com/' + (root.getAttribute('data-repo') || '') + '/commit/' + response.commit;
      $('.mp-done-commit').hidden = !root.getAttribute('data-repo');
      editor.hidden = true; done.hidden = false; $('.mp-preview-wrap').hidden = true;
      busy = false; setMsg('');
    });
  }

  var queueItems = [], queueLoaded = false, queueBusy = false;
  function showQueue(error) {
    queueNote.hidden = !queueItems.length && !error;
    if (!queueNote.hidden) {
      queueNote.innerHTML = esc(error ? error + '（' + queueItems.length + ' 条待发）' : (queueItems.length + ' 条待发，联网后自动发布')) +
        (error ? ' <button type="button" class="mp-queue-discard">丢弃</button>' : '');
    }
  }
  function loadQueue() {
    return dbAction(QUEUE_DB, 'queue', 'readonly', 'get', 'items').then(function (items) {
      queueItems = items || []; queueLoaded = true; showQueue();
    }).catch(function () { queueLoaded = true; });
  }
  function saveQueue() {
    return dbAction(QUEUE_DB, 'queue', 'readwrite', 'put', 'items', queueItems);
  }
  function queuePayload(payload) {
    queueItems.push({ key: Date.now() + Math.random(), payload: payload });
    return saveQueue().then(function () {
      showQueue(); setMsg('已离线保存，联网后自动发布（' + queueItems.length + ' 条待发）');
      busy = false; updateSubmit();
    }).catch(function (error) {
      queueItems.pop(); busy = false; updateSubmit();
      setMsg('离线保存失败：' + error.message, true);
    });
  }
  function processQueue() {
    if (!queueLoaded || queueBusy || !navigator.onLine || !queueItems.length || EDIT) return;
    queueBusy = true;
    function next() {
      if (!queueItems.length) { queueBusy = false; showQueue(); return; }
      token = null;
      ensureToken().then(function (fresh) {
        return api('/moments', { method: 'POST', token: fresh, body: queueItems[0].payload });
      }).then(function (response) {
        var current = queueItems[0];
        return savePending(current.payload, response).then(function () {
          queueItems.shift(); return saveQueue();
        }).catch(function (error) {
          if (queueItems[0] !== current) queueItems.unshift(current);
          throw error;
        }).then(next);
      }).catch(function (error) {
        queueBusy = false;
        showQueue(error.status >= 400 && error.status < 500 ? error.message : '');
      });
    }
    next();
  }
  queueNote.addEventListener('click', function (event) {
    if (!event.target.closest('.mp-queue-discard')) return;
    queueItems.shift(); saveQueue().then(function () { showQueue(); processQueue(); });
  });
  window.addEventListener('online', processQueue);

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
      .then(function (imgs) {
        payload.images = imgs;
        if (!EDIT && !navigator.onLine) return queuePayload(payload);
        return ensureToken().then(function (t) {
          return api('/moments', { method: EDIT ? 'PUT' : 'POST', token: t, body: payload });
        }).then(function (response) { return completePublish(payload, response); });
      })
      .catch(function (e) {
        if (!EDIT && (e.network || !navigator.onLine)) return queuePayload(payload);
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
    clearDraft();
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
  else {
    loadDraft();
    if (REF && text.value.indexOf('[[' + REF + ']]') < 0) { text.value = '[[' + REF + ']] ' + text.value; saveDraft(); }
  }
  function finishBoot() {
    renderPics(); renderPreview(); renderAuth();
    if (!API) setMsg('未配置 annotations.api，无法发布', true);
    else if (session()) whoAmI().then(function () { if (EDIT && isAuthor()) return loadEntry(); }).catch(function (e) { setMsg(e.message, true); });
    else if (EDIT) setMsg('先登录 GitHub 才能读取并修改这条随笔', true);
    loadQueue().then(processQueue);
  }
  if (!EDIT) restorePictures().then(importShare).then(finishBoot);
  else finishBoot();
})();
