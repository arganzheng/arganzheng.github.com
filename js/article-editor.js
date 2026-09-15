/*
 * Author-only Markdown editor for ordinary post pages.
 * The source Markdown is the saved representation; the rendered article is
 * shown beside it as a stable reference rather than converted back to HTML.
 */
(function () {
  'use strict';

  var host = document.querySelector('.article-editor-host');
  if (!host || !window.BlogAnnotations || !BlogAnnotations.core) return;
  host.hidden = true;

  var core = BlogAnnotations.core;
  var api = (host.getAttribute('data-annotations-api') || '').replace(/\/$/, '');
  var author = host.getAttribute('data-author') || '';
  var sourcePath = host.getAttribute('data-source-path') || '';
  var editor = null;
  var source = null;
  var sha = null;
  var busy = false;

  var openButton = document.createElement('button');
  openButton.type = 'button';
  openButton.className = 'article-editor-open';
  openButton.hidden = true;
  openButton.innerHTML = '<i class="fa fa-pencil"></i> 编辑文章';
  host.appendChild(openButton);

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }

  function setStatus(message, kind) {
    var status = editor && editor.querySelector('.article-editor-status');
    if (!status) return;
    status.textContent = message || '';
    status.className = 'article-editor-status' + (kind ? ' is-' + kind : '');
  }

  function setBusy(value) {
    busy = value;
    if (!editor) return;
    editor.querySelector('.article-editor-save').disabled = value;
    editor.querySelector('.article-editor-reload').disabled = value;
    editor.querySelector('.article-editor-text').disabled = value;
    editor.classList.toggle('is-busy', value);
  }

  function buildEditor() {
    if (editor) return editor;
    editor = document.createElement('section');
    editor.className = 'article-editor';
    editor.hidden = true;
    editor.setAttribute('aria-label', '文章 Markdown 编辑器');
    editor.innerHTML =
      '<div class="article-editor-head">' +
        '<div><strong>编辑文章</strong><span class="article-editor-path"></span></div>' +
        '<button type="button" class="article-editor-close" aria-label="关闭编辑器">×</button>' +
      '</div>' +
      '<p class="article-editor-hint">左侧保存的是原始 Markdown；右侧是当前线上页面参考。保存会直接提交到 master，并触发网站重新构建。</p>' +
      '<div class="article-editor-grid">' +
        '<div class="article-editor-source"><label for="article-editor-text">Markdown</label><textarea id="article-editor-text" class="article-editor-text" spellcheck="false"></textarea></div>' +
        '<div class="article-editor-reference"><label>当前页面参考</label><div class="article-editor-reference-body"></div></div>' +
      '</div>' +
      '<div class="article-editor-foot"><span class="article-editor-status" role="status" aria-live="polite"></span><div><button type="button" class="article-editor-reload">重新加载</button><button type="button" class="article-editor-cancel">取消</button><button type="button" class="article-editor-save">保存到 master</button></div></div>';
    host.appendChild(editor);

    editor.querySelector('.article-editor-path').textContent = ' · ' + sourcePath;
    editor.querySelector('.article-editor-close').addEventListener('click', close);
    editor.querySelector('.article-editor-cancel').addEventListener('click', close);
    editor.querySelector('.article-editor-reload').addEventListener('click', loadSource);
    editor.querySelector('.article-editor-save').addEventListener('click', saveSource);
    return editor;
  }

  function renderReference() {
    var target = editor.querySelector('.article-editor-reference-body');
    var current = document.querySelector('.post-container');
    if (!current) return;
    var clone = current.cloneNode(true);
    var junk = clone.querySelectorAll('.series-nav, .series-context, .series-toc, .pager, .related-posts, .post-license, .post-actions, .comment, .annotation-panel, .annotation-marker, .annotation-toolbar, .sec-react, script, style, noscript');
    for (var i = 0; i < junk.length; i++) junk[i].parentNode.removeChild(junk[i]);
    target.innerHTML = '';
    target.appendChild(clone);
  }

  function request(path, options) {
    options = options || {};
    return core.ensureToken().then(function (token) {
      var headers = { Authorization: 'Bearer ' + token };
      if (options.body) headers['Content-Type'] = 'application/json';
      return fetch(api + path, {
        method: options.method || 'GET',
        headers: headers,
        body: options.body ? JSON.stringify(options.body) : undefined
      });
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) {
          var error = new Error((data && data.error) || ('HTTP ' + response.status));
          error.status = response.status;
          throw error;
        }
        return data;
      });
    });
  }

  function loadSource() {
    if (busy) return Promise.resolve();
    setBusy(true);
    setStatus('正在读取文章源码…');
    return request('/article?path=' + encodeURIComponent(sourcePath)).then(function (data) {
      source = data.content;
      sha = data.sha;
      editor.querySelector('.article-editor-text').value = source;
      setStatus('已加载当前版本。');
    }).catch(function (error) {
      setStatus(error.message, 'error');
    }).then(function () { setBusy(false); });
  }

  function saveSource() {
    if (busy) return;
    var text = editor.querySelector('.article-editor-text').value;
    if (text === source) {
      setStatus('没有检测到修改。');
      return;
    }
    if (!window.confirm('这会直接提交到 master，并触发网站重新构建。确定保存吗？')) return;
    setBusy(true);
    setStatus('正在保存到 master…');
    request('/article', { method: 'PUT', body: { path: sourcePath, sha: sha, content: text } }).then(function (data) {
      source = text;
      sha = data.sha || sha;
      setStatus('已提交。等待 GitHub Pages 构建后刷新页面。', 'success');
    }).catch(function (error) {
      setStatus(error.message + (error.status === 409 ? '（请先重新加载）' : ''), 'error');
    }).then(function () { setBusy(false); });
  }

  function open() {
    if (busy) return;
    buildEditor();
    editor.hidden = false;
    openButton.setAttribute('aria-expanded', 'true');
    renderReference();
    if (source === null) loadSource();
    else editor.querySelector('.article-editor-text').focus();
  }

  function close() {
    if (!editor || busy) return;
    editor.hidden = true;
    openButton.setAttribute('aria-expanded', 'false');
  }

  function updateVisibility(user) {
    var allowed = !!(user && author && user.login === author && api && /^_posts\//.test(sourcePath));
    host.hidden = !allowed;
    openButton.hidden = !allowed;
  }

  openButton.addEventListener('click', open);
  document.addEventListener('blog:viewer', function (event) { updateVisibility(event.detail); });
  updateVisibility(BlogAnnotations.viewer());
})();
