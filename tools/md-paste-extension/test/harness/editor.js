const editorKind = new URLSearchParams(location.search).get('editor') || 'textarea';
const textarea = document.getElementById('textarea');
const cmHost = document.getElementById('cm-editor');
const contenteditable = document.getElementById('contenteditable');
const preview = document.getElementById('preview');
const fileLabel = document.getElementById('file-label');
const uploadedFiles = [];
const pasteEvents = [];
let codeMirror = null;

function isTextareaEditor() {
  return ['textarea', 'trusted-only', 'async-unprevent', 'prevented-no-change'].includes(editorKind);
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  })[character]);
}

function renderMarkdown(value) {
  const blocks = [];
  let html = escapeHtml(value);
  html = html.replace(/^ {0,3}(`{3,}|~{3,})[^\n]*\n([\s\S]*?)^ {0,3}\1[ \t]*$/gm, (_match, _fence, code) => {
    const index = blocks.push('<pre><code>' + code + '</code></pre>') - 1;
    return '\u0000' + index + '\u0000';
  });
  html = html.replace(/^#{1,6}\s+(.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/!\[([^\]]*)\]\((blob:[^)]+)\)/g, '<img alt="$1" src="$2">');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\n/g, '<br>');
  html = html.replace(/\u0000(\d+)\u0000/g, (_match, index) => blocks[Number(index)]);
  preview.innerHTML = html;
}

function serializeEditable(node) {
  return Array.from(node.childNodes).map((child) => {
    if (child.nodeType === Node.TEXT_NODE) return child.nodeValue;
    if (child.nodeType !== Node.ELEMENT_NODE) return '';
    if (child.tagName === 'IMG') return '![' + (child.alt || '') + '](' + child.src + ')';
    if (child.tagName === 'BR') return '\n';
    const content = serializeEditable(child);
    return /^(DIV|P|PRE)$/.test(child.tagName) ? content + '\n' : content;
  }).join('');
}

function getSource() {
  if (isTextareaEditor()) return textarea.value;
  if (editorKind === 'codemirror6') return codeMirror.state.doc.toString();
  return serializeEditable(contenteditable).replace(/\n+$/, '');
}

function updatePreview() {
  if (editorKind === 'contenteditable') {
    preview.innerHTML = contenteditable.innerHTML;
    return;
  }
  renderMarkdown(getSource());
}

function setRangeText(element, text) {
  const start = element.selectionStart;
  const end = element.selectionEnd;
  element.setRangeText(text, start, end, 'end');
  element.dispatchEvent(new Event('input', { bubbles: true }));
}

function insertFileMarker(element, file, delay) {
  const marker = '![上传中…]()';
  const start = element.selectionStart;
  setRangeText(element, marker);
  setTimeout(() => {
    const position = element.value.indexOf(marker, start);
    if (position < 0) return;
    const replacement = '![' + file.name + '](' + URL.createObjectURL(file) + ')';
    element.setRangeText(replacement, position, position + marker.length, 'end');
    element.dispatchEvent(new Event('input', { bubbles: true }));
    updatePreview();
  }, delay || 600);
}

function onTextareaPaste(event) {
  const file = event.clipboardData && event.clipboardData.files[0];
  pasteEvents.push({
    type: 'paste',
    isTrusted: event.isTrusted,
    files: event.clipboardData ? event.clipboardData.files.length : 0
  });
  if (editorKind === 'trusted-only' && !event.isTrusted) return;
  if (editorKind === 'prevented-no-change') {
    event.preventDefault();
    return;
  }
  if (!file) return;
  if (editorKind !== 'async-unprevent') event.preventDefault();
  uploadedFiles.push({ name: file.name, type: file.type, isTrusted: event.isTrusted });
  insertFileMarker(textarea, file, editorKind === 'async-unprevent' ? 500 : 600);
}

function onTextareaDrop(event) {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  pasteEvents.push({
    type: event.type,
    isTrusted: event.isTrusted,
    files: event.dataTransfer ? event.dataTransfer.files.length : 0
  });
  if (editorKind === 'trusted-only' && !event.isTrusted) return;
  if (!file) return;
  event.preventDefault();
  uploadedFiles.push({ name: file.name, type: file.type, isTrusted: event.isTrusted });
  insertFileMarker(textarea, file);
}

function onEditablePaste(event) {
  const file = event.clipboardData && event.clipboardData.files[0];
  if (!file) return;
  event.preventDefault();
  uploadedFiles.push({ name: file.name, type: file.type });
  const placeholder = document.createElement('span');
  placeholder.className = 'uploading';
  placeholder.textContent = '上传中…';
  const selection = window.getSelection();
  if (selection && selection.rangeCount && contenteditable.contains(selection.anchorNode)) {
    const range = selection.getRangeAt(0);
    range.deleteContents();
    range.insertNode(placeholder);
    range.setStartAfter(placeholder);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
  } else {
    contenteditable.appendChild(placeholder);
  }
  setTimeout(() => {
    const image = document.createElement('img');
    image.alt = file.name;
    image.src = URL.createObjectURL(file);
    placeholder.replaceWith(image);
    const range = document.createRange();
    range.setStartAfter(image);
    range.collapse(true);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    contenteditable.dispatchEvent(new InputEvent('input', {
      bubbles: true,
      inputType: 'insertReplacementText'
    }));
    updatePreview();
  }, 600);
}

async function initialize() {
  document.getElementById('editor-title').textContent =
    editorKind === 'codemirror6' ? 'CodeMirror 6' :
      editorKind === 'contenteditable' ? 'ProseMirror-like contenteditable' : 'textarea';

  if (isTextareaEditor()) {
    textarea.hidden = false;
    textarea.addEventListener('paste', onTextareaPaste);
    if (editorKind === 'trusted-only') textarea.addEventListener('drop', onTextareaDrop);
    textarea.addEventListener('input', updatePreview);
    fileLabel.textContent = editorKind === 'trusted-only' ?
      '图片：只接受可信的 paste/drop 事件' :
      editorKind === 'async-unprevent' ?
        '图片：异步处理 paste 文件，不调用 preventDefault' :
        editorKind === 'prevented-no-change' ?
          '图片：阻止 paste 但不修改编辑器' :
        '文本：粘贴事件未处理时使用 insertText；图片：paste 文件事件由编辑器处理';
  } else if (editorKind === 'codemirror6') {
    textarea.hidden = true;
    cmHost.hidden = false;
    const { EditorView, basicSetup } = await import('https://esm.sh/codemirror@6.0.1?bundle');
    codeMirror = new EditorView({
      parent: cmHost,
      doc: '',
      extensions: [
        basicSetup,
        EditorView.domEventHandlers({
          paste(event, view) {
            const file = event.clipboardData && event.clipboardData.files[0];
            if (file) {
              event.preventDefault();
              uploadedFiles.push({ name: file.name, type: file.type });
              const position = view.state.selection.main.head;
              const marker = '![上传中…]()';
              view.dispatch({
                changes: { from: view.state.selection.main.from, to: view.state.selection.main.to, insert: marker },
                selection: { anchor: position + marker.length }
              });
              setTimeout(() => {
                const current = view.state.doc.toString();
                const found = current.indexOf(marker, position);
                if (found < 0) return;
                const replacement = '![' + file.name + '](' + URL.createObjectURL(file) + ')';
                view.dispatch({ changes: { from: found, to: found + marker.length, insert: replacement } });
                updatePreview();
              }, 600);
              return true;
            }
            const text = event.clipboardData && event.clipboardData.getData('text/plain');
            if (!text) return false;
            event.preventDefault();
            view.dispatch({
              changes: {
                from: view.state.selection.main.from,
                to: view.state.selection.main.to,
                insert: text
              },
              selection: { anchor: view.state.selection.main.from + text.length }
            });
            return true;
          }
        }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) updatePreview();
        })
      ]
    });
    fileLabel.textContent = '文本：CodeMirror 默认 insertText；图片：paste 文件事件由编辑器处理';
  } else if (editorKind === 'contenteditable') {
    textarea.hidden = true;
    contenteditable.hidden = false;
    contenteditable.addEventListener('paste', onEditablePaste);
    contenteditable.addEventListener('input', updatePreview);
    fileLabel.textContent = '文本：contenteditable insertText；图片：paste 文件事件插入 blob 图片';
  } else {
    throw new Error('unknown editor: ' + editorKind);
  }

  window.__mdPasteHarness = {
    getSource,
    getUploadedFiles: () => uploadedFiles.slice(),
    getPasteEvents: () => pasteEvents.slice(),
    focus: () => {
      if (isTextareaEditor()) textarea.focus();
      else if (editorKind === 'codemirror6') codeMirror.focus();
      else contenteditable.focus();
    }
  };
  window.__mdPasteHarnessReady = true;
  updatePreview();
}

initialize().catch((error) => {
  document.getElementById('editor-title').textContent = '初始化失败';
  fileLabel.textContent = String(error);
  window.__mdPasteHarnessError = String(error);
});
