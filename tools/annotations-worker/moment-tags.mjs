export function rewriteMomentTags(markdown, from, to) {
  const escaped = from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const tag = new RegExp(`(?<![\\p{L}\\p{N}_/&\\\\])#${escaped}((?:/[\\p{L}\\p{N}_\\-·]+)*)(?![\\p{L}\\p{N}_\\-·/])`, 'gu');
  const lines = markdown.match(/[^\n]*\n|[^\n]+$/g) || [];
  const output = [];
  let fence = null, inEntry = false, entryChanged = false, changed = 0;

  for (const line of lines) {
    if (/^##\s+\d{4}-\d{2}-\d{2}(?:\s|$)/.test(line)) {
      if (inEntry && entryChanged) changed++;
      inEntry = true;
      entryChanged = false;
    }
    const marker = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (marker) {
      const char = marker[1][0], length = marker[1].length;
      if (!fence) fence = { char, length };
      else if (fence.char === char && length >= fence.length) fence = null;
      output.push(line);
      continue;
    }
    if (!inEntry || fence || /^\s{4}/.test(line)) {
      output.push(line);
      continue;
    }
    const rewritten = line.split(/(`[^`]*`|https?:\/\/[^\s<>]+)/gi).map((part, index) => {
      if (index % 2) return part;
      return part.replace(tag, (_, child) => `#${to}${child}`);
    }).join('');
    if (rewritten !== line) entryChanged = true;
    output.push(rewritten);
  }
  if (inEntry && entryChanged) changed++;
  return { content: output.join(''), changed };
}
