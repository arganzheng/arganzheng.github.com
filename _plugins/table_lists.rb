# Lists inside table cells, done at build time.
#
# kramdown only parses span-level markup inside a table cell, so a literal
# `<ul>` there is escaped to text and a Markdown `- item` is not a list. The
# authoring convention is therefore one item per line with `<br/>`:
#
#   | 章 | 内容 |
#   |---|---|
#   | 二 | embedding 查表<br/>attention 不知道顺序<br/>两种给位置的方法 |
#
# and this hook turns every <td> whose content is `<br />`-separated into a
# real list — `<td><ul class="cell-list"><li>…</li>…</ul></td>` — so the page
# carries list semantics (bullets, screen readers, copy as lines) with or
# without JavaScript. Styling: `.post-container td ul.cell-list` in
# css/github-markdown.css. Only <td>, never <th>: a two-line header
# (`MLIR<br/>Triton`) is a wrapped label, not a list. A <br /> inside
# <code>…</code> cannot occur (kramdown escapes it), so splitting the cell's
# HTML on the tag is safe.
module TableLists
  CELL = %r{(<td\b[^>]*>)((?:(?!</td>).)*?<br\s*/?>(?:(?!</td>).)*)(</td>)}m
  BR = %r{\s*<br\s*/?>\s*}

  def self.process(html)
    html.gsub(CELL) do
      open, body, close = $1, $2, $3
      items = body.split(BR).map(&:strip).reject(&:empty?)
      next "#{open}#{body}#{close}" if items.size < 2
      "#{open}<ul class=\"cell-list\">#{items.map { |i| "<li>#{i}</li>" }.join}</ul>#{close}"
    end
  end
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'
  doc.output = TableLists.process(doc.output)
  doc.content = TableLists.process(doc.content)
end
