# Table captions, Pandoc style: a paragraph right after a table that starts
# with `Table:` (or `表：` / `表1：`, either colon) becomes the table's
# <caption>. kramdown has no syntax for this, so the paragraph is folded
# into the <table> here, at render time, and the page carries a semantic
# caption with or without JavaScript (feed, WeChat export, review pages).
#
#   | a | b |
#   |---|---|
#   | 1 | 2 |
#
#   Table: 各调度器对比
#
# -> <table><caption>各调度器对比</caption>…</table>
#
# A hand-written number (`表 2：`) is dropped: js/figures.js numbers every
# table in document order and shows 「表 N：标题」 under it (「表 N」 when there
# is no caption). Inline markup in the caption is kept.
module TableCaptions
  CAPTION = %r{(<table\b[^>]*>)(\s*(?:<thead|<tbody|<tr|<colgroup|<col\b)[\s\S]*?</table>)\s*<p>\s*(?:Table|表)\s*\d*\s*[:：]\s*([\s\S]*?)\s*</p>}

  def self.process(html)
    html.gsub(CAPTION) { "#{$1}<caption>#{$3}</caption>#{$2}" }
  end
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'
  doc.output = TableCaptions.process(doc.output)
  doc.content = TableCaptions.process(doc.content)
end
