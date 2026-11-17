# Titles on fenced blocks, MDX / VitePress style:
#
#     ```cpp title="Dispatcher::call 的完整签名"
#     ...
#     ```
#
# kramdown's GFM fence only takes a language (`\S+` then end of line), so a
# `title=` after it would turn the whole block into a paragraph. This hook runs
# before rendering and moves the title onto the kramdown block IAL that
# precedes the fence — `{: data-title="…"}` — merged into an existing one
# (`{:.no-lineno}` → `{:.no-lineno data-title="…"}`), with a blank line before
# it when the fence directly follows text (an IAL right after a paragraph line
# belongs to the paragraph). kramdown puts the attribute on the rouge wrapper
# (`<div data-title class="language-x highlighter-rouge">`) or, for an untyped
# fence, on the `<pre>`; js/figures.js turns it into the block's header bar
# and the passage 划线评论 anchor on.
#
# Fences inside a longer fence (a ```` block showing the syntax itself) are
# content and left alone: an open fence is only closed by the same character,
# at least as long, on a line of its own.
#
# A fence inside a blockquote — a `> [!NOTE]` callout, nested quotes too —
# carries the quote markers as its line prefix (`> ```java title="…"`). The
# prefix is kept on the IAL line and on the blank line inserted before it, so
# both stay inside the quote; a fence is only closed by a line at the same
# quote depth. Until 2026-11-17 the prefix was not recognised and such a
# block rendered as a paragraph of raw text (4 places in the Python series).
module CodeTitles
  PREFIX = /(?:[ \t]*>)*[ \t]*/
  FENCE = /\A(#{PREFIX})(`{3,}|~{3,})(.*)\z/
  TITLE = /\A\s*([^\s`]*)\s+title=(?:"([^"]*)"|'([^']*)')\s*\z/
  IAL = /\A(#{PREFIX})\{:(.*)\}\s*\z/
  QUOTE_MARKS = /\A(?:[ \t]*>)*/

  def self.process(text)
    out = []
    open = nil # [char, length, quote depth] of the fence we are inside
    text.each_line do |raw|
      line = raw.chomp
      m = FENCE.match(line)
      if open
        out << raw
        open = nil if m && m[2][0] == open[0] && m[2].size >= open[1] && m[3].strip.empty? &&
                      m[1].count('>') == open[2]
        next
      end
      unless m
        out << raw
        next
      end
      indent, fence, info = m[1], m[2], m[3]
      open = [fence[0], fence.size, indent.count('>')]
      t = TITLE.match(info)
      unless t
        out << raw
        next
      end
      title = (t[2] || t[3]).gsub('"', '&quot;')
      attr = %(data-title="#{title}")
      prev = out.last.to_s.chomp
      if (ial = IAL.match(prev))
        out[-1] = "#{ial[1]}{:#{ial[2]} #{attr}}\n"
      else
        out << "#{indent.rstrip}\n" unless prev.sub(QUOTE_MARKS, '').strip.empty?
        out << "#{indent}{: #{attr}}\n"
      end
      out << "#{indent}#{fence}#{t[1].empty? ? '' : t[1]}\n"
    end
    out.join
  end
end

Jekyll::Hooks.register :documents, :pre_render do |doc|
  next unless doc.output_ext == '.html' && doc.content.include?('title=')
  doc.content = CodeTitles.process(doc.content)
end
