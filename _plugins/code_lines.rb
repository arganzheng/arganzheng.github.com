# Line numbers for code blocks, done at build time.
#
# Every `<pre><code>` in a rendered post gets each line wrapped in
# `<span class="line">` (Rouge's own spans may cross lines — multi-line strings,
# block comments — so they are closed at the newline and reopened on the next
# line; the text content is unchanged) and `data-lines="N"` on the <pre>.
# The numbers themselves are CSS counters on `.line::before`
# (less/theme-overrides.less), so they are not part of the text: copy,
# 划线 quotes, the search index and the WeChat export never see them.
#
# Which blocks show numbers (`pre.lineno`): blocks with a real language
# (```python, ```cpp … — not ```text and not an untyped fence, which are mostly
# shell output, logs and ASCII art) and at least two lines. Override per block
# with a kramdown IAL on the line before the fence: `{:.lineno}` forces them
# on, `{:.no-lineno}` off. Mermaid sources are left alone (the diagram
# renderer reads the <code> as is).
#
# Not Rouge's `line_numbers` option: that is a global table layout whose
# numbers land in innerText, and ~70 % of this blog's fences never reach Rouge.
module CodeLines
  # optional rouge wrapper (carries the language and the IAL classes), then the pre
  BLOCK = %r{(<div class="([^"]*)highlighter-rouge"><div class="highlight">)?<pre\b([^>]*)><code\b([^>]*)>(.*?)</code></pre>}m
  TOKEN = %r{(<span\b[^>]*>|</span>|\n)}
  SKIP_LANGS = %w[text plaintext txt mermaid].freeze

  # class names out of an attribute string (`class="a b"`)
  def self.classes(attrs)
    m = /class="([^"]*)"/.match(attrs.to_s)
    m ? m[1].split : []
  end

  def self.language(class_names)
    class_names.map { |c| c[/\Alanguage-(.+)\z/, 1] }.compact.first
  end

  # Wrap each line; returns [html, line_count] or nil when the block holds
  # markup we do not understand (anything but spans).
  def self.wrap(inner)
    return nil if inner =~ %r{<(?!/?span\b)[a-zA-Z!/]}
    body = inner.sub(/\n\z/, '')
    open = []
    out = +'<span class="line">'
    lines = 1
    body.split(TOKEN).each do |part|
      next if part.empty?
      case part
      when "\n"
        out << ('</span>' * open.size) << "</span>\n<span class=\"line\">" << open.join
        lines += 1
      when '</span>'
        open.pop
        out << part
      when /\A<span/
        open.push(part)
        out << part
      else
        out << part
      end
    end
    out << ('</span>' * open.size) << "</span>\n"
    [out, lines]
  end

  def self.process(html)
    html.gsub(BLOCK) do
      wrapper, wrapper_cls, pre_attrs, code_attrs, inner = $1, $2, $3, $4, $5
      cls = wrapper_cls.to_s.split + classes(pre_attrs)
      lang = language(cls + classes(code_attrs))
      next $& if lang == 'mermaid' || inner.include?('class="line"')
      wrapped = wrap(inner)
      next $& unless wrapped
      body, count = wrapped
      show = if cls.include?('no-lineno') then false
             elsif cls.include?('lineno') then true
             else lang && !SKIP_LANGS.include?(lang) && count >= 2
             end
      attrs = pre_attrs.dup
      if show
        # lineno-3 / lineno-4: gutter width for 3- and 4-digit numbers (CSS cannot count)
        add = (classes(pre_attrs).include?('lineno') ? [] : ['lineno']) + (count >= 100 ? ["lineno-#{count.to_s.size}"] : [])
        unless add.empty?
          attrs = attrs =~ /class="/ ? attrs.sub(/class="/, "class=\"#{add.join(' ')} ") : " class=\"#{add.join(' ')}\"#{attrs}"
        end
      end
      attrs << %( data-lines="#{count}")
      "#{wrapper}<pre#{attrs}><code#{code_attrs}>#{body}</code></pre>"
    end
  end
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'
  doc.output = CodeLines.process(doc.output)
end
