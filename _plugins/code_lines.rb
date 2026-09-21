# Line numbers and code refs for code blocks, done at build time.
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
# Code refs (the Code Hike "code mentions" model): a comment line of its own
#
#     # !ref base        <- marks the next line
#     # !ref rec +1      <- marks the next two lines
#
# is *removed* from the rendered block (not counted, not copied, not indexed)
# and the following line(s) become `<span class="line ref-line" data-ref="base"
# id="base">`. Prose refers to it with a plain kramdown link `[基例](#base)`,
# which becomes `<a class="code-ref" data-ref="base" data-line="N">`;
# js/code-refs.js wires hover / click / popover between the two, without JS it
# is an ordinary anchor jump (`.line:target`). Blocks with refs always show
# line numbers (the marker lives in the gutter). A `!ref` nobody links to is
# logged as a warning; a link to a ref that does not exist is caught by lychee.
# `{:.no-refs}` on a block leaves its directive lines alone (for showing the
# syntax itself, as the memo does).
#
# Not Rouge's `line_numbers` option: that is a global table layout whose
# numbers land in innerText, and ~70 % of this blog's fences never reach Rouge.
module CodeLines
  # optional rouge wrapper (carries the language and the IAL classes), then the pre
  BLOCK = %r{(<div class="([^"]*)highlighter-rouge"><div class="highlight">)?<pre\b([^>]*)><code\b([^>]*)>(.*?)</code></pre>}m
  TOKEN = %r{(<span\b[^>]*>|</span>|\n)}
  SKIP_LANGS = %w[text plaintext txt mermaid].freeze
  # `# !ref name`, `// !ref name +2`, `/* !ref name */`, `<!-- !ref name -->` … on a line of its own
  DIRECTIVE = %r{\A\s*(?:#+|/{2,}|--|;+|%+|/\*|<!--|\(\*|")\s*!ref\s+([A-Za-z][\w-]*)(?:\s+\+(\d+))?\s*(?:\*/|-->|\*\))?\s*\z}
  ENTITIES = { '&lt;' => '<', '&gt;' => '>', '&amp;' => '&', '&quot;' => '"', '&#39;' => "'" }.freeze

  # class names out of an attribute string (`class="a b"`)
  def self.classes(attrs)
    m = /class="([^"]*)"/.match(attrs.to_s)
    m ? m[1].split : []
  end

  def self.language(class_names)
    class_names.map { |c| c[/\Alanguage-(.+)\z/, 1] }.compact.first
  end

  def self.text_of(html)
    html.gsub(/<[^>]*>/, '').gsub(/&(lt|gt|amp|quot|#39);/) { ENTITIES[$&] }
  end

  # Wrap each line; returns [html, line_count, refs] (refs: name => first line
  # number) or nil when the block holds markup we do not understand (anything
  # but spans).
  def self.wrap(inner, known = {}, directives = true)
    return nil if inner =~ %r{<(?!/?span\b)[a-zA-Z!/]}
    body = inner.sub(/\n\z/, '')
    open = []          # rouge spans open at the current position
    line_open = []     # … at the start of the current line (reopened after a directive is dropped)
    out = +''
    line = +''
    lines = 0
    refs = {}
    pending = nil      # [name, remaining lines]

    flush = lambda do
      # a directive line is dropped — unless it sits inside a multi-line
      # string / block comment, where it is just text. Spans carried over
      # from the previous line that close right at the start of this one do
      # not count (Rouge's Python comments swallow the newline).
      carried = line_open.size - line[%r{\A(?:</span>)*}].count('/')
      if directives && carried <= 0 && (m = DIRECTIVE.match(text_of(line)))
        pending = [m[1], (m[2] || 0).to_i + 1]
      else
        lines += 1
        tag = '<span class="line">'
        if pending
          name, left = pending
          # the id goes on the ref's first line only (anchor target); a name
          # reused in another block (code-tabs panels) gets no second id
          tag = %(<span class="line ref-line" data-ref="#{name}"#{refs[name] || known[name] ? '' : %( id="#{name}")}>)
          refs[name] ||= lines
          pending = left > 1 ? [name, left - 1] : nil
        end
        out << tag << line_open.join << line << ('</span>' * open.size) << "</span>\n"
      end
      line = +''
      line_open = open.dup
    end

    body.split(TOKEN).each do |part|
      next if part.empty?
      case part
      when "\n"
        flush.call
      when '</span>'
        open.pop
        line << part
      when /\A<span/
        open.push(part)
        line << part
      else
        line << part
      end
    end
    flush.call
    [out, lines, refs]
  end

  def self.process(html)
    all_refs = {}
    html = html.gsub(BLOCK) do
      wrapper, wrapper_cls, pre_attrs, code_attrs, inner = $1, $2, $3, $4, $5
      cls = wrapper_cls.to_s.split + classes(pre_attrs)
      lang = language(cls + classes(code_attrs))
      next $& if lang == 'mermaid' || inner.include?('class="line"')
      wrapped = wrap(inner, all_refs, !cls.include?('no-refs'))
      next $& unless wrapped
      body, count, refs = wrapped
      refs.each { |name, n| (all_refs[name] ||= []) << n }
      show = if cls.include?('no-lineno') then false
             elsif cls.include?('lineno') || !refs.empty? then true
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
      attrs << %( data-refs="#{refs.keys.join(' ')}") unless refs.empty?
      "#{wrapper}<pre#{attrs}><code#{code_attrs}>#{body}</code></pre>"
    end
    return [html, all_refs, []] if all_refs.empty?

    # prose links to the refs (outside <pre>): mark them for the CSS / JS / export
    linked = {}
    html = html.gsub(%r{<pre\b.*?</pre>|<a\s+href="#([A-Za-z][\w-]*)"(?![^>]*class=)}m) do
      name = $1
      next $& unless name && all_refs[name]
      linked[name] = true
      %(<a class="code-ref" href="##{name}" data-ref="#{name}" data-line="#{all_refs[name].first}")
    end
    [html, all_refs, all_refs.keys - linked.keys]
  end
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'
  doc.output, _refs, unused = CodeLines.process(doc.output)
  # the search index (_plugins/search_index.rb) and the WeChat export read
  # doc.content, so the directive lines must go from there as well
  doc.content = CodeLines.process(doc.content)[0]
  unused.each { |name| Jekyll.logger.warn 'code refs:', "#{doc.relative_path}: `!ref #{name}` is never linked from the text" }
end

# slide landing pages (`layout: slides`, see _plugins/slides_deck.rb) show every
# slide flat under the player with the post styles, so their code gets the same
# treatment; the bare deck (`layout: deck`) keeps reveal's own code rendering
Jekyll::Hooks.register :pages, :post_render do |page|
  next unless page.data['layout'] == 'slides' && page.output_ext == '.html'
  page.output = CodeLines.process(page.output)[0]
end
