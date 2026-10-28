require 'cgi'
require 'pathname'
require 'safe_yaml'
require 'kramdown'
require 'kramdown-parser-gfm'

module WikiLinks
  FRONT_MATTER = /\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\z)/m
  POST_GLOB = '**/*.{md,markdown}'
  LINK = /\G\[\[([^\[\]\n|]+)(?:\|([^\[\]\n]*))?\]\]/
  OPEN_FENCE = /\A( {0,3}(?:> ?){0,3})(`{3,}|~{3,})([^\r\n]*)\r?\z/
  ATX_HEADING = /\A {0,3}(\#{1,6})[ \t]+(.+?)[ \t]*\#*[ \t]*\r?\z/
  NUMBERING = /\A(?:第[ \t]*)?(?:\d+(?:[.．]\d+)*|[〇零一二三四五六七八九十百千万]+)(?:[、.．:：)）][ \t]*|[ \t]+)/

  Token = Struct.new(:type, :raw, :line, :target, :label, keyword_init: true)
  Record = Struct.new(:file, :relative_path, :title, :slug, :date, :published, :visibility,
                      :body, :line_offset, :doc, keyword_init: true)
  Index = Struct.new(:records, :by_title, :by_slug, :by_path, :headings, :kramdown_options,
                     keyword_init: true)
  Resolution = Struct.new(:record, :heading_id, :heading_text, :display, :pending,
                          keyword_init: true)

  def self.tokenize(content)
    text = content.to_s
    encoding = text.encoding
    # Scan a binary copy so offset access stays O(1) for multibyte text.
    # Delimiters are ASCII; token strings are restored to the source encoding below.
    text = text.b
    tokens = []
    cursor = 0
    line = 1

    while cursor < text.length
      ending = protected_end(text, cursor)
      if ending
        raw = text[cursor...ending]
        append_token(tokens, Token.new(type: :protected, raw: raw, line: line))
        line += raw.count("\n")
        cursor = ending
        next
      end

      match = text[cursor, 2] == '[[' && LINK.match(text, cursor)
      if match && valid_candidate?(match)
        raw = match[0]
        append_token(tokens, Token.new(type: :wikilink, raw: raw, line: line,
                                       target: match[1], label: match[2]))
        line += raw.count("\n")
        cursor = match.end(0)
        next
      end

      next_special = text.index(/[`$\\\[\n]/, cursor) || text.length
      if next_special > cursor
        raw = text[cursor...next_special]
        append_token(tokens, Token.new(type: :text, raw: raw, line: line))
        line += raw.count("\n")
        cursor = next_special
      else
        char = text[cursor]
        append_token(tokens, Token.new(type: :text, raw: char, line: line))
        line += 1 if char == "\n"
        cursor += 1
      end
    end

    tokens.each do |token|
      [:raw, :target, :label].each do |field|
        value = token.public_send(field)
        value.force_encoding(encoding) if value
      end
    end

    tokens
  end

  def self.rewrite(content, source_doc, site)
    source = index_for(site).by_path[File.expand_path(source_doc.path)]
    raise Jekyll::Errors::FatalException, "wikilinks: source is not a post: #{source_doc.path}" unless source

    tokenize(content).map do |token|
      next token.raw unless token.type == :wikilink

      resolution = resolve(token, source, index_for(site))
      if resolution.pending
        location = "#{source.relative_path}:#{source.line_offset + token.line}"
        Jekyll.logger.info 'WikiLinks:', "Pending #{token.raw} in #{location}"
        %(<span class="wikilink wikilink-pending" title="尚未发布">#{CGI.escapeHTML(resolution.display)}</span>)
      else
        href = resolution.record.doc.url
        href = "#{href}##{resolution.heading_id}" if resolution.heading_id
        %(<a class="wikilink" href="#{CGI.escapeHTML(href)}">#{CGI.escapeHTML(resolution.display)}</a>)
      end
    end.join
  end

  def self.references(content, source_doc, site)
    index = index_for(site)
    source = index.by_path[File.expand_path(source_doc.path)]
    raise Jekyll::Errors::FatalException, "wikilinks: source is not a post: #{source_doc.path}" unless source

    tokenize(content).filter_map do |token|
      next unless token.type == :wikilink

      resolution = resolve(token, source, index)
      resolution.record.doc unless resolution.pending
    end
  end

  def self.heading_map(file, kramdown_options = {})
    source = File.read(file, encoding: 'utf-8')
    match = source.match(FRONT_MATTER)
    body = match ? source[match.end(0)..].to_s : source
    lines = heading_lines(body)
    document = Kramdown::Document.new(lines.join, symbolize_keys(kramdown_options))
    document.root.children.filter_map do |node|
      next unless node.type == :header

      [node_text(node).strip, node.attr['id']]
    end
  end

  def self.node_text(node)
    case node.type
    when :text, :codespan, :math
      node.value.to_s
    when :entity
      CGI.unescapeHTML(node.value.to_s)
    when :img
      node.attr['alt'].to_s
    when :br
      ' '
    else
      node.children.map { |child| node_text(child) }.join
    end
  end

  def self.index_for(site)
    site.instance_variable_get(:@wikilinks_index) ||
      site.instance_variable_set(:@wikilinks_index, build_index(site))
  end

  def self.build_index(site)
    docs = site.posts.docs + Array(site.collections['private_posts']&.docs)
    docs_by_path = docs.each_with_object({}) do |doc, result|
      result[File.expand_path(doc.path)] = doc
    end
    files = Dir.glob(File.join(site.source, '_posts', POST_GLOB)).sort
    records = files.map { |file| build_record(file, site, docs_by_path) }
    by_title = Hash.new { |hash, key| hash[key] = [] }
    by_slug = Hash.new { |hash, key| hash[key] = [] }
    by_path = {}
    records.each do |record|
      by_title[record.title] << record unless record.title.empty?
      by_slug[record.slug] << record unless record.slug.empty?
      by_path[File.expand_path(record.file)] = record
    end
    options = symbolize_keys(site.config['kramdown'] || {})
    Index.new(records: records, by_title: by_title, by_slug: by_slug, by_path: by_path,
              headings: {}, kramdown_options: options)
  end

  def self.build_record(file, site, docs_by_path)
    source = File.read(file, encoding: 'utf-8')
    match = source.match(FRONT_MATTER)
    data = match ? (SafeYAML.load(match[1]) || {}) : {}
    body = match ? source[match.end(0)..].to_s : source
    relative_path = Pathname.new(file).relative_path_from(Pathname.new(site.source)).to_s
    doc = docs_by_path[File.expand_path(file)]
    date = post_date(file, data, match && match[1])
    title = data['title'].to_s.strip
    slug = if doc
      File.basename(doc.url, File.extname(doc.url))
    else
      slug_value = data['slug'].to_s.strip
      slug_value = File.basename(file).sub(/\A\d{4}-\d{2}-\d{2}-/, '').sub(/\.(md|markdown)\z/i, '') if slug_value.empty?
      Jekyll::Utils.slugify(slug_value, mode: 'pretty', cased: true)
    end
    Record.new(file: file, relative_path: relative_path, title: title, slug: slug, date: date,
               published: data.fetch('published', true), visibility: data['visibility'].to_s,
               body: body, line_offset: match ? source[0...match.end(0)].count("\n") : 0, doc: doc)
  rescue Psych::Exception => e
    raise Jekyll::Errors::FatalException, "wikilinks: invalid front matter in #{file}: #{e.message}"
  end

  def self.post_date(file, data, front_matter)
    if data.key?('date')
      raw_date = front_matter&.match(/^date[ \t]*:[ \t]*(.*?)\s*$/)&.captures&.first
      raw_date = raw_date&.sub(/\s+#.*\z/, '')&.strip
      raw_date = data['date'].to_s if raw_date.nil? || raw_date.start_with?('"', "'", '*', '!', '&')
      Jekyll::Utils.parse_date(raw_date)
    else
      filename_date = File.basename(file)[/\A(\d{4}-\d{2}-\d{2})-/, 1]
      filename_date && Jekyll::Utils.parse_date(filename_date)
    end
  end

  def self.resolve(token, source, index)
    target_name, heading_request = token.target.split('#', 2)
    target = target_name.empty? ? source : find_target(target_name, source, token, index)
    private_source = source.visibility == 'private'
    if target.visibility == 'private' && !private_source
      fail_link(source, token, 'public post cannot link to a private target')
    end

    heading = if heading_request
      match_heading(target, heading_request, source, token, index)
    end
    heading_id = heading && heading[1]
    heading_text = heading && heading[0]
    display = if !token.label.nil?
      token.label
    elsif heading_text && target.equal?(source)
      heading_text
    elsif heading_text
      "#{target.title.empty? ? target.slug : target.title} · #{heading_text}"
    else
      target.title.empty? ? target.slug : target.title
    end

    Resolution.new(record: target, heading_id: heading_id, heading_text: heading_text,
                   display: display, pending: target.doc.nil?)
  end

  def self.find_target(name, source, token, index)
    slugs = index.by_slug[name]
    return slugs.first if slugs.size == 1
    fail_link(source, token, "ambiguous slug #{name.inspect}") if slugs.size > 1

    titles = index.by_title[name]
    return titles.first if titles.size == 1
    fail_link(source, token, "ambiguous title #{name.inspect}; use the slug instead") if titles.size > 1

    fail_link(source, token, "unknown target #{name.inspect}")
  end

  def self.match_heading(record, request, source, token, index)
    headings = index.headings[record.file] ||= heading_map(record.file, index.kramdown_options)
    exact = headings.find { |text, _id| text == request }
    return exact if exact

    normalized_request = without_numbering(request)
    match = headings.find { |text, _id| without_numbering(text) == normalized_request }
    return match if match

    fail_link(source, token, "unknown heading #{request.inspect} in #{record.title.inspect}")
  end

  def self.without_numbering(text)
    text.to_s.sub(NUMBERING, '').strip
  end

  def self.heading_lines(content)
    lines = []
    open_fence = nil
    content.each_line do |line|
      plain = line.chomp
      if open_fence
        closing = /\A#{Regexp.escape(open_fence[:prefix])}[ \t]{0,3}#{Regexp.escape(open_fence[:char])}{#{open_fence[:length]},}[ \t]*\z/
        open_fence = nil if plain.match?(closing)
        next
      end

      if (fence = OPEN_FENCE.match(plain))
        char = fence[2][0]
        info = fence[3]
        next if char == '`' && info.include?('`')

        open_fence = { prefix: fence[1], char: char, length: fence[2].length }
        next
      end

      lines << line if ATX_HEADING.match?(plain)
    end
    lines
  end

  def self.protected_end(text, cursor)
    if cursor.zero? || text[cursor - 1] == "\n"
      block_end = fenced_block_end(text, cursor) || indented_code_end(text, cursor)
      return block_end if block_end
    end

    math_end(text, cursor) || inline_code_end(text, cursor)
  end

  def self.fenced_block_end(text, cursor)
    line_end = text.index("\n", cursor) || text.length
    opening = OPEN_FENCE.match(text[cursor...line_end])
    return unless opening
    return if opening[2][0] == '`' && opening[3].include?('`')

    fence_char = opening[2][0]
    fence_length = opening[2].length
    prefix = Regexp.escape(opening[1])
    closer = /\A#{prefix}[ \t]{0,3}#{Regexp.escape(fence_char)}{#{fence_length},}[ \t]*\r?\z/
    position = line_end < text.length ? line_end + 1 : line_end
    while position < text.length
      next_end = text.index("\n", position) || text.length
      return next_end < text.length ? next_end + 1 : next_end if text[position...next_end].match?(closer)

      position = next_end < text.length ? next_end + 1 : next_end
    end
    text.length
  end

  def self.indented_code_end(text, cursor)
    line_end = text.index("\n", cursor) || text.length
    return unless text[cursor...line_end].match?(/\A(?: {4}|\t)/)

    position = line_end < text.length ? line_end + 1 : line_end
    while position < text.length
      next_end = text.index("\n", position) || text.length
      line = text[position...next_end]
      break unless line.match?(/\A(?: {4}|\t|\r?\z)/)

      position = next_end < text.length ? next_end + 1 : next_end
    end
    position
  end

  def self.math_end(text, cursor)
    delimiter = if text[cursor, 2] == '$$'
      '$$'
    elsif text[cursor, 2] == '\\['
      '\\]'
    elsif text[cursor, 2] == '\\('
      '\\)'
    end
    return unless delimiter && !escaped?(text, cursor)

    closing = text.index(delimiter, cursor + 2)
    closing && !escaped?(text, closing) ? closing + 2 : nil
  end

  def self.inline_code_end(text, cursor)
    return unless text[cursor] == '`'

    length = 1
    length += 1 while text[cursor + length] == '`'
    delimiter = '`' * length
    position = cursor + length
    while (closing = text.index(delimiter, position))
      before = closing.positive? && text[closing - 1] == '`'
      after = text[closing + length] == '`'
      return closing + length unless before || after

      position = closing + length
    end
    nil
  end

  def self.escaped?(text, position)
    count = 0
    cursor = position - 1
    while cursor >= 0 && text[cursor] == '\\'
      count += 1
      cursor -= 1
    end
    count.odd?
  end

  def self.valid_candidate?(match)
    inner = match[0][2...-2]
    target = match[1]
    return false if inner.match?(/\A\s|\s\z/) || target != target.strip

    target_name = target.split('#', 2).first.to_s
    return false if target_name.match?(/\A[:$-]|\A\d/)

    true
  end

  def self.append_token(tokens, token)
    if token.type == :text && tokens.last&.type == :text
      tokens.last.raw << token.raw
    else
      tokens << token
    end
  end

  def self.fail_link(source, token, message)
    line = source.line_offset + token.line
    raise Jekyll::Errors::FatalException,
          "wikilinks: #{source.relative_path}:#{line}: #{message}: #{token.raw}"
  end

  def self.symbolize_keys(hash)
    hash.each_with_object({}) { |(key, value), result| result[key.to_sym] = value }
  end

  [:posts, :private_posts].each do |collection|
    Jekyll::Hooks.register collection, :pre_render do |doc, _payload|
      doc.content = WikiLinks.rewrite(doc.content, doc, doc.site)
    end
  end

  Jekyll::Hooks.register :site, :after_reset do |site|
    site.remove_instance_variable(:@wikilinks_index) if site.instance_variable_defined?(:@wikilinks_index)
  end
end
