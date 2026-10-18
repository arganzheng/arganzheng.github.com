# Series-level glossary: `_data/glossary/<series>.yml` feeds
# `site.data['glossary']` (per series, each term + its first member post) and
# `site.data['glossary_maps']` (per roadmap, merged and deduped). The
# :post_convert hook then wraps the first prose occurrence of every term in a
# post in <span class="inline-tip glossary-term" data-tip data-url>. Matching
# happens only in text nodes outside pre/code/a/headings/table/figcaption/
# script/style/svg/summary, outside \( \) \[ \] math and outside existing
# .inline-tip elements; both neighbours must not be [A-Za-z0-9_-] so GPT never
# matches inside GPT-2 and RL never inside RLHF.
module Glossary
  BLOCKED = /\A(?:pre|code|a|h[1-6]|table|figcaption|script|style|svg|summary)\z/i
  TAG = %r{</?([a-zA-Z][a-zA-Z0-9-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)>}
  BOUND = '[A-Za-z0-9_-]'
  MATH_MARK = /\\\(|\\\[|\\\)|\\\]/

  def self.mask_markdown(text)
    s = text.to_s.gsub(/\A---\s*\n.*?\n---\s*\n/m, '')
    s.gsub!(/```.*?```/m) { |m| ' ' * m.length }
    s.gsub!(/`[^`\n]*`/) { |m| ' ' * m.length }
    s.gsub!(/\$\$.*?\$\$/m) { |m| ' ' * m.length }
    s.gsub!(/\]\(([^)\s]*)[^)]*\)/) { |m| ']' + ' ' * (m.length - 1) }
    s
  end

  def self.term_re(term)
    /(?<![#{BOUND}])#{Regexp.escape(term)}(?![#{BOUND}])/
  end

  # Alphabetical (case-insensitive) order for the overview's series table.
  def self.entries_for(site, key, raw)
    overview = (site.data['series'] || {}).dig(key, 'overview')
    raw.map { |e| e.merge('overview' => overview) }
      .sort_by { |e| e['term'].to_s.downcase }
  end

  # Wrap each term's first occurrence inside every h2 section (section 0 =
  # content before the first <h2). Spans are located over the unblocked text
  # nodes only, then spliced in right-to-left so offsets stay valid.
  def self.process(html, entries, overview)
    segments = text_segments(html)
    return html if segments.empty?

    h2s = html.enum_for(:scan, /<h2\b/).map { Regexp.last_match.begin(0) }
    section_of = ->(pos) { h2s.count { |h| h <= pos } }
    spans = []
    entries.sort_by { |e| -e['term'].to_s.length }.each do |e|
      seen_sections = {}
      ([e['term']] + Array(e['aliases'])).sort_by { |t| -t.length }.each do |t|
        re = term_re(t)
        segments.each do |pos, len|
          sec = section_of.call(pos)
          next if seen_sections[sec]
          hit = re.match(html[pos, len])
          next unless hit
          range = (pos + hit.begin(0))...(pos + hit.end(0))
          next if spans.any? { |s| range.begin < s[:range].end && s[:range].begin < range.end }

          spans << { range: range, entry: e, text: hit[0] }
          seen_sections[sec] = true
        end
      end
    end
    spans.sort_by { |s| -s[:range].begin }.each do |s|
      e = s[:entry]
      full = e['full'].to_s.strip
      tip = "**#{e['term']}**#{full == '—' || full.empty? ? '' : "（#{full}）"}：#{e['desc']}"
      url = e['see'].to_s.empty? ? "#{overview}#glossary" : e['see']
      span = %(<span class="inline-tip glossary-term" data-tip="#{esc_attr(tip)}" data-url="#{esc_attr(url)}">#{s[:text]}</span>)
      html[s[:range]] = span
    end
    html
  end

  # [offset, length] of every text stretch that may hold a match: outside the
  # blocked elements (properly nested) and outside \( \) / \[ \] math, which
  # can open in one node and close in a later one.
  def self.text_segments(html)
    segs = []
    stack = []
    in_math = nil
    last = 0
    consume = lambda do |base, text|
      if stack.any?
        in_math = nil
      else
        pos = 0
        while pos < text.length
          if in_math
            closer = in_math == :paren ? text.index('\\)', pos) : text.index('\\]', pos)
            if closer
              in_math = nil
              pos = closer + 2
            else
              pos = text.length
            end
          else
            opener = MATH_MARK.match(text, pos)
            if opener
              if opener[0] == '\\)' || opener[0] == '\\]'
                pos = opener.end(0)
              else
                segs << [base + pos, opener.begin(0) - pos] if opener.begin(0) > pos
                in_math = opener[0] == '\\(' ? :paren : :bracket
                pos = opener.end(0)
              end
            else
              segs << [base + pos, text.length - pos]
              pos = text.length
            end
          end
        end
      end
    end

    while (tag = TAG.match(html, last))
      consume.call(last, html[last...tag.begin(0)])
      name = tag[1].downcase
      if tag[0].start_with?('</')
        i = stack.rindex(name)
        stack.slice!(i..) if i
      elsif !tag[0].end_with?('/>') &&
            (BLOCKED =~ name || tag[2].to_s =~ /class\s*=\s*["'][^"']*inline-tip/)
        stack.push(name)
      end
      last = tag.end(0)
    end
    consume.call(last, html[last..] || '')
    segs.select { |_, l| l > 0 }
  end

  def self.esc_attr(s)
    s.gsub('&', '&amp;').gsub('"', '&quot;').gsub('<', '&lt;').gsub('>', '&gt;')
  end
end

Jekyll::Hooks.register :site, :post_read do |site|
  # _data/glossary/ is auto-loaded into site.data['glossary'] keyed by file
  # name; read the raw entries, then replace the node with enriched ones.
  raw = site.data['glossary'] || {}
  glossaries = raw.each_with_object({}) do |(key, entries), h|
    next unless (site.data['series'] || {})[key]
    enriched = Glossary.entries_for(site, key, entries || [])
    h[key] = enriched unless enriched.empty?
  end
  site.data['glossary'] = glossaries
end

Jekyll::Hooks.register :documents, :post_convert do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'

  glossaries = doc.site.data['glossary'] || {}
  next if glossaries.empty?

  key = doc.data['series'] ||
        (doc.site.data['series'] || {}).find { |_, m| m['overview'] == doc.url }&.first
  entries = key && glossaries[key]
  next unless entries && !entries.empty?

  overview = doc.site.data.dig('series', key, 'overview').to_s
  converted = Glossary.process(doc.content.dup, entries, overview)
  doc.content = converted
  doc.output = Glossary.process(doc.output.to_s.empty? ? converted : doc.output.to_s.dup, entries, overview)
end
