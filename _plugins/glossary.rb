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
  NON_NAME = /[^a-zA-Z0-9-]/
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

  BOUND_RE = /[A-Za-z0-9_-]/

  # Char-indexed literal candidates for a series' match strings, cached:
  # first char => strings starting with it (longest-first), plus one
  # character-class regex that jumps between positions where any term can
  # start. Literal prefix checks replace a regex per term per segment.
  def self.combined_index(entries)
    @index_cache ||= {}
    @index_cache[entries] ||= begin
      strings = entries.flat_map { |e| [e['term']] + Array(e['aliases']) }.uniq
      first = {}
      strings.sort_by { |t| -t.length }.each do |t|
        (first[t[0]] ||= []) << t
      end
      heads = first.keys.uniq
      heads_re = /[#{Regexp.escape(heads.join)}]/
      [first, heads_re]
    end
  end

  # Wrap each term's first occurrence in the whole post. Spans are located
  # over the unblocked text nodes only, then spliced in right-to-left so
  # offsets stay valid.
  def self.process(html, entries, overview)
    # Work on a binary copy: String#[]/getbyte/index are byte-based there, so
    # scattered offset access stays O(1) instead of quadratic on multibyte
    # text. Every check is ASCII or literal, so byte-precision equals the
    # char-precision result; encoding is restored before returning.
    html = html.b
    segments = text_segments(html)
    return html.force_encoding(Encoding::UTF_8) if segments.empty?

    first, _heads_re = combined_index(entries)

    # Phase 1: for every match string, its first boundary-valid occurrence
    # fully inside each segment — the same candidates the old per-term regex
    # found (slice edges count as boundaries, like the lookbehind/lookahead
    # did). String#index does the scanning; segment membership comes from the
    # sorted segment starts.
    starts = segments.map(&:first)
    cand = {}
    first.each_value do |strings|
      strings.each do |t|
        tb = t.b
        tl = tb.bytesize
        from = 0
        seen = cand[tb] ||= {}
        while (q = html.index(tb, from))
          si = (starts.bsearch_index { |s| s > q } || segments.length) - 1
          spos, slen = segments[si] || [0, 0]
          if si >= 0 && q >= spos && q + tl <= spos + slen &&
             (q == spos || html[q - 1] !~ BOUND_RE) &&
             (q + tl == spos + slen || html[q + tl] !~ BOUND_RE) &&
             !seen.key?(si)
            seen[si] = q
          end
          from = q + 1
        end
      end
    end

    # Phase 2: same claim order as before — entries longest-first, aliases
    # longest-first, segments in order, first candidate wins; one span per
    # entry. Claimed ranges are disjoint and sorted so overlap is a two-
    # neighbour check.
    spans = []
    entries.sort_by { |e| -e['term'].to_s.length }.each do |e|
      seen = false
      ([e['term']] + Array(e['aliases'])).sort_by { |t| -t.length }.each do |t|
        break if seen
        cts = cand[t.b] or next
        segments.each_with_index do |(pos, _), si|
          break if seen
          hp = cts[si] or next
          range = hp...(hp + t.b.bytesize)
          i = spans.bsearch_index { |s| s[:range].begin >= range.begin } || spans.length
          overlap = (i.positive? && spans[i - 1][:range].end > range.begin) ||
                    (i < spans.length && spans[i][:range].begin < range.end)
          next if overlap

          spans.insert(i, { range: range, entry: e, text: t })
          seen = true
        end
      end
    end
    spans.sort_by { |s| -s[:range].begin }.each do |s|
      e = s[:entry]
      full = e['full'].to_s.strip
      tip = "**#{e['term']}**#{full == '—' || full.empty? ? '' : "（#{full}）"}：#{e['desc']}"
      url = e['see'].to_s.empty? ? "#{overview}#glossary" : e['see']
      span = %(<span class="inline-tip glossary-term" data-tip="#{esc_attr(tip)}" data-url="#{esc_attr(url)}">#{s[:text]}</span>)
      html[s[:range]] = span.b
    end
    html.force_encoding(Encoding::UTF_8)
  end

  # [offset, length] of every text stretch that may hold a match: outside the
  # blocked elements (properly nested) and outside \( \) / \[ \] math, which
  # can open in one node and close in a later one.
  #
  # Tags are located with a byte-level scanner (String#index + getbyte) that
  # accepts exactly the same regions as the old TAG regex:
  # </?name(attrs)> where '>' inside a complete "…" or '…' pair does not end
  # the tag; an unterminated tag is not a match at all.
  def self.text_segments(html)
    segs = []
    stack = []
    in_math = nil
    last = 0
    n = html.length
    # positions of \( \[ \) \] sorted ascending; the openers/closers of a gap
    # are then found by bsearch instead of a regex scan to the next mark.
    marks = ['\\(', '\\[', '\\)', '\\]'].flat_map do |mk|
      offs = []
      i = 0
      while (p = html.index(mk, i))
        offs << p
        i = p + 1
      end
      offs
    end.sort
    consume = lambda do |base, len|
      if stack.any?
        in_math = nil
      else
        pos = base
        fin = base + len
        while pos < fin
          if in_math
            closer = html.index(in_math == :paren ? '\\)' : '\\]', pos)
            if closer && closer + 2 <= fin
              in_math = nil
              pos = closer + 2
            else
              pos = fin
            end
          else
            mi = marks.bsearch_index { |m| m >= pos }
            mk = mi && marks[mi]
            if mk && mk + 2 <= fin
              two = html[mk, 2]
              if two == '\\)' || two == '\\]'
                pos = mk + 2
              else
                segs << [pos, mk - pos] if mk > pos
                in_math = two == '\\(' ? :paren : :bracket
                pos = mk + 2
              end
            else
              segs << [pos, fin - pos]
              pos = fin
            end
          end
        end
      end
    end

    lt = 0
    while (lt = html.index('<', lt))
      j = lt + 1
      break if j >= n
      closing = false
      if html[j] == '/'
        closing = true
        j += 1
        break if j >= n
      end
      c = html[j]
      unless (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        lt += 1
        next
      end
      name_start = j
      name_end = html.index(NON_NAME, j) || n
      # attrs: '>' outside a complete quote pair ends the tag
      j = name_end
      gt = nil
      while j < n
        cand_gt = html.index('>', j)
        break if cand_gt.nil? # unterminated: not a tag
        qp = html.index(/["']/, j)
        if qp.nil? || qp > cand_gt
          gt = cand_gt
          break
        end
        qc = html.index(html[qp], qp + 1)
        if qc.nil? || qc >= n # unclosed quote: not a tag
          cand_gt = nil
          break
        end
        j = qc + 1
      end
      if gt.nil?
        lt += 1
        next
      end
      te = gt + 1
      consume.call(last, lt - last)
      name = html[name_start...name_end].downcase
      if closing
        i = stack.rindex(name)
        stack.slice!(i..) if i
      elsif html[gt - 1] != '/' &&
            (BLOCKED =~ name ||
             ((tip = html.index('inline-tip', name_end)) && tip < gt &&
              html[name_end...gt] =~ /class\s*=\s*["'][^"']*inline-tip/))
        stack.push(name)
      end
      last = te
      lt = te
    end
    consume.call(last, n - last)
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
