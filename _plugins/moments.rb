# 随笔 / Moments — a 朋友圈-style timeline of short notes.
#
# One Markdown file per month, `moments/YYYY-MM.md` (layout `moments` via the
# `_config.yml` defaults), entries separated by a dated heading:
#
#   ## 2026-09-21 08:02 @深圳湾      date, optional HH:MM, optional @place
#   text, images, a quote, a music link — plain Markdown
#
# This generator turns each month page into structured data the layout and the
# feed iterate over instead of `{{ content }}`:
#
#   page.moments = [{ 'id' => '20260921-0802', 'title' => '2026-09-21 08:02',
#                     'time' => Time, 'place' => '深圳湾', 'html' => …, 'text' => … }, …]
#   site.data['moments'] = { 'months' => [pages, newest first], 'entries' => [all entries + 'url'] }
#
# newest first. The Markdown of each entry is rendered with the site's kramdown
# converter, then touched up:
#   - a paragraph made only of images (one per line, or several paragraphs in a
#     row) becomes `.moment-gallery.n-<count>` — 1 large, 2/4 two columns, 3+ a grid;
#   - a blockquote keeps its line breaks (a poem), and when its last line starts
#     with —— / — / -- that line becomes `.moment-cite` (the attribution);
#   - a line that is just a URL of 网易云 / QQ 音乐 / Spotify / Apple Music, or of
#     an .mp3/.m4a/.ogg file, becomes a player card (`.moment-music`).
# `/moments/` (the section's front door) is a copy of the newest month whose
# comments / views / reactions stay keyed on the month's URL
# (`comments_path`), so nothing forks between the two addresses.
module Moments
  HEAD = /\A##\s+(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{1,2}):(\d{2}))?(?:\s+@\s*(.+?))?\s*\z/
  IMG_P = %r{\A<p>\s*(?:<img\b[^>]*>\s*(?:<br\s*/?>)?\s*)+</p>\z}m
  MUSIC = [
    # 网易云: https://music.163.com/#/song?id=347230  (also /song?id=)
    [%r{\Ahttps?://music\.163\.com/(?:#/)?song\?(?:.*&)?id=(\d+)}i,
     ->(m) { iframe("https://music.163.com/outchain/player?type=2&id=#{m[1]}&auto=0&height=66", 86) }],
    # QQ 音乐: https://y.qq.com/n/ryqq/songDetail/003OUlho2HcRHC
    [%r{\Ahttps?://y\.qq\.com/n/(?:ryqq|yqq)/song(?:Detail)?/([0-9A-Za-z]+)}i,
     ->(m) { iframe("https://i.y.qq.com/n2/m/outchain/player/index.html?songmid=#{m[1]}&songtype=0", 90) }],
    # Spotify: https://open.spotify.com/track/<id>
    [%r{\Ahttps?://open\.spotify\.com/(?:intl-[a-z]+/)?track/([0-9A-Za-z]+)}i,
     ->(m) { iframe("https://open.spotify.com/embed/track/#{m[1]}", 152) }],
    # Apple Music: https://music.apple.com/cn/album/xxx/123?i=456
    [%r{\Ahttps?://music\.apple\.com/(.+)\z}i,
     ->(m) { iframe("https://embed.music.apple.com/#{m[1]}", 175) }],
    # a file: <audio>
    [%r{\Ahttps?://\S+\.(mp3|m4a|ogg|wav)(?:\?\S*)?\z}i,
     ->(m) { %(<audio controls preload="none" src="#{m[0]}"></audio>) }],
    [%r{\A/\S+\.(mp3|m4a|ogg|wav)\z}i,
     ->(m) { %(<audio controls preload="none" src="#{m[0]}"></audio>) }]
  ]

  def self.iframe(src, height)
    %(<iframe src="#{src}" height="#{height}" loading="lazy" frameborder="0" allow="autoplay; encrypted-media" title="音乐播放器"></iframe>)
  end

  # A line holding only a URL → the player card, or nil.
  def self.music_card(line)
    url = line.strip
    MUSIC.each do |re, build|
      m = url.match(re)
      return %(<div class="moment-music">#{build.call(m)}<a class="moment-music-link" href="#{url}" target="_blank" rel="noopener">打不开？去原站听</a></div>) if m
    end
    nil
  end

  # Markdown of one entry → HTML with the gallery / quote / music touch-ups.
  def self.render(site, md)
    lines = md.lines.map { |l| music_card(l) ? "\n#{music_card(l)}\n" : l }
    # quote lines break where the author broke them: `> a` / `> b` → a<br>b
    lines.each_with_index { |l, i| lines[i] = l.chomp + "  \n" if l =~ /\A>\s*\S/ && lines[i + 1] =~ /\A>\s*\S/ }
    md = lines.join
    html = site.find_converter_instance(Jekyll::Converters::Markdown).convert(md)
    html = html.gsub(/<img\b(?![^>]*\bloading=)/, '<img loading="lazy" decoding="async"')
    html = galleries(html)
    quotes(html).strip
  end

  def self.galleries(html)
    # consecutive image-only paragraphs merge into one gallery
    html.gsub(%r{(?:<p>\s*(?:<img\b[^>]*>\s*(?:<br\s*/?>)?\s*)+</p>\s*)+}m) do |run|
      imgs = run.scan(/<img\b[^>]*>/)
      items = imgs.map { |i| %(<a class="moment-pic" href="#{i[/\bsrc="([^"]*)"/, 1]}">#{i}</a>) }
      %(<div class="moment-gallery n-#{imgs.size}">#{items.join}</div>\n)
    end
  end

  def self.quotes(html)
    html.gsub(%r{<blockquote>(.*?)</blockquote>}m) do
      inner = Regexp.last_match(1)
      # the attribution is the last line of the last paragraph: `—— 苏轼《…》`
      if inner =~ %r{\A(.*?)(?:<br\s*/?>\s*|\n)(\s*(?:——|—|--|―)\s*)([^<\n]+?)\s*</p>\s*\z}m
        %(<blockquote class="moment-quote">#{Regexp.last_match(1)}</p><div class="moment-cite">#{Regexp.last_match(3)}</div></blockquote>)
      else
        %(<blockquote class="moment-quote">#{inner}</blockquote>)
      end
    end
  end

  def self.parse(site, page)
    entries = []
    cur = nil
    page.content.each_line do |line|
      if (m = line.chomp.match(HEAD))
        entries << cur if cur
        y, mo, d, h, mi, place = m.captures
        time = Time.new(y.to_i, mo.to_i, d.to_i, (h || 0).to_i, (mi || 0).to_i)
        cur = { 'time' => time, 'has_time' => !h.nil?, 'place' => place, 'md' => +'' }
      elsif cur
        cur['md'] << line
      elsif line.strip != ''
        Jekyll.logger.warn 'moments:', "#{page.relative_path}: text before the first `## YYYY-MM-DD` heading is dropped"
      end
    end
    entries << cur if cur
    ids = Hash.new(0)
    entries.each do |e|
      base = e['time'].strftime(e['has_time'] ? '%Y%m%d-%H%M' : '%Y%m%d')
      ids[base] += 1
      e['id'] = ids[base] > 1 ? "#{base}-#{ids[base]}" : base
      e['title'] = e['time'].strftime(e['has_time'] ? '%Y-%m-%d %H:%M' : '%Y-%m-%d')
      e['html'] = render(site, e['md'])
      e['text'] = e['html'].gsub(%r{<(script|style|iframe|audio)\b.*?</\1>}m, ' ').gsub(/<[^>]+>/, ' ').gsub(/\s+/, ' ').strip
      e.delete('md')
    end
    entries.sort_by { |e| e['time'] }.reverse
  end

  FILE = /\A(\d{4})-(\d{2})\.md\z/

  # Before anything asks for a URL: pages take no `.html` from the site's
  # `/:title.html` permalink style (that is for posts), and the worker's path
  # check (`VIEW_PATH`) wants `.html`; date / title come from the file name.
  Jekyll::Hooks.register :site, :post_read do |site|
    site.pages.each do |page|
      next unless page.data['layout'] == 'moments' && (m = File.basename(page.name).match(FILE))
      y, mo = m.captures.map(&:to_i)
      page.data['permalink'] ||= format('/moments/%04d-%02d.html', y, mo)
      page.data['date'] ||= Time.new(y, mo, 1)
      page.data['month'] = format('%04d-%02d', y, mo)
      page.data['title'] ||= "随笔 · #{y} 年 #{mo} 月"
    end
  end

  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      months = site.pages.select { |p| p.data['layout'] == 'moments' && p.data['month'] }
      months.each do |page|
        page.data['moments'] = Moments.parse(site, page)
        page.data['comments_path'] = page.url
      end
      months.sort_by! { |p| p.data['month'] }.reverse!
      months.each_with_index do |p, i|
        p.data['newer'] = months[i - 1].url if i > 0
        p.data['older'] = months[i + 1].url if months[i + 1]
      end
      all = months.flat_map { |p| p.data['moments'].map { |e| e.merge('url' => "#{p.url}##{e['id']}", 'month' => p.data['month']) } }
      site.data['moments'] = { 'months' => months.map { |p| { 'url' => p.url, 'month' => p.data['month'], 'title' => p.data['title'], 'count' => p.data['moments'].size } }, 'entries' => all }
      return if months.empty?

      # /moments/ = the newest month, same discussion / counters
      latest = months.first
      index = Jekyll::PageWithoutAFile.new(site, site.source, 'moments', 'index.md')
      index.content = latest.content
      index.data = latest.data.merge('permalink' => '/moments/', 'canonical' => latest.url, 'sitemap' => false, 'is_index' => true)
      site.pages << index
    end
  end
end
