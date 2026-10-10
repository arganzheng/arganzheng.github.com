# 随笔 / Moments — a 朋友圈-style timeline of short notes.
#
# One Markdown file per month, `moments/YYYY-MM.md` (layout `moments` via the
# `_config.yml` defaults), entries separated by a dated heading:
#
#   ## 2026-09-21 08:02 @深圳湾      date, optional HH:MM, optional @place
#   text, images, a quote, a music link — plain Markdown. One short video uses:
#   ![视频](https://media.example/moments/2026/10/clip.mp4 "/img/moments/2026/10/poster.jpg")
#
# This generator turns each month page into structured data the layout and the
# feed iterate over instead of `{{ content }}`:
#
#   page.moments = [{ 'id' => '20260921-0802', 'month' => '2026-09', 'title' => '2026-09-21 08:02',
#                     'time' => Time, 'place' => '深圳湾', 'html' => …, 'text' => … }, …]
#   site.data['moments'] = { 'months' => [pages, newest first], 'entries' => [all entries + 'url/page/refs/related'] }
#   _data/moments.yml supplies entry ids in `pinned`; only /moments/ renders them in its leading block.
#
# newest first. The Markdown of each entry is rendered with the site's kramdown
# converter, then touched up:
#   - a paragraph made only of images (one per line, or several paragraphs in a
#     row) becomes `.moment-gallery.n-<count>` — 1 large, 2/4 two columns, 3+ a grid;
#     local images get cached 640px WebP thumbnails; gallery links keep originals;
#     a video URL in that image syntax becomes `.moment-video`, with the optional
#     title used as its local poster image;
#   - a blockquote keeps its line breaks (a poem), and when its last line starts
#     with —— / — / -- that line becomes `.moment-cite` (the attribution);
#   - a line that is just a URL of 网易云 / QQ 音乐 / Spotify / Apple Music, or of
#     an .mp3/.m4a/.ogg file, becomes a player card (`.moment-music`);
#   - `#标签` / `#读书/技术` anywhere in the text (flomo style: a `#` after a
#     space or at the start of a line, outside code and URLs) becomes an
#     `a.moment-tag` link to the tag's page and lands in `entry['tags']`.
#   - `[[id]]` references are code-aware: render empty anchors first so entry
#     text stays independent, then resolve links and newest-first backlinks.
# Every entry also gets a `/moments/<id>.html` page, shared-tag recommendations
# (excluding references), and `/moments/` remains a copy of the newest month.
# Discussions, views, and reactions stay keyed on the month URL.
#
# flomo-style extras, all built here so the layout only loops:
#   /moments/tag/<标签>.html   every tag and every ancestor of a `父/子` tag: the
#                              matching entries across all months (same layout,
#                              `is_tag`, no comments / reactions — those stay on
#                              the month pages the entries link back to);
#   site.data.moments.stats    { 'count', 'tags', 'days' } for the sidebar;
#   site.data.moments.tags     [{ 'tag', 'name', 'url', 'count', 'depth', 'last' }]
#                              sorted by recent use and count within each parent;
#   site.data.moments.heatmap  17- and 53-week grids ending at site.time, Monday first;
#   /moments/index.json        [{ id, url, date, month, time, place, tags, text,
#                                img, quote, refs }] for js/moments.js (sidebar /
#                                stream filters).
require 'json'
require 'date'
require 'cgi'
require 'fileutils'
require 'open3'

module Moments
  HEAD = /\A##\s+(\d{4})-(\d{2})-(\d{2})(?:\s+(\d{1,2}):(\d{2}))?(?:\s+@\s*(.+?))?\s*\z/
  IMG_P = %r{\A<p>\s*(?:<img\b[^>]*>\s*(?:<br\s*/?>)?\s*)+</p>\z}m
  VIDEO = /\.(?:mp4|mov|m4v|webm)(?:\?|#|$)/i
  # `#读书` `#跑步/马拉松` `#AI-infra`: a `#` not glued to a word (`C#`), a path
  # (`…/#/song`) or an entity (`&#39;`); letters, digits, `_` `-` `·`, `/` for
  # levels; stops at punctuation, so `#读书，` tags 读书. `#1` is not a tag.
  TAG = %r{(?<![\p{L}\p{N}_/&\\])#([\p{L}_][\p{L}\p{N}_\-·]*(?:/[\p{L}\p{N}_\-·]+)*)}
  REF = /\[\[(\d{8}(?:-\d{4})?(?:-\d+)?)\]\]/
  HEAT_WEEKS = 17
  HEAT_YEAR_WEEKS = 53
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

  class Thumbs
    WIDTH = 640

    def initialize(site)
      @site = site
      @cache = File.join(site.source, '.jekyll-cache', 'moment-thumbs')
      @registered = {}
      @cwebp_available = nil
      @warned_missing = false
    end

    def url(src)
      return src unless src&.start_with?('/img/moments/')
      return src unless cwebp_available?

      path = src.split(/[?#]/, 2).first
      relative = path.delete_prefix('/')
      source = File.expand_path(relative, @site.source)
      root = File.join(File.expand_path(@site.source), 'img', 'moments')
      return src unless source.start_with?("#{root}#{File::SEPARATOR}") && File.file?(source)

      source_stat = File.stat(source)
      relative_thumb = File.join(File.dirname(relative), 'thumb', "#{File.basename(relative, File.extname(relative))}.webp")
      cached = File.join(@cache, relative_thumb)
      marker = "#{cached}.source"
      key = "#{source_stat.mtime.to_r}:#{source_stat.size}"
      unless File.file?(cached) && File.file?(marker) && File.read(marker) == key && File.mtime(cached) > source_stat.mtime
        FileUtils.mkdir_p(File.dirname(cached))
        args = ["cwebp", "-q", "78"]
        width = image_width(source)
        args.concat(["-resize", WIDTH.to_s, "0"]) if width && width > WIDTH
        args.concat([source, "-o", cached])
        _, stderr, status = Open3.capture3(*args)
        unless status.success?
          Jekyll.logger.warn "moments:", "cwebp failed for #{src}: #{stderr.strip}"
          return src
        end
        File.write(marker, key)
        File.utime(File.atime(cached), [Time.now, source_stat.mtime + 1].max, cached)
      end

      unless @registered[relative_thumb]
        static_file = Jekyll::StaticFile.new(@site, @cache, File.dirname(relative_thumb), File.basename(relative_thumb))
        unless @site.static_files.any? { |file| file.destination(@site.dest) == static_file.destination(@site.dest) }
          @site.static_files << static_file
        end
        @registered[relative_thumb] = true
      end
      "/#{relative_thumb}"
    end

    private

    def cwebp_available?
      return @cwebp_available unless @cwebp_available.nil?

      _, _, status = Open3.capture3("cwebp", "-version")
      @cwebp_available = status.success?
    rescue Errno::ENOENT
      @cwebp_available = false
    ensure
      unless @cwebp_available || @warned_missing
        Jekyll.logger.warn "moments:", "cwebp not found; using original images"
        @warned_missing = true
      end
    end

    def image_width(path)
      output, _, status = Open3.capture3("file", "-b", path)
      return unless status.success?

      output.scan(/(\d+)\s*x\s*\d+/).last&.first&.to_i
    rescue Errno::ENOENT
      nil
    end
  end

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

  def self.tag_url(tag)
    "/moments/tag/#{tag}.html"
  end

  # `#读书/技术` → the link, outside fenced code, inline code and URLs. Returns
  # [markdown, tags found] — tags keep their written case, `父/子` stays one tag.
  def self.link_tags(md)
    tags = []
    fenced = false
    out = md.lines.map do |line|
      fenced = !fenced if line =~ /\A\s*(```|~~~)/
      next line if fenced || line =~ /\A\s{4}/ || music_card(line)
      # keep inline code as is: split on backticks, touch the even segments only
      line.split(/(`[^`]*`)/).each_with_index.map do |seg, i|
        next seg if i.odd?
        seg.gsub(TAG) do
          t = Regexp.last_match(1)
          tags << t unless tags.include?(t)
          %(<a class="moment-tag" href="#{tag_url(t)}">##{t}</a>)
        end
      end.join
    end.join
    [out, tags]
  end

  # `[[id]]` → an empty anchor resolved after every month's entries are parsed.
  def self.link_refs(md)
    refs = []
    fenced = false
    out = md.lines.map do |line|
      fenced = !fenced if line =~ /\A\s*(```|~~~)/
      next line if fenced || line =~ /\A\s{4}/ || music_card(line)
      line.split(/(`[^`]*`)/).each_with_index.map do |seg, i|
        next seg if i.odd?
        seg.gsub(REF) do
          id = Regexp.last_match(1)
          refs << id
          %(<a class="moment-ref" data-ref="#{id}"></a>)
        end
      end.join
    end.join
    [out, refs]
  end

  # Markdown of one entry → HTML with the gallery / quote / music touch-ups.
  def self.render(site, md, thumbs)
    lines = md.lines.map { |l| music_card(l) ? "\n#{music_card(l)}\n" : l }
    # quote lines break where the author broke them: `> a` / `> b` → a<br>b
    lines.each_with_index { |l, i| lines[i] = l.chomp + "  \n" if l =~ /\A>\s*\S/ && lines[i + 1] =~ /\A>\s*\S/ }
    md = lines.join
    html = site.find_converter_instance(Jekyll::Converters::Markdown).convert(md)
    html = html.gsub(/<img\b(?![^>]*\bloading=)/, '<img loading="lazy" decoding="async"')
    html = galleries(html, thumbs)
    quotes(html).strip
  end

  def self.galleries(html, thumbs)
    # consecutive image-only paragraphs merge into one gallery
    html.gsub(%r{(?:<p>\s*(?:<img\b[^>]*>\s*(?:<br\s*/?>)?\s*)+</p>\s*)+}m) do |run|
      imgs = run.scan(/<img\b[^>]*>/)
      items = imgs.map do |img|
        src = img[/\bsrc="([^"]*)"/, 1]
        if src.match?(VIDEO)
          poster_attr = img[/\btitle="([^"]*)"/, 1]
          poster = poster_attr && CGI.unescapeHTML(poster_attr)
          media = if poster
                    thumb = thumbs.url(poster)
                    %(<img src="#{thumb || poster}" alt="视频" loading="lazy" decoding="async"><span class="moment-play" aria-hidden="true"></span>)
                  else
                    %(<video src="#{src}#t=0.1" muted playsinline preload="metadata"></video>)
                  end
          data = poster ? %( data-poster="#{poster}") : ''
          next %(<a class="moment-pic moment-video" href="#{src}"#{data}>#{media}</a>)
        end
        thumb = thumbs.url(src)
        if thumb && thumb != src
          if imgs.size > 1
            img = img.sub(/\bsrc="[^"]*"/, %(src="#{thumb}"))
          else
            img = img.sub(/\/?>\z/) do |ending|
              %( srcset="#{thumb} 640w, #{src} 1600w" sizes="(max-width: 768px) 100vw, 600px"#{ending})
            end
          end
        end
        %(<a class="moment-pic" href="#{src}">#{img}</a>)
      end
      video_class = imgs.any? { |img| (img[/\bsrc="([^"]*)"/, 1] || '').match?(VIDEO) } ? ' has-video' : ''
      %(<div class="moment-gallery n-#{imgs.size}#{video_class}">#{items.join}</div>\n)
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

  def self.parse(site, page, thumbs)
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
      e['date'] = e['time'].strftime('%Y-%m-%d')
      e['url'] = "#{page.url}##{e['id']}"
      e['page'] = "/moments/#{e['id']}.html"
      e['month'] = page.url[/(\d{4}-\d{2})\.html\z/, 1]
      md, e['tags'] = link_tags(e['md'])
      md, e['refs_out'] = link_refs(md)
      e['html'] = render(site, md, thumbs)
      e['text'] = e['html'].gsub(%r{<(script|style|iframe|audio)\b.*?</\1>}m, ' ').gsub(/<[^>]+>/, ' ').gsub(/\s+/, ' ').strip
      image = e['html'][/<img\b[^>]*>/]
      srcset = image && image[/\bsrcset="([^"]*)"/, 1]
      e['img'] = srcset ? srcset.split(',').first.split.first : image && image[/\bsrc="([^"]*)"/, 1]
      e['video'] = e['html'][/<a class="moment-pic moment-video" href="([^"]+)"/, 1]
      e['original_img'] = if e['video']
                            e['html'][/<a class="moment-pic moment-video"[^>]*\bdata-poster="([^"]+)"/, 1]
                          else
                            e['html'][/<a class="moment-pic" href="([^"]+)"/, 1]
                          end
      e.delete('md')
    end
    entries.sort_by { |e| e['time'] }.reverse
  end

  # ---- the sidebar's numbers

  # All tags with counts and most recent use, parents before children.
  def self.tag_tree(entries)
    counts = Hash.new(0)
    last_used = {}
    entries.each do |e|
      seen = []
      e['tags'].each do |t|
        parts = t.split('/')
        parts.each_index { |i| seen << parts[0..i].join('/') }
      end
      seen.uniq.each do |tag|
        counts[tag] += 1
        last_used[tag] = e['date'] if !last_used[tag] || e['date'] > last_used[tag]
      end
    end
    nodes = counts.map do |tag, count|
      parts = tag.split('/')
      { 'tag' => tag, 'name' => parts.last, 'url' => tag_url(tag), 'count' => count,
        'depth' => parts.size - 1, 'last' => last_used[tag] }
    end
    children = nodes.group_by { |node| node['tag'].rpartition('/').first }
    order = lambda do |parent|
      (children[parent] || []).sort_by do |node|
        [-node['last'].delete('-').to_i, -node['count'], node['tag'].downcase]
      end.flat_map { |node| [node] + order.call(node['tag']) }
    end
    order.call('')
  end

  # Recent and full-year calendars share the same ending week and day records.
  def self.heatmap(entries, today)
    per_day = {}
    entries.each do |e|
      d = per_day[e['date']] ||= { 'count' => 0, 'url' => e['url'] }
      d['count'] += 1
    end
    recent = heatmap_weeks(per_day, today, HEAT_WEEKS)
    year = heatmap_weeks(per_day, today, HEAT_YEAR_WEEKS)
    { 'weeks' => recent['weeks'], 'months' => recent['months'],
      'year' => year, 'days' => per_day.size }
  end

  def self.heatmap_weeks(per_day, today, week_count)
    last = today + (7 - today.cwday) % 7            # this week's Sunday
    first = last - (week_count * 7 - 1)
    weeks = (0...week_count).map do |w|
      (0..6).map do |i|
        day = first + w * 7 + i
        key = day.strftime('%Y-%m-%d')
        n = per_day[key] ? per_day[key]['count'] : 0
        level = n == 0 ? 0 : [1 + Math.log2(n).floor, 4].min
        { 'date' => key, 'count' => n, 'url' => per_day[key] && per_day[key]['url'], 'level' => level, 'future' => day > today }
      end
    end
    months = []
    weeks.each_with_index do |week, col|
      day = Date.parse(week[0]['date'])
      months << { 'col' => col, 'label' => "#{day.month} 月" } if day.day <= 7 || col == 0
    end
    months.shift if months.size > 1 && months[1]['col'] < 2   # a label right at the edge would overlap the next
    { 'weeks' => weeks, 'months' => months }
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
      thumbs = Thumbs.new(site)
      pinned_ids = Array(site.data.dig('moments', 'pinned')).map(&:to_s)
      months = site.pages.select { |p| p.data['layout'] == 'moments' && p.data['month'] }
      months.each do |page|
        page.data['moments'] = Moments.parse(site, page, thumbs)
        page.data['comments_path'] = page.url
      end
      months.sort_by! { |p| p.data['month'] }.reverse!
      months.each_with_index do |p, i|
        p.data['newer'] = months[i - 1].url if i > 0
        p.data['older'] = months[i + 1].url if months[i + 1]
      end
      all = months.flat_map { |p| p.data['moments'] }
      all.each { |entry| entry['pinned'] = true if pinned_ids.include?(entry['id']) }
      (pinned_ids - all.map { |entry| entry['id'] }).each do |id|
        Jekyll.logger.warn 'moments:', "pinned entry #{id} does not exist"
      end
      entries_by_id = all.to_h { |e| [e['id'], e] }
      all.each { |e| e['refs_in'] = [] }
      all.sort_by { |e| e['time'] }.reverse_each do |source|
        source['refs_out'].uniq.each do |id|
          target = entries_by_id[id]
          if target
            target['refs_in'] << { 'id' => source['id'], 'title' => source['title'], 'page' => source['page'] }
          else
            Jekyll.logger.warn 'moments:', "#{source['id']} references unknown entry #{id}"
          end
        end
      end
      all.each do |e|
        e['html'] = e['html'].gsub(/<a class="moment-ref" data-ref="([^"]+)">\s*<\/a>/) do
          id = Regexp.last_match(1)
          target = entries_by_id[id]
          unless target
            next CGI.escapeHTML("[[#{id}]]")
          end

          text = target['text']
          preview = text.length > 60 ? "#{text[0, 59]}…" : text
          %(<a class="moment-ref" href="#{target['page']}"><span class="moment-ref-when">#{CGI.escapeHTML(target['title'])}</span><span class="moment-ref-text">#{CGI.escapeHTML(preview)}</span></a>)
        end
      end
      all.each do |entry|
        linked = [entry['id']] + entry['refs_out']
        linked.concat(all.select { |other| other['refs_out'].include?(entry['id']) }.map { |other| other['id'] })
        tags_for_entry = entry['tags'].flat_map do |tag|
          parts = tag.split('/')
          (1..parts.length).map { |length| parts.first(length).join('/') }
        end.uniq
        entry['related'] = all.filter_map do |other|
          next if linked.include?(other['id'])

          other_tags = other['tags'].flat_map do |tag|
            parts = tag.split('/')
            (1..parts.length).map { |length| parts.first(length).join('/') }
          end
          score = (tags_for_entry & other_tags).size
          next if score.zero?

          text = other['text']
          { 'id' => other['id'], 'title' => other['title'], 'page' => other['page'],
            'text' => (text.length > 60 ? "#{text[0, 59]}…" : text), 'img' => other['img'],
            'score' => score, 'distance' => (other['time'] - entry['time']).abs }
        end.sort_by { |other| [-other['score'], other['distance']] }.first(3).map do |other|
          other.reject { |key, _| %w[score distance].include?(key) }
        end
      end
      tags = Moments.tag_tree(all)
      heat = Moments.heatmap(all, site.time.to_date)
      site.data['moments'] = {
        'months' => months.map { |p| { 'url' => p.url, 'month' => p.data['month'], 'title' => p.data['title'], 'count' => p.data['moments'].size } },
        'entries' => all,
        'tags' => tags,
        'heatmap' => heat,
        'stats' => { 'count' => all.size, 'tags' => tags.count { |t| t['depth'] == 0 }, 'days' => heat['days'] }
      }
      return if months.empty?

      # /moments/ = the newest month, same discussion / counters
      latest = months.first
      index = Jekyll::PageWithoutAFile.new(site, site.source, 'moments', 'index.md')
      index.content = latest.content
      index.data = latest.data.merge('permalink' => '/moments/', 'canonical' => latest.url, 'sitemap' => false, 'is_index' => true)
      site.pages << index

      # /moments/<id>.html — a shareable, searchable page for each entry
      all.each do |e|
        text = e['text']
        description = text.length > 120 ? "#{text[0, 119]}…" : text
        month_url = e['url'].split('#', 2).first
        entry_page = Jekyll::PageWithoutAFile.new(site, site.source, 'moments', "#{e['id']}.md")
        entry_page.content = ''
        entry_page.data = {
          'layout' => 'moments', 'permalink' => e['page'], 'is_entry' => true, 'sitemap' => true,
          'moments' => [e], 'month' => e['month'], 'month_url' => month_url,
          'title' => "随笔 · #{e['title']}", 'description' => description,
          'header-img' => e['original_img'], 'comments_path' => month_url
        }.compact
        site.pages << entry_page
      end

      # /moments/tag/<标签>.html — the entries of a tag (and of its sub-tags) across months
      tags.each do |t|
        pg = Jekyll::PageWithoutAFile.new(site, site.source, 'moments/tag', "#{t['tag'].tr('/', '--')}.md")
        pg.content = ''
        pg.data = {
          'layout' => 'moments', 'permalink' => t['url'], 'sitemap' => false, 'is_tag' => true,
          'tag' => t['tag'], 'title' => "随笔 · ##{t['tag']}",
          'moments' => all.select { |e| e['tags'].any? { |x| x == t['tag'] || x.start_with?("#{t['tag']}/") } }
        }
        site.pages << pg
      end

      # /moments/index.json — what js/moments.js draws 随机漫步 / 每日回顾 from
      json = Jekyll::PageWithoutAFile.new(site, site.source, 'moments', 'index.json')
      json.content = JSON.generate(all.map do |e|
        { 'id' => e['id'], 'url' => e['page'], 'date' => e['date'], 'time' => (e['has_time'] ? e['time'].strftime('%H:%M') : nil),
          'month' => e['month'], 'place' => e['place'], 'tags' => e['tags'],
          'text' => CGI.unescapeHTML(e['text']), 'img' => e['img'], 'video' => e['video'],
          'quote' => e['html'].include?('class="moment-quote"'),
          'refs' => e['refs_out'].any? || e['refs_in'].any? }
      end)
      json.data = { 'layout' => nil, 'permalink' => '/moments/index.json', 'sitemap' => false }
      site.pages << json
    end
  end
end
