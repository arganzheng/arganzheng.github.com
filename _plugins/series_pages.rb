# /series.html — one page, one tree: learning map → series → posts (Chirpy-style
# categories page, folded with <details>). No per-series pages.
#
# Inputs: `_data/series.yml` (key → name, overview, roadmap, number, shared_with),
# `_data/roadmaps.yml` (map key → name, url) and `series: <key>` in post front
# matter. This generator adds to each `site.data.series[key]`: `key`, `url`
# (/series.html#<key>), `count` (published), `planned` (every _posts file, future
# ones too), `body_count` / `body_planned` (recap excluded), `hours` (planned
# body at 450 字/min — the roadmap posts' rule), `status` (完结 when the last
# post is the recap, 连载中, 即将发布 when nothing is published yet), `first` /
# `last` dates and `posts` ([title, subtitle after 「）：」, url, date, recap]).
# `_includes/series-row.html` renders a roadmap table row from these; then builds
# `site.data.series_index` = { roadmaps: [map + series (by number) + shared
# (unnumbered, by first post)], other: [series on no map] } and emits /series.html
# (layout series-index) and /series-map.json (public posts only). Future-dated
# posts are not in site.posts on a normal build, so the counts are what is
# published today.
require 'json'

module SeriesPages
  RECAP = /-series-recap-and-self-test(\.html|\.md)?\z/
  CJK = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]/
  # the roadmap posts' 时长 rule: 450 字/min, code included, recap excluded
  CHARS_PER_MINUTE = 450

  def self.subtitle(title)
    t = title.to_s
    t.include?('）：') ? t.split('）：', 2).last : t
  end

  # Every _posts file with `series:` — including future-dated ones, which a
  # normal build leaves out of site.posts — so the roadmap tables can show the
  # planned size of a series (`planned`, `hours`) next to what is published.
  #   { key => { 'planned' => n, 'body_planned' => n (recap excluded), 'minutes' => m (body) } }
  def self.planned(site)
    out = Hash.new { |h, k| h[k] = { 'planned' => 0, 'body_planned' => 0, 'minutes' => 0 } }
    Dir.glob(File.join(site.source, '_posts', '**', '*.{md,markdown}')).each do |file|
      src = File.read(file, encoding: 'utf-8')
      next unless (fm = src.match(/\A---\s*\n(.*?)\n---\s*\n/m))
      next unless (key = fm[1][/^series:\s*['"]?([\w-]+)/, 1])
      next if fm[1] =~ /^published:\s*false/
      out[key]['planned'] += 1
      next if File.basename(file) =~ RECAP
      body = src[fm[0].size..].gsub(/<[^>]+>/, ' ')
      out[key]['body_planned'] += 1
      out[key]['minutes'] += (body.scan(CJK).size + body.scan(/[A-Za-z0-9_]+/).size) / CHARS_PER_MINUTE.to_f
    end
    out
  end

  # Reading order of each learning map = the order of the
  # `{% include series-row.html key="…" %}` rows in its roadmap post (the map's
  # 系列总览 table, shared series included where they are read); without the
  # post, own series by `number` then shared ones. Sets roadmaps[k]['order']
  # (series keys) and, on every series, `maps` = [{ key, name, short, url, index,
  # label, total, prev, next }] (prev / next are series keys; own map first) — read by
  # _includes/roadmap-box.html (overview posts) and series-nav.html.
  def self.link_maps(series, roadmaps, by_url)
    series.each_value { |m| m['maps'] = [] }
    roadmaps.each do |rkey, r|
      post = by_url[r['url']]
      keys = post ? post.content.scan(/series-row\.html\s+key="([\w-]+)"/).flatten.uniq.select { |k| series[k] } : []
      if keys.empty?
        own = series.select { |_, m| m['roadmap'] == rkey }.sort_by { |_, m| m['number'].to_i }.map(&:first)
        shared = series.select { |_, m| Array(m['shared_with']).include?(rkey) }.sort_by { |_, m| m['first'] || Time.at(0) }.map(&:first)
        keys = own + shared
      end
      r['order'] = keys
      keys.each_with_index do |k, i|
        entry = { 'key' => rkey, 'name' => r['name'], 'short' => r['short'], 'label' => r['short'].to_s =~ /\A\p{Han}/ ? "#{r['short']}地图" : "#{r['short']} 地图", 'url' => r['url'],
                  'index' => i + 1, 'total' => keys.size, 'prev' => i.positive? ? keys[i - 1] : nil, 'next' => keys[i + 1] }
        series[k]['roadmap'] == rkey ? series[k]['maps'].unshift(entry) : series[k]['maps'] << entry
      end
    end
  end

  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      series = site.data['series'] || {}
      roadmaps = site.data['roadmaps'] || {}
      by_url = site.posts.docs.each_with_object({}) { |p, h| h[p.url] = p }
      planned = SeriesPages.planned(site)
      series_map = {}

      series.each do |key, meta|
        posts = site.posts.docs.select { |p| p.data['series'] == key }.sort_by(&:date)
        posts.each do |p|
          body = p.content.to_s.gsub(/<[^>]+>/, ' ')
          chars = body.scan(CJK).size + body.scan(/[A-Za-z0-9_]+/).size
          p.data['series_minutes'] = [(chars / CHARS_PER_MINUTE.to_f).ceil, 1].max
        end
        public_posts = posts.reject do |p|
          p.data['published'] == false || p.data['visibility'] == 'private'
        end
        unless public_posts.empty?
          series_map[key] = {
            'name' => meta['name'],
            'posts' => public_posts.map { |p| [p.url, SeriesPages.subtitle(p.data['title'])] }
          }
        end
        overview = by_url[meta['overview']]
        meta['key'] = key
        meta['url'] = "/series.html##{key}"
        meta['count'] = posts.size
        meta['body_count'] = posts.count { |p| p.url !~ RECAP }
        meta['planned'] = [planned[key]['planned'], posts.size].max
        meta['body_planned'] = [planned[key]['body_planned'], meta['body_count']].max
        meta['hours'] = (planned[key]['minutes'] / 60.0).round(1)
        meta['status'] = posts.empty? ? '即将发布' : (posts.last.url =~ RECAP ? '完结' : '连载中')
        meta['first'] = posts.first && posts.first.date
        meta['last'] = posts.last && posts.last.date
        meta['overview_title'] = overview && overview.data['title']
        meta['posts'] = posts.map do |p|
          { 'title' => p.data['title'], 'subtitle' => SeriesPages.subtitle(p.data['title']), 'url' => p.url,
            'date' => p.date, 'recap' => !!(p.url =~ RECAP) }
        end
      end

      SeriesPages.link_maps(series, roadmaps, by_url)

      groups = roadmaps.map do |rkey, r|
        own = series.values.select { |s| s['roadmap'] == rkey }.sort_by { |s| s['number'].to_i }
        shared = series.values.select { |s| Array(s['shared_with']).include?(rkey) }.sort_by { |s| s['first'] || Time.at(0) }
        r.merge('key' => rkey, 'series' => own, 'shared' => shared, 'count' => (own + shared).sum { |s| s['count'] },
                'hours' => own.sum { |s| s['hours'] }.round(1))
      end
      other = series.values.select { |s| s['roadmap'].nil? }.sort_by { |s| s['first'] || Time.at(0) }
      site.data['series_index'] = { 'roadmaps' => groups, 'other' => other,
                                    'total' => series.size, 'posts' => series.values.sum { |s| s['count'] } }

      index = Jekyll::PageWithoutAFile.new(site, site.source, '', 'series.html')
      index.content = ''
      index.data = { 'layout' => 'series-index', 'title' => '系列文章', 'permalink' => '/series.html',
                     'description' => '成体系地读：按学习地图组织，地图 → 系列 → 单篇，点开一层即可展开' }
      site.pages << index

      map = Jekyll::PageWithoutAFile.new(site, site.source, '', 'series-map.json')
      map.content = JSON.generate('series' => series_map)
      map.data = { 'layout' => nil, 'sitemap' => false, 'permalink' => '/series-map.json' }
      site.pages << map
    end
  end
end
