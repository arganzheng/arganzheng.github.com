# /series.html — one page, one tree: learning map → series → posts (Chirpy-style
# categories page, folded with <details>). Each series also has a /series/<key>/
# page with its posts.
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
# (layout series-index). Future-dated posts are not in site.posts on a normal
# build, so the counts are what is published today.
require 'cgi'

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

  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      series = site.data['series'] || {}
      roadmaps = site.data['roadmaps'] || {}
      by_url = site.posts.docs.each_with_object({}) { |p, h| h[p.url] = p }
      planned = SeriesPages.planned(site)

      series.each do |key, meta|
        posts = site.posts.docs.select { |p| p.data['series'] == key }.sort_by(&:date)
        overview = by_url[meta['overview']]
        meta['key'] = key
        meta['url'] = "/series/#{key}/"
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

        post_list = meta['posts'].map do |post|
          href = CGI.escapeHTML("#{site.baseurl}#{post['url']}")
          title = CGI.escapeHTML(post['title'].to_s)
          recap_class = post['recap'] ? ' class="is-recap"' : ''
          "<li#{recap_class}><a href=\"#{href}\">#{title}</a></li>"
        end.join("\n")
        overview = if meta['overview']
                     href = CGI.escapeHTML("#{site.baseurl}#{meta['overview']}")
                     "<p><a href=\"#{href}\">系列总纲</a></p>"
                   else
                     ''
                   end
        detail = Jekyll::PageWithoutAFile.new(site, site.source, 'series', File.join(key, 'index.html'))
        detail.content = "#{overview}<ol class=\"st-posts\">#{post_list}</ol>"
        detail.data = { 'layout' => 'page', 'title' => meta['name'], 'permalink' => "/series/#{key}/",
                        'description' => "#{meta['name']}：系列文章与总结" }
        site.pages << detail
      end

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
    end
  end
end
