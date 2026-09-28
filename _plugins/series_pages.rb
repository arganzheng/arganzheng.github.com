# /series.html — one page, one tree: learning map → series → posts (Chirpy-style
# categories page, folded with <details>). No per-series pages.
#
# Inputs: `_data/series.yml` (key → name, overview, roadmap, number, shared_with),
# `_data/roadmaps.yml` (map key → name, url) and `series: <key>` in post front
# matter. This generator adds to each `site.data.series[key]`: `key`, `url`
# (/series.html#<key>), `count`, `status` (完结 when the last post is the recap,
# 连载中, 即将发布 when nothing is published yet), `first` / `last` dates and
# `posts` ([title, subtitle after 「）：」, url, date, recap]); then builds
# `site.data.series_index` = { roadmaps: [map + series (by number) + shared
# (unnumbered, by first post)], other: [series on no map] } and emits /series.html
# (layout series-index). Future-dated posts are not in site.posts on a normal
# build, so the counts are what is published today.
module SeriesPages
  RECAP = /-series-recap-and-self-test\.html\z/

  def self.subtitle(title)
    t = title.to_s
    t.include?('）：') ? t.split('）：', 2).last : t
  end

  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      series = site.data['series'] || {}
      roadmaps = site.data['roadmaps'] || {}
      by_url = site.posts.docs.each_with_object({}) { |p, h| h[p.url] = p }

      series.each do |key, meta|
        posts = site.posts.docs.select { |p| p.data['series'] == key }.sort_by(&:date)
        overview = by_url[meta['overview']]
        meta['key'] = key
        meta['url'] = "/series.html##{key}"
        meta['count'] = posts.size
        meta['status'] = posts.empty? ? '即将发布' : (posts.last.url =~ RECAP ? '完结' : '连载中')
        meta['first'] = posts.first && posts.first.date
        meta['last'] = posts.last && posts.last.date
        meta['overview_title'] = overview && overview.data['title']
        meta['posts'] = posts.map do |p|
          { 'title' => p.data['title'], 'subtitle' => SeriesPages.subtitle(p.data['title']), 'url' => p.url,
            'date' => p.date, 'recap' => !!(p.url =~ RECAP) }
        end
      end

      groups = roadmaps.map do |rkey, r|
        own = series.values.select { |s| s['roadmap'] == rkey }.sort_by { |s| s['number'].to_i }
        shared = series.values.select { |s| Array(s['shared_with']).include?(rkey) }.sort_by { |s| s['first'] || Time.at(0) }
        r.merge('key' => rkey, 'series' => own, 'shared' => shared, 'count' => (own + shared).sum { |s| s['count'] })
      end
      other = series.values.select { |s| s['roadmap'].nil? }.sort_by { |s| s['first'] || Time.at(0) }
      site.data['series_index'] = { 'roadmaps' => groups, 'other' => other,
                                    'total' => series.size, 'posts' => series.values.sum { |s| s['count'] } }

      index = Jekyll::PageWithoutAFile.new(site, site.source, '', 'series.html')
      index.content = ''
      index.data = { 'layout' => 'series-index', 'title' => 'Series', 'permalink' => '/series.html',
                     'description' => "#{series.size} 个系列 · #{site.data['series_index']['posts']} 篇文章" }
      site.pages << index
    end
  end
end
