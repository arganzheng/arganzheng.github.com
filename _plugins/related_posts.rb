# Related posts ("YOU MIGHT ALSO LIKE"), precomputed once per build and stored
# in post.data['related'] for _includes/related-posts.html.
#
# The old Liquid version took the *newest* posts sharing *any* tag, so every
# `Java` post recommended the three latest interview posts. Here:
#   - shared tags are weighted by rarity (IDF: log(N / posts_with_tag)), so
#     `quartz` counts far more than `AI`; the sum is normalised by the tag
#     counts of both posts so tag-heavy posts do not dominate;
#   - members of the current post's own series are excluded (they already have
#     上一篇/下一篇 and the series TOC), and at most one post per other series
#     is taken so a single series cannot fill all slots;
#   - candidates stay within the same category (tech / life) and must be
#     published;
#   - ties go to the newer post; when fewer than N posts share a tag the list
#     is topped up with the newest same-category posts.
# N = site.related_posts_threshold (default 3). `:site, :post_read` runs after
# front matter is loaded and before rendering, like home_flow.rb.
Jekyll::Hooks.register :site, :post_read do |site|
  n = (site.config['related_posts_threshold'] || 3).to_i
  posts = site.posts.docs.reject { |p| p.data['published'] == false }
  tags_of = posts.to_h { |p| [p, Array(p.data['tags']).map(&:to_s).reject(&:empty?).uniq] }
  df = Hash.new(0)
  tags_of.each_value { |ts| ts.each { |t| df[t] += 1 } }
  total = posts.size.to_f
  idf = df.to_h { |t, c| [t, Math.log(total / c)] }

  by_tag = Hash.new { |h, k| h[k] = [] }
  tags_of.each { |p, ts| ts.each { |t| by_tag[t] << p } }

  entry = lambda do |p, shared|
    { 'title' => p.data['title'], 'url' => p.url, 'date' => p.date,
      'shared_tags' => shared }
  end

  posts.each do |post|
    mine = tags_of[post]
    scores = Hash.new(0.0)
    mine.each do |t|
      by_tag[t].each { |o| scores[o] += idf[t] unless o.equal?(post) }
    end
    scores.each_key { |o| scores[o] /= Math.sqrt(mine.size * tags_of[o].size) }

    same_cat = ->(o) { o.data['category'] == post.data['category'] }
    ranked = scores.keys.select(&same_cat)
    ranked.reject! { |o| post.data['series'] && o.data['series'] == post.data['series'] }
    ranked.sort_by! { |o| [-scores[o], -o.date.to_i] }

    picked, series_seen = [], {}
    ranked.each do |o|
      break if picked.size >= n
      s = o.data['series']
      next if s && series_seen[s]
      series_seen[s] = true if s
      shared = (mine & tags_of[o]).sort_by { |t| -idf[t] }
      picked << entry.call(o, shared)
    end

    if picked.size < n
      taken = picked.map { |e| e['url'] }
      posts.select { |o| !o.equal?(post) && same_cat.call(o) && !taken.include?(o.url) &&
                         !(post.data['series'] && o.data['series'] == post.data['series']) }
           .sort_by { |o| -o.date.to_i }
           .first(n - picked.size)
           .each { |o| picked << entry.call(o, []) }
    end

    post.data['related'] = picked
  end
end
