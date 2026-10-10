# Backlinks for published posts. Scan the original Markdown once after posts
# have been read, resolve current URLs and redirect aliases, then expose each
# source post on its target for _includes/backlinks.html.
require 'uri'

module Backlinks
  def self.normalized_path(path, baseurl)
    path = path.to_s
    base = baseurl.to_s.sub(%r{/\z}, '')
    path = path.delete_prefix(base) if !base.empty? && (path == base || path.start_with?("#{base}/"))
    path = URI::DEFAULT_PARSER.unescape(path)
    path = "/#{path}" unless path.start_with?('/')
    path
  end

  def self.source_text(content)
    content.to_s
      .gsub(/^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$/m, ' ')
      .gsub(/(`+)(?!`)[^\n]*?\1(?!`)/, ' ')
  end
end

Jekyll::Hooks.register :site, :post_read do |site|
  posts = site.posts.docs.reject { |post| post.data['published'] == false }
  host = URI.parse(site.config['url'].to_s).host
  next unless host

  url_pattern = %r{(?:https?://(?:www\.)?#{Regexp.escape(host)})?(/[^)\s"'#?>]+)}i
  markdown_link = %r{\]\(\s*<?#{url_pattern}}i
  html_link = %r{href\s*=\s*["']#{url_pattern}}i
  reference_link = %r{^\s*\[[^\]]+\]:\s*<?#{url_pattern}}im
  by_url = {}

  posts.each do |post|
    by_url[Backlinks.normalized_path(post.url, site.baseurl)] = post
    Array(post.data['redirect_from']).each do |alias_url|
      path = alias_url.to_s
      path = URI.parse(path).path if path.match?(%r{\Ahttps?://}i)
      by_url[Backlinks.normalized_path(path, site.baseurl)] = post
    end
    post.data['backlinks'] = []
  end

  sources_by_target = Hash.new { |hash, target| hash[target] = {} }
  posts.each do |source|
    text = Backlinks.source_text(source.content)
    paths = text.scan(markdown_link).flatten
      .concat(text.scan(html_link).flatten)
      .concat(text.scan(reference_link).flatten)

    paths.each do |path|
      target = by_url[Backlinks.normalized_path(path, site.baseurl)]
      next unless target && !target.equal?(source)

      sources_by_target[target][source] = true
    end
  end

  links = 0
  sources_by_target.each do |target, sources|
    backlinks = sources.keys.sort_by { |source| -source.date.to_i }.map do |source|
      series_key = source.data['series']
      series = (site.data['series'] || {})[series_key]
      { 'title' => source.data['title'], 'url' => source.url, 'date' => source.date,
        'series_name' => series && series['name'] }
    end
    target.data['backlinks'] = backlinks
    links += backlinks.size
  end

  with_backlinks = sources_by_target.count { |_target, sources| !sources.empty? }
  Jekyll.logger.info 'Backlinks:', "#{links} links, #{with_backlinks} posts with backlinks"
end
