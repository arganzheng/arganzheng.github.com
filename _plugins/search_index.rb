# Site search index for js/search.js — replaces the old search.json +
# search-content.ndjson pair, which made the browser download every post's
# full text (15 MB, 5.5 MB gzipped) before the first keystroke.
#
# Emitted under /search/ (all as pages, so `jekyll build` cleans them up like
# any other output):
#   meta.json        [[url, title, "YYYY-MM-DD", [tags…]], …]   one entry per post, id = index
#   idx/<0..N-1>.json {key: [id deltas…]}                         inverted index, keys hashed into N buckets
#   doc/<slug>.txt   plain text of one post (title not included)
#
# Keys are what the client can derive from a query without a tokenizer, so the
# matching semantics stay exactly those of the old in-memory search:
#   - ASCII words `[a-z0-9_]+` (the runs JS `\b` delimits)  → whole-word hit
#   - `~part` for each `_`-separated part of such a word   → substring hit inside identifiers
#   - every CJK character and every CJK bigram inside a run  → substring hit
# A query term becomes the AND of its keys; the client then fetches the text of
# the posts it is about to display and re-checks the real match there (bigram
# chains for 3+ character terms and non-word ASCII terms are approximations),
# so nothing is shown that the old code would not have shown.
#
# The bucket hash must stay in sync with `bucketOf` in js/search.js.
require 'cgi'
require 'json'

module SearchIndex
  BUCKETS = 256
  CJK = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]/
  CJK_RUN = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]+/
  BLOCK_TAG = %r{</?(?:p|div|li|ul|ol|h[1-6]|pre|br|hr|tr|td|th|table|thead|tbody|blockquote|section|figure|figcaption|dt|dd|details|summary)\b[^>]*>}i

  # Rendered HTML -> searchable plain text. Block tags become a space, inline
  # tags vanish (so `分布式<strong>锁</strong>` still contains `分布式锁`).
  def self.text_of(html)
    t = html.to_s
      .gsub(%r{<script.*?</script>}mi, ' ')
      .gsub(%r{<style.*?</style>}mi, ' ')
      .gsub(/<!--.*?-->/m, ' ')
      .gsub(BLOCK_TAG, ' ')
      .gsub(/<[^>]*>/m, '')
      .gsub('&nbsp;', ' ')
    CGI.unescapeHTML(t).gsub(/\s+/, ' ').strip
  end

  def self.keys_of(text)
    keys = text.downcase.scan(/[a-z0-9_]+/)
    # `load_state_dict` is one word to `\b`, but a substring search for
    # `state_dict` must still find it: file its parts under `~part`.
    keys.grep(/_/).each { |w| w.split('_').each { |p| keys << "~#{p}" unless p.empty? } }
    text.scan(CJK_RUN).each do |run|
      chars = run.chars
      keys.concat(chars)
      keys.concat(chars.each_cons(2).map(&:join)) if chars.size > 1
    end
    keys.uniq
  end

  # Same arithmetic as JS: h = (h * 31 + charCodeAt(i)) >>> 0 over UTF-16 units.
  def self.bucket_of(key)
    h = 0
    key.encode('UTF-16LE').unpack('v*').each { |c| h = (h * 31 + c) & 0xffffffff }
    h % BUCKETS
  end

  def self.page(site, dir, name, body)
    page = Jekyll::PageWithoutAFile.new(site, site.source, dir, name)
    page.content = body
    page.output = body
    page.data['sitemap'] = false
    page
  end
end

Jekyll::Hooks.register :site, :post_render do |site|
  posts = site.posts.docs.sort_by { |p| p.date }.reverse
  meta = []
  postings = Hash.new { |h, k| h[k] = [] }
  pages = []

  posts.each_with_index do |post, id|
    url = File.join(site.baseurl.to_s, post.url)
    title = post.data['title'].to_s
    text = SearchIndex.text_of(post.content)
    meta << [url, title, post.date.strftime('%Y-%m-%d'), Array(post.data['tags']).map(&:to_s)]
    SearchIndex.keys_of("#{title} #{text}").each { |k| postings[k] << id }
    pages << SearchIndex.page(site, '/search/doc', "#{File.basename(post.url, '.html')}.txt", text)
  end

  buckets = Array.new(SearchIndex::BUCKETS) { {} }
  postings.each do |key, ids|
    prev = -1
    buckets[SearchIndex.bucket_of(key)][key] = ids.map { |i| d = i - prev - 1; prev = i; d }
  end
  buckets.each_with_index do |b, i|
    pages << SearchIndex.page(site, '/search/idx', "#{i}.json", JSON.generate(b))
  end
  pages << SearchIndex.page(site, '/search', 'meta.json', JSON.generate(meta))

  site.pages.concat(pages)
  Jekyll.logger.info 'Search index:', "#{posts.size} posts, #{postings.size} keys, #{SearchIndex::BUCKETS} buckets"
end
