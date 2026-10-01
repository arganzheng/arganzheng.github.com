# Slide decks live in `slides/` as pages, and pages — unlike posts — take no
# date from their filename and are never held back for being future-dated.
# Give them the post convention on both counts:
#
# - a deck named `slides/2026-08-01-reveal-demo.md` gets `date: 2026-08-01`
#   unless its front matter already sets one (`archive.html`, `slides.html`
#   and the slides layout all read `page.date`, so a dateless deck no longer
#   floats to the top of the archive with an empty year);
# - a deck dated after `site.time` is dropped from a build without
#   `--future`, exactly like a post (the deploy has no `--future`; the daily
#   rebuild publishes it on its date). A series deck is dated the same day as
#   the series' 系列总结 post, so the two go live together. The `slides:` link
#   of such a series is removed from `site.data['series']` for that build, so
#   series-nav / series-deck / the /series/ tree do not point at a page that
#   is not there yet.
Jekyll::Hooks.register :site, :post_read do |site|
  held = []
  site.pages.reject! do |page|
    next false unless page.data['layout'] == 'slides'
    unless page.data['date']
      m = File.basename(page.name).match(/\A(\d{4})-(\d{2})-(\d{2})-/)
      page.data['date'] = Time.new(*m.captures.map(&:to_i)) if m   # ENV['TZ'] is already site.timezone here
    end
    date = page.data['date']
    next false if site.future || date.nil?
    date = page.data['date'] = Jekyll::Utils.parse_date(date.to_s) unless date.is_a?(Time)   # YAML gives a Date for `date: 2026-01-15`
    next false if date <= site.time
    held << page.url
    Jekyll.logger.info 'Slides:', "held back (future-dated #{date.strftime('%Y-%m-%d')}): #{page.path}"
    true
  end
  next if held.empty?
  (site.data['series'] || {}).each_value do |s|
    s.delete('slides') if s.is_a?(Hash) && held.include?(s['slides'])
  end
end
