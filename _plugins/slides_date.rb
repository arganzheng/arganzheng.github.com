# Slide decks live in `slides/` as pages, and pages — unlike posts — take no
# date from their filename. Give them the post convention: a deck named
# `slides/2026-08-01-reveal-demo.md` gets `date: 2026-08-01` unless its front
# matter already sets one. `archive.html`, `slides.html` and the slides layout
# all read `page.date`, so a dateless deck no longer floats to the top of the
# archive with an empty year.
Jekyll::Hooks.register :site, :post_read do |site|
  site.pages.each do |page|
    next unless page.data['layout'] == 'slides'
    next if page.data['date']
    m = File.basename(page.name).match(/\A(\d{4})-(\d{2})-(\d{2})-/)
    next unless m
    y, mo, d = m.captures.map(&:to_i)
    page.data['date'] = Time.new(y, mo, d)   # ENV['TZ'] is already site.timezone here
  end
end
