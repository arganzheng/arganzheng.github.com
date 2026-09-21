# A deck in `slides/` produces two pages from the one Markdown file:
#
#   /slides/foo.html        layout `slides` — the landing page (player in an
#                           iframe, meta line with views / likes / comments,
#                           every slide laid out flat below it, comments)
#   /slides/foo/play.html   layout `deck`   — the bare reveal.js presentation
#                           the player embeds; also what fullscreen, `?print-pdf`
#                           and `keynote` posts use
#
# The second one is generated here so authors keep writing a single file. It
# copies the front matter (so theme / transition / title-slide still apply) and
# stays out of the sitemap; its canonical URL is the landing page. The landing
# page learns where its deck lives through `page.deck`.
module SlidesDeck
  class Generator < Jekyll::Generator
    safe true
    priority :low   # after slides_date.rb has filled in `date` (a :post_read hook)

    def generate(site)
      decks = []
      site.pages.each do |page|
        next unless page.data['layout'] == 'slides'
        deck_url = page.url.sub(/\.html\z/, '') + '/play.html'
        page.data['deck'] = deck_url

        deck = Jekyll::PageWithoutAFile.new(site, site.source, File.dirname(page.path), 'play.md')
        deck.content = page.content
        deck.data = page.data.merge(
          'layout' => 'deck', 'permalink' => deck_url, 'sitemap' => false,
          'canonical' => page.url, 'source_path' => page.path
        )
        decks << deck
      end
      site.pages.concat(decks)
    end
  end
end
