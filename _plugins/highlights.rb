require_relative 'wikilinks'

module MarkHighlights
  HL = /(?<![=<!\\A-Za-z0-9_])==(?![\s=>])([^\n=]*?[^\s=\\])==(?![=>A-Za-z0-9_])/

  def self.rewrite(content)
    output = +''
    run = +''

    WikiLinks.tokenize(content).each do |token|
      if token.type == :protected
        output << highlight(run)
        run.clear
        output << token.raw
      else
        run << token.raw
      end
    end

    output << highlight(run)
    output
  end

  def self.highlight(text)
    text.gsub(HL) { %(<mark class="md-hl" markdown="span">#{Regexp.last_match(1)}</mark>) }
  end
  private_class_method :highlight

  [:posts, :private_posts].each do |collection|
    Jekyll::Hooks.register collection, :pre_render, priority: :low do |doc, _payload|
      doc.content = MarkHighlights.rewrite(doc.content)
    end
  end
end
