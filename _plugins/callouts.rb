module Callouts
  TITLES = {
    'note' => '说明',
    'tip' => '提示',
    'important' => '重要',
    'warning' => '注意',
    'caution' => '警告'
  }.freeze

  ICONS = {
    'note' => 'M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0Zm0 1.5a6.5 6.5 0 1 1 0 13 6.5 6.5 0 0 1 0-13ZM7.25 7a.75.75 0 0 1 1.5 0v4a.75.75 0 0 1-1.5 0Zm.75-3a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z',
    'tip' => 'M8 0a6 6 0 0 0-3.7 10.72c.42.33.7.72.7 1.28v.25h6V12c0-.56.28-.95.7-1.28A6 6 0 0 0 8 0Zm0 1.5a4.5 4.5 0 0 1 2.78 8.04c-.53.42-.98 1-.98 1.96H6.2c0-.96-.45-1.54-.98-1.96A4.5 4.5 0 0 1 8 1.5ZM6.25 13h3.5a.75.75 0 0 1 0 1.5h-3.5a.75.75 0 0 1 0-1.5Zm.75 2.5h2a.75.75 0 0 1 0 1.5H7a.75.75 0 0 1 0-1.5Z',
    'important' => 'M3.5 1A2.5 2.5 0 0 0 1 3.5v9A2.5 2.5 0 0 0 3.5 15h9a2.5 2.5 0 0 0 2.5-2.5V5.56a2.5 2.5 0 0 0-.73-1.77l-2.06-2.06A2.5 2.5 0 0 0 10.44 1Zm.25 3h8.5a.75.75 0 0 1 0 1.5h-8.5a.75.75 0 0 1 0-1.5Zm0 3h8.5a.75.75 0 0 1 0 1.5h-8.5a.75.75 0 0 1 0-1.5Zm0 3h5a.75.75 0 0 1 0 1.5H4a.75.75 0 0 1 0-1.5Z',
    'warning' => 'M6.457 1.047c.813-1.396 2.833-1.396 3.646 0l5.87 10.077c.826 1.419-.198 3.216-1.823 3.216H2.41c-1.625 0-2.649-1.797-1.823-3.216l5.87-10.077ZM8 5.75a.75.75 0 0 1 .75.75v2.75a.75.75 0 0 1-1.5 0V6.5A.75.75 0 0 1 8 5.75Zm0 6.75a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z',
    'caution' => 'M3.58.5h8.84c.27 0 .52.11.71.29l2.08 2.08c.18.19.29.44.29.71v8.84c0 .27-.11.52-.29.71l-2.08 2.08a1 1 0 0 1-.71.29H3.58c-.27 0-.52-.11-.71-.29L.79 13.13a1 1 0 0 1-.29-.71V3.58c0-.27.11-.52.29-.71L2.87.79A1 1 0 0 1 3.58.5ZM3.9 2 2 3.9v8.2L3.9 14h8.2l1.9-1.9V3.9L12.1 2H3.9Z'
  }.freeze
  ICON_NAMES = {
    'note' => 'info',
    'tip' => 'light-bulb',
    'important' => 'report',
    'warning' => 'alert',
    'caution' => 'stop'
  }.freeze

  OPEN_BLOCKQUOTE = /<blockquote\b[^>]*>/i
  BLOCKQUOTE_TAG = /<\/?blockquote\b[^>]*>/i
  FIRST_PARAGRAPH = /\A(\s*(?:<!--[\s\S]*?-->\s*)*)(<p\b[^>]*>)([\s\S]*?)(<\/p\s*>)/i
  MARKER = /\A\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\](?:[ \t]+([^<\r\n]+))?(?:<br\s*\/?>[ \t\r\n]*)?/i

  def self.process(html)
    result = +''
    cursor = 0
    while (opening = OPEN_BLOCKQUOTE.match(html, cursor))
      closing = matching_close(html, opening.end(0))
      break unless closing

      result << html[cursor...opening.begin(0)]
      inner = process(html[opening.end(0)...closing.begin(0)])
      result << (convert(inner) || "#{opening[0]}#{inner}#{closing[0]}")
      cursor = closing.end(0)
    end
    result << html[cursor..] if cursor < html.length
    result
  end

  def self.matching_close(html, cursor)
    depth = 1
    while (tag = BLOCKQUOTE_TAG.match(html, cursor))
      if tag[0].start_with?('</')
        depth -= 1
        return tag if depth.zero?
      else
        depth += 1
      end
      cursor = tag.end(0)
    end
    nil
  end
  private_class_method :matching_close

  def self.convert(inner)
    paragraph = FIRST_PARAGRAPH.match(inner)
    return unless paragraph

    marker = MARKER.match(paragraph[3])
    return unless marker

    type = marker[1].downcase
    title = marker[2].to_s.strip
    title = TITLES.fetch(type) if title.empty?
    remaining = paragraph[3][marker[0].length..].to_s
    following = inner[paragraph.end(0)..].to_s
    body = if remaining.strip.empty?
      paragraph[1] + following
    else
      "#{paragraph[1]}#{paragraph[2]}#{remaining}#{paragraph[4]}#{following}"
    end
    icon = %(<svg class="callout-icon octicon octicon-#{ICON_NAMES.fetch(type)}" width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="#{ICONS.fetch(type)}"/></svg>)

    %(<div class="callout callout-#{type}"><p class="callout-title">#{icon}#{title}</p>#{body}</div>)
  end
  private_class_method :convert
end

Jekyll::Hooks.register :documents, :post_convert do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'

  converted = Callouts.process(doc.content)
  doc.content = converted
  doc.output = Callouts.process(doc.output.to_s.empty? ? converted : doc.output)
end
