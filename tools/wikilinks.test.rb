require 'benchmark'
require 'minitest/autorun'
require 'jekyll'
require_relative '../_plugins/wikilinks'

class WikiLinksTest < Minitest::Test
  def test_tokenizes_labeled_heading_links_with_chinese_neighbors
    tokens = WikiLinks.tokenize('中文 [[article#heading|标题]] 内容')

    link = tokens.find { |token| token.type == :wikilink }
    assert_equal '[[article#heading|标题]]', link.raw
    assert_equal 'article#heading', link.target
    assert_equal '标题', link.label
    assert_equal Encoding::UTF_8, link.raw.encoding
    assert_equal Encoding::UTF_8, link.target.encoding
    assert_equal Encoding::UTF_8, link.label.encoding
    assert_equal '中文 ', tokens.first.raw
    assert_equal ' 内容', tokens.last.raw
  end

  def test_wikilinks_in_code_and_math_remain_protected
    source = <<~'MARKDOWN'
      中文 [[visible]]

      ```ruby
      [[fenced]]
      ```

      `[[inline]]` $$[[math]]$$ [[also-visible]]
    MARKDOWN

    tokens = WikiLinks.tokenize(source)

    assert_equal %w[visible also-visible],
                 tokens.select { |token| token.type == :wikilink }.map(&:target)
    assert_includes tokens.select { |token| token.type == :protected }.map(&:raw).join,
                    "[[fenced]]"
    assert_includes tokens.select { |token| token.type == :protected }.map(&:raw).join,
                    '`[[inline]]`'
    assert_includes tokens.select { |token| token.type == :protected }.map(&:raw).join,
                    '$$[[math]]$$'
  end

  def test_all_token_string_fields_use_source_encoding
    tokens = WikiLinks.tokenize("中文 [[页面|标签]] `代码`")

    tokens.each do |token|
      [:raw, :target, :label].each do |field|
        value = token.public_send(field)
        assert_equal Encoding::UTF_8, value.encoding if value
      end
    end
  end

  def test_large_multibyte_document_tokenizes_quickly
    source = Array.new(4_000) do
      '中文段落内容 [[x]] 更多中文内容 `代码` 结尾文字' + "\n"
    end.join
    assert_operator source.bytesize, :>, 190_000

    elapsed = Benchmark.realtime { WikiLinks.tokenize(source) }

    assert_operator elapsed, :<, 1.0, "tokenization took #{elapsed.round(3)}s"
  end
end
