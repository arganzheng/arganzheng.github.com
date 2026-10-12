require 'minitest/autorun'
require 'jekyll'
require_relative '../_plugins/highlights'

class MarkHighlightsTest < Minitest::Test
  def test_basic_highlight
    assert_equal '<mark class="md-hl" markdown="span">重点</mark>',
                 MarkHighlights.rewrite('==重点==')
  end

  def test_cjk_neighbors_are_allowed
    assert_equal '这是<mark class="md-hl" markdown="span">重点</mark>内容',
                 MarkHighlights.rewrite('这是==重点==内容')
  end

  def test_ascii_neighbors_arrows_spaces_and_escapes_are_ignored
    [
      'a==b==c',
      'a ==> b ==> c',
      '== x ==',
      '\\==x=='
    ].each do |source|
      assert_equal source, MarkHighlights.rewrite(source)
    end
  end

  def test_inline_fenced_and_math_code_is_untouched
    source = <<~'MARKDOWN'
      `==inline==`

      ```ruby
      ==fenced==
      ```

          ==indented==

      $$==display math==$$

      \(==inline math==\)
    MARKDOWN

    assert_equal source, MarkHighlights.rewrite(source)
  end

  def test_markdown_inside_highlight_is_preserved
    assert_equal '<mark class="md-hl" markdown="span">**粗**</mark>',
                 MarkHighlights.rewrite('==**粗**==')
  end

  def test_two_highlights_on_one_line
    assert_equal '<mark class="md-hl" markdown="span">一</mark> 和 <mark class="md-hl" markdown="span">二</mark>',
                 MarkHighlights.rewrite('==一== 和 ==二==')
  end

  def test_highlight_does_not_cross_newlines
    source = "==跨\n行=="
    assert_equal source, MarkHighlights.rewrite(source)
  end
end
