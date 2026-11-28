require 'minitest/autorun'
require 'jekyll'
require_relative '../_plugins/code_lines'

class CodeLinesTest < Minitest::Test
  def wrap(inner)
    CodeLines.wrap(inner)
  end

  def test_plain_block_counts_its_lines
    html, count, = wrap("a\nb\n")
    assert_equal 2, count
    assert_equal %(<span class="line">a</span>\n<span class="line">b</span>\n), html
  end

  # Rouge's Python lexer keeps the newline inside a trailing comment token.
  def test_final_newline_inside_the_last_token_adds_no_line
    html, count, = wrap(%(<span class="n">x</span>\n<span class="n">y</span> <span class="c1"># 原地写回\n</span>))
    assert_equal 2, count
    refute_match %r{<span class="line"><span class="c1"></span></span>}, html
    assert html.end_with?(%(<span class="c1"># 原地写回</span></span>\n))
  end

  def test_final_newline_inside_nested_tokens
    _, count, = wrap(%(<span class="s"><span class="si">a\n</span></span>))
    assert_equal 1, count
  end

  def test_blank_line_in_the_middle_is_kept
    _, count, = wrap("a\n\nb\n")
    assert_equal 3, count
  end

  def test_text_content_is_unchanged
    inner = %(<span class="n">x</span> <span class="c1"># c\n</span><span class="n">y</span>\n)
    html, = wrap(inner)
    assert_equal CodeLines.text_of(inner).chomp, CodeLines.text_of(html).chomp
  end
end
