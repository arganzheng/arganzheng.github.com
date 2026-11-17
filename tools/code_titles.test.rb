require 'minitest/autorun'
require 'jekyll'
require 'kramdown'
require 'kramdown-parser-gfm'
require_relative '../_plugins/code_titles'

class CodeTitlesTest < Minitest::Test
  def html(md)
    Kramdown::Document.new(CodeTitles.process(md), input: 'GFM', syntax_highlighter: nil).to_html
  end

  def test_plain_fence_gets_data_title
    out = html("正文。\n\n```python title=\"一个例子\"\nprint(1)\n```\n")
    assert_includes out, 'data-title="一个例子"'
    assert_includes out, 'print(1)'
    refute_includes out, 'title=&quot;'
  end

  def test_fence_directly_after_text_gets_a_blank_line
    src = CodeTitles.process("正文。\n```python title='紧跟文字'\nprint(1)\n```\n")
    assert_equal "正文。\n\n{: data-title=\"紧跟文字\"}\n```python\nprint(1)\n```\n", src
  end

  def test_fence_inside_blockquote
    md = "> [!NOTE] 与 Java 对照\n> 下面是定义：\n>\n> ```java title='Java record'\n> public record User(Long id) {}\n> ```\n"
    src = CodeTitles.process(md)
    assert_includes src, "> {: data-title=\"Java record\"}\n> ```java\n"
    out = html(md)
    assert_match(/<blockquote>.*data-title="Java record".*<\/blockquote>/m, out)
    assert_includes out, 'public record User(Long id) {}'
    refute_includes out, '```'
  end

  def test_fence_in_blockquote_directly_after_text_keeps_the_quote
    src = CodeTitles.process("> 说明：\n> ```java title=\"X\"\n> int a;\n> ```\n")
    assert_equal "> 说明：\n>\n> {: data-title=\"X\"}\n> ```java\n> int a;\n> ```\n", src
    out = html("> 说明：\n> ```java title=\"X\"\n> int a;\n> ```\n")
    assert_equal 1, out.scan('<blockquote>').size
  end

  def test_nested_blockquote
    md = "> 外层\n>\n> > ```sql title=\"嵌套\"\n> > SELECT 1;\n> > ```\n"
    assert_includes CodeTitles.process(md), "> > {: data-title=\"嵌套\"}\n> > ```sql\n"
    assert_includes html(md), 'data-title="嵌套"'
  end

  def test_existing_ial_in_blockquote_is_merged
    src = CodeTitles.process("> {:.no-lineno}\n> ```text title=\"输出\"\n> ok\n> ```\n")
    assert_equal "> {:.no-lineno data-title=\"输出\"}\n> ```text\n> ok\n> ```\n", src
  end

  def test_fence_shown_inside_a_longer_fence_is_left_alone
    md = "````markdown\n```cpp title=\"示例语法\"\nint a;\n```\n````\n"
    assert_equal md, CodeTitles.process(md)
  end

  def test_quoted_line_inside_a_top_level_fence_does_not_close_it
    md = "```text title=\"邮件\"\n> ```\n仍在代码块里\n```\n"
    out = html(md)
    assert_includes out, '仍在代码块里'
    assert_equal 1, out.scan('<pre').size
  end

  def test_fence_without_title_is_untouched
    md = "> ```java\n> int a;\n> ```\n"
    assert_equal md, CodeTitles.process(md)
  end
end
