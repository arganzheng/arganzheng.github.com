require 'minitest/autorun'
require 'jekyll'
require 'kramdown'
require_relative '../_plugins/table_rowspan'

class TableRowspanTest < Minitest::Test
  def render(md)
    TableRowspan.process(Kramdown::Document.new(md).to_html)
  end

  def cells(html)
    html.scan(%r{<t[dh]\b[^>]*>.*?</t[dh]>}m).map { |c| c.gsub(/\s+/, ' ') }
  end

  def test_consecutive_marks_become_one_rowspan
    html = render(<<~MD)
      | 层 | 指标 |
      |---|---|
      | 任务 | loss |
      | ^^ | step 时间 |
      | ^^ | MFU |
      | 进程 | 显存 |
    MD
    assert_includes html, '<td rowspan="3">任务</td>'
    refute_includes html, '^^'
    assert_equal 2 + 6, cells(html).size
    assert_includes html, '<td>进程</td>'
  end

  def test_mark_under_header_is_left_as_text
    html = render("| a | b |\n|---|---|\n| ^^ | 1 |\n| x | 2 |\n")
    assert_includes html, '<td>^^</td>'
    refute_includes html, 'rowspan'
  end

  def test_columns_are_independent
    html = render("| a | b |\n|---|---|\n| 1 | 2 |\n| ^^ | 3 |\n| 4 | ^^ |\n")
    assert_includes html, '<td rowspan="2">1</td>'
    assert_includes html, '<td rowspan="2">3</td>'
    assert_includes html, '<td>4</td>'
    refute_includes html, '^^'
  end

  def test_alignment_attributes_survive
    html = render("| a | b |\n|:--|--:|\n| 1 | 2 |\n| ^^ | 3 |\n")
    assert_match %r{<td style="text-align: left" rowspan="2">1</td>}, html
  end

  def test_all_mark_row_disappears
    html = render("| a | b |\n|---|---|\n| 1 | 2 |\n| ^^ | ^^ |\n")
    assert_equal 1, html.scan('<tr>').size - 1
    assert_includes html, 'rowspan="2">1<'
  end

  def test_tables_without_marks_are_untouched
    src = Kramdown::Document.new("| a | b |\n|---|---|\n| 1 | 2 |\n").to_html
    assert_equal src, TableRowspan.process(src)
  end

  def test_caret_in_code_is_not_a_mark
    html = render("| a | b |\n|---|---|\n| 1 | 2 |\n| `^^` | 3 |\n")
    assert_includes html, '<td><code>^^</code></td>'
    refute_includes html, 'rowspan'
  end
end
