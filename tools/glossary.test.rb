require 'benchmark'
require 'minitest/autorun'
require 'jekyll'

require ENV.fetch('GLOSSARY_PLUGIN') { File.expand_path('../_plugins/glossary.rb', __dir__) }

class GlossaryTest < Minitest::Test
  def test_finds_segments_outside_blocked_elements_inline_tips_and_math
    html = %q{<p title="a > b">甲<b>乙</b><code>丙</code><span class="inline-tip" data-tip="x">丁</span>\(戊\)<i>己</i></p>}.b

    assert_equal [[17, 3], [23, 3], [104, 3]], Glossary.text_segments(html)
  end

  def test_large_plain_paragraph_document_has_bounded_scans
    source = Array.new(10_500) { "<p>#{'中文段落'}</p>" }.join.b
    assert_operator source.bytesize, :>, 190_000

    elapsed = Benchmark.realtime { Glossary.text_segments(source) }

    assert_operator elapsed, :<, 0.5, "text_segments took #{elapsed.round(3)}s"
  end
end
