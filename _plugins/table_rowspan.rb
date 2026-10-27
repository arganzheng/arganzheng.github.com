# Merged cells in Markdown tables, MultiMarkdown style: a body cell that
# contains only `^^` is merged into the cell above it (same column), which
# gets `rowspan`. kramdown has no syntax for this, so the merge happens here,
# on the rendered HTML, like table_captions.rb / table_lists.rb.
#
#   | 层 | 指标 |
#   |---|---|
#   | 任务 | loss |
#   | ^^   | step 时间 |
#   | 进程 | 显存 |
#
# -> <td rowspan="2">任务</td> on the first row, and the `^^` cell is gone.
#
# Only <td> merges into a <td>: a `^^` on the first body row (under a <th>)
# is left as text, so a header never grows a rowspan. A row whose cells are
# all `^^` disappears. kramdown never emits rowspan/colspan itself, so the
# column of a cell is just its index in the row.
module TableRowspan
  TABLE = %r{<table\b[^>]*>[\s\S]*?</table>}
  ROW = %r{(<tr\b[^>]*>[\s\S]*?</tr>)}
  CELL = %r{(<td\b[^>]*>[\s\S]*?</td>)}
  OPEN = %r{\A(<td\b[^>]*)(>)([\s\S]*)(</td>)\z}
  MARK = %r{\A\s*\^\^\s*\z}

  Cell = Struct.new(:open, :body, :close, :span, :drop)

  def self.process(html)
    return html unless html.include?('^^')
    html.gsub(TABLE) { |table| merge(table) }
  end

  def self.merge(table)
    return table unless table.include?('^^')
    owners = {}
    parts = table.split(ROW).map do |part|
      next part unless part.start_with?('<tr')
      pieces = part.split(CELL)
      col = 0
      cells = pieces.map do |p|
        next p unless (m = p.match(OPEN))
        cell = Cell.new(m[1], m[3], m[4], 1, false)
        if cell.body =~ MARK && owners[col]
          owners[col].span += 1
          cell.drop = true
        else
          owners[col] = cell
        end
        col += 1
        cell
      end
      cells
    end
    parts.map { |part| part.is_a?(Array) ? render_row(part) : part }.join
  end

  def self.render_row(pieces)
    cells = pieces.grep(Cell)
    return '' if !cells.empty? && cells.all?(&:drop)
    pieces.map do |p|
      next p unless p.is_a?(Cell)
      next '' if p.drop
      attr = p.span > 1 ? %( rowspan="#{p.span}") : ''
      "#{p.open}#{attr}>#{p.body}#{p.close}"
    end.join
  end
end

Jekyll::Hooks.register :documents, :post_render do |doc|
  next unless doc.output_ext == '.html' && doc.collection.label == 'posts'
  doc.output = TableRowspan.process(doc.output)
  doc.content = TableRowspan.process(doc.content)
end
