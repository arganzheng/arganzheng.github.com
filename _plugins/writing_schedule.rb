require 'pathname'
require 'safe_yaml'

module WritingSchedule
  FRONT_MATTER = /\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\z)/m
  RECAP = /-series-recap-and-self-test\.(md|markdown)\z/
  POST_GLOB = '**/*.{md,markdown}'

  def self.front_matter(file)
    source = File.read(file, encoding: 'utf-8')
    match = source.match(FRONT_MATTER)
    return [{}, nil] unless match

    [SafeYAML.load(match[1]) || {}, match[1]]
  rescue Psych::Exception => e
    raise Jekyll::Errors::FatalException, "writing_schedule: invalid front matter in #{file}: #{e.message}"
  end

  def self.date_for(file, data, front_matter)
    if data.key?('date')
      raw_date = front_matter&.match(/^date[ \t]*:[ \t]*(.*?)\s*$/)&.captures&.first
      raw_date = raw_date&.sub(/\s+#.*\z/, '')&.strip
      raw_date = data['date'].to_s if raw_date.nil? || raw_date.start_with?('"', "'", '*', '!', '&')
      Jekyll::Utils.parse_date(raw_date)
    else
      filename_date = File.basename(file)[/\A(\d{4}-\d{2}-\d{2})-/, 1]
      return unless filename_date

      Jekyll::Utils.parse_date(filename_date)
    end
  end

  def self.repository_path(site, collection, file)
    base = Pathname.new(File.join(site.source, collection))
    relative = Pathname.new(file).relative_path_from(base).to_s
    File.join(collection, relative).tr('\\', '/')
  end

  class Generator < Jekyll::Generator
    safe true
    priority :lowest

    def generate(site)
      series = site.data['series'] || {}
      now = site.time
      scheduled = Dir.glob(File.join(site.source, '_posts', POST_GLOB)).filter_map do |file|
        data, front_matter = WritingSchedule.front_matter(file)
        next if data['published'] == false || data['visibility'] == 'private'

        date = WritingSchedule.date_for(file, data, front_matter)
        next unless date && date > now

        {
          'date' => date.strftime('%Y-%m-%d'),
          'title' => data['title'].to_s,
          'path' => WritingSchedule.repository_path(site, '_posts', file),
          'series' => data['series'],
          'recap' => !!File.basename(file).match?(RECAP)
        }
      end
      scheduled.sort_by! { |post| [post['date'], post['path']] }

      drafts = Dir.glob(File.join(site.source, '_drafts', POST_GLOB)).map do |file|
        data, = WritingSchedule.front_matter(file)
        title = data['title'].to_s.strip
        title = File.basename(file, File.extname(file)) if title.empty?
        { 'title' => title, 'path' => WritingSchedule.repository_path(site, '_drafts', file) }
      end.sort_by { |draft| [draft['title'], draft['path']] }

      site.data['writing_schedule'] = {
        'scheduled' => scheduled,
        'drafts' => drafts,
        'published' => series.each_with_object({}) { |(key, meta), counts| counts[key] = meta['count'].to_i },
        'series' => series.each_with_object({}) do |(key, meta), entries|
          entries[key] = { 'name' => meta['name'], 'level' => meta['level'] }
        end
      }
    end
  end
end
