require 'date'
require 'pathname'
require 'time'
require 'yaml'

module WritingSchedule
  FRONT_MATTER = /\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\z)/m
  RECAP = /-series-recap-and-self-test\.(md|markdown)\z/
  POST_GLOB = '**/*.{md,markdown}'

  def self.front_matter(file)
    source = File.read(file, encoding: 'utf-8')
    match = source.match(FRONT_MATTER)
    return {} unless match

    YAML.safe_load(match[1], permitted_classes: [Date, DateTime, Time], aliases: true) || {}
  rescue Psych::Exception => e
    raise Jekyll::Errors::FatalException, "writing_schedule: invalid front matter in #{file}: #{e.message}"
  end

  def self.local_time(time, offset)
    if time.utc_offset.zero?
      Time.new(time.year, time.month, time.day, time.hour, time.min, time.sec, offset)
    else
      time.getlocal(offset)
    end
  end

  def self.date_for(file, data, offset)
    value = data['date']
    if value.nil?
      filename_date = File.basename(file)[/\A(\d{4}-\d{2}-\d{2})-/, 1]
      return unless filename_date

      value = Date.iso8601(filename_date)
    end

    case value
    when DateTime
      local_time(value.to_time, offset)
    when Date
      Time.new(value.year, value.month, value.day, 0, 0, 0, offset)
    when Time
      local_time(value, offset)
    else
      text = value.to_s
      if text.match?(/\A\d{4}-\d{2}-\d{2}\z/)
        date = Date.iso8601(text)
        Time.new(date.year, date.month, date.day, 0, 0, 0, offset)
      else
        local_time(Time.parse(text), offset)
      end
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
      offset = site.time.utc_offset
      now = site.time.getlocal(offset)
      scheduled = Dir.glob(File.join(site.source, '_posts', POST_GLOB)).filter_map do |file|
        data = WritingSchedule.front_matter(file)
        next if data['published'] == false || data['visibility'] == 'private'

        date = WritingSchedule.date_for(file, data, offset)
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
        data = WritingSchedule.front_matter(file)
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
