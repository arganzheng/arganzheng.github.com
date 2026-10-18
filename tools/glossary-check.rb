#!/usr/bin/env ruby
# tools/glossary-check.rb — validates _data/glossary/*.yml.
#
# Fails on:
#   - a duplicate term within a file
#   - a term or alias that appears in none of its series' posts (same masking
#     as _plugins/glossary.rb: front matter, code fences, inline code, $$…$$,
#     and link URLs stripped; same [A-Za-z0-9_-] boundary on both sides)
#   - a missing full or desc
#
# Warns only when the same term has a different `full` across series files.
require 'yaml'

ROOT = File.expand_path('..', __dir__)
BOUND = '[A-Za-z0-9_-]'

def mask_markdown(text)
  s = text.to_s.gsub(/\A---\s*\n.*?\n---\s*\n/m, '')
  s.gsub!(/```.*?```/m) { |m| ' ' * m.length }
  s.gsub!(/`[^`\n]*`/) { |m| ' ' * m.length }
  s.gsub!(/\$\$.*?\$\$/m) { |m| ' ' * m.length }
  s.gsub!(/\]\(([^)\s]*)[^)]*\)/) { |m| ']' + ' ' * (m.length - 1) }
  s
end

def term_re(term)
  /(?<![#{BOUND}])#{Regexp.escape(term)}(?![#{BOUND}])/
end

# series key => [masked bodies in date order]
bodies = Hash.new { |h, k| h[k] = [] }
Dir.glob(File.join(ROOT, '_posts', '*.{md,markdown}')).sort.each do |file|
  src = File.read(file, encoding: 'utf-8')
  next unless src =~ /\A---\s*\n(.*?)\n---\s*\n/m
  fm = Regexp.last_match(1)
  next unless (key = fm[/^series:\s*['"]?([\w-]+)/, 1])
  bodies[key] << mask_markdown(src)
end

failures = []
warnings = []
fulls = Hash.new { |h, k| h[k] = {} }

Dir.glob(File.join(ROOT, '_data', 'glossary', '*.yml')).sort.each do |file|
  key = File.basename(file, '.yml')
  entries = YAML.load_file(file) || []
  seen = {}
  entries.each_with_index do |e, i|
    term = e['term'].to_s
    failures << "#{key}.yml entry #{i}: missing term" if term.empty?
    failures << "#{key}.yml: duplicate term '#{term}'" if seen[term]
    seen[term] = true
    failures << "#{key}.yml '#{term}': missing full" if e['full'].to_s.strip.empty?
    failures << "#{key}.yml '#{term}': missing desc" if e['desc'].to_s.strip.empty?

    patterns = ([term] + Array(e['aliases'])).map { |t| term_re(t) }
    found = bodies[key].any? { |body| patterns.any? { |re| body =~ re } }
    failures << "#{key}.yml '#{term}': term and aliases appear in no #{key} post" unless found

    prev = fulls[term]
    full = e['full'].to_s
    warnings << "'#{term}' has full '#{prev.keys.first}' in #{prev.values.first} but '#{full}' in #{key}" if !prev.empty? && !prev.key?(full)
    prev[full] ||= key
  end
end

warnings.each { |w| puts "warn: #{w}" }
if failures.empty?
  puts 'glossary-check: ok'
  exit 0
else
  failures.each { |f| puts "FAIL: #{f}" }
  exit 1
end
