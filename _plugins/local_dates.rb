# A bare front-matter timestamp (`date: 2026-10-04 20:00:00`) is UTC by the
# YAML spec, so Psych hands Jekyll a UTC Time and the post renders as 04:00
# the next day in Asia/Shanghai — while a filename date (no `date:` key) is
# midnight *local*. Same-day posts therefore sorted wrong (every series recap
# landed inside the following series, 2026-09-16). Reinterpret such values as
# wall-clock time in site.timezone (ENV['TZ'] is already set to it here), so
# `date:` / `updated:` read like the rest of the site. A value with an
# non-zero explicit offset (`+0800`, `-05:00`) is left alone; a value written
# as UTC on purpose (`Z`, `+00:00`) is indistinguishable from a bare one and
# would be shifted too — nothing on this site needs that.
module Jekyll
  module LocalDates
    KEYS = %w[date updated].freeze

    def self.localize(value)
      return value unless value.is_a?(Time) && value.utc_offset.zero?
      Time.new(value.year, value.month, value.day, value.hour, value.min, value.sec)
    end

    def merge_data!(other, source: "YAML front matter")
      KEYS.each { |k| other[k] = LocalDates.localize(other[k]) if other.key?(k) } if other.is_a?(Hash)
      super
    end
  end

  Document.prepend(LocalDates)
end
