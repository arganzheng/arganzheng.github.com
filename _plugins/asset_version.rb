require 'digest'

module Jekyll
  module AssetVersionFilter
    def asset_v(path)
      site = @context.registers[:site]
      site_root = File.expand_path(site.source)
      path = path.to_s
      relative_path = path.sub(%r{\A/+}, '')
      asset_path = File.expand_path(relative_path, site_root)

      unless path.start_with?('/') &&
             asset_path.start_with?("#{site_root}#{File::SEPARATOR}") &&
             File.file?(asset_path)
        raise Jekyll::Errors::FatalException,
              "asset_v: missing site asset #{path.inspect} (looked for #{asset_path})"
      end

      cache = site.instance_variable_get(:@asset_version_cache)
      unless cache
        cache = {}
        site.instance_variable_set(:@asset_version_cache, cache)
      end
      cache[path] ||= Digest::MD5.file(asset_path).hexdigest[0, 10]
    end
  end
end

Liquid::Template.register_filter(Jekyll::AssetVersionFilter)

Jekyll::Hooks.register :site, :pre_render do |site|
  site.instance_variable_set(:@asset_version_cache, {})
end
