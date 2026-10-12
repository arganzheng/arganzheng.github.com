module TagTree
  def self.configuration(site)
    site.data['tag_tree'] || {}
  end

  def self.aliases(config)
    (config['aliases'] || {}).each_with_object({}) do |(alias_name, canonical), out|
      out[alias_name.to_s] = canonical.to_s
    end
  end

  def self.tree_names(config)
    names = Hash.new { |hash, key| hash[key] = [] }
    walk = lambda do |nodes, parent|
      Array(nodes).each do |node|
        unless node.is_a?(Hash) && node['name'].is_a?(String) && !node['name'].empty?
          raise Jekyll::Errors::FatalException, "tag_tree.yml: each tree node needs a non-empty name (under #{parent})"
        end

        name = node['name']
        names[name] << "node #{name}"
        Array(node['tags']).each do |tag|
          names[tag.to_s] << "tag #{tag} under #{name}"
        end
        walk.call(node['children'], name)
      end
    end
    walk.call(config['tree'], 'root')
    names
  end

  def self.pairs(config, key)
    Array(config[key]).map do |pair|
      unless pair.is_a?(Array) && pair.length == 2
        raise Jekyll::Errors::FatalException, "tag_tree.yml: each #{key} entry must contain exactly two tag names"
      end
      pair.map(&:to_s)
    end
  end

  def self.validate!(config)
    alias_map = aliases(config)
    name_occurrences = tree_names(config)
    duplicate = name_occurrences.find { |_name, entries| entries.length > 1 }
    if duplicate
      raise Jekyll::Errors::FatalException,
            "tag_tree.yml: #{duplicate[0].inspect} appears more than once in tree (#{duplicate[1].join(', ')})"
    end

    alias_keys_in_tree = alias_map.keys & name_occurrences.keys
    unless alias_keys_in_tree.empty?
      raise Jekyll::Errors::FatalException,
            "tag_tree.yml: alias key(s) appear in tree: #{alias_keys_in_tree.join(', ')}"
    end

    chained_aliases = alias_map.select { |_name, target| alias_map.key?(target) }
    unless chained_aliases.empty?
      pairs = chained_aliases.map { |name, target| "#{name} → #{target}" }
      raise Jekyll::Errors::FatalException,
            "tag_tree.yml: alias target cannot itself be an alias: #{pairs.join(', ')}"
    end

    %w[similar distinct].each do |key|
      conflict = pairs(config, key).flatten & alias_map.keys
      next if conflict.empty?

      raise Jekyll::Errors::FatalException,
            "tag_tree.yml: alias key(s) appear in #{key}: #{conflict.uniq.join(', ')}"
    end
    true
  end

  def self.documents(site)
    docs = site.collections.values.flat_map(&:docs)
    docs.concat(site.posts.docs) if site.posts
    docs.concat(site.pages)
    docs.uniq(&:object_id)
  end

  def self.normalize_documents(site, alias_map)
    documents(site).each do |doc|
      next unless doc.data['tags']

      doc.data['tags'] = Array(doc.data['tags']).map do |tag|
        alias_map.fetch(tag.to_s, tag.to_s)
      end.uniq
    end
  end

  def self.levenshtein_one?(left, right)
    a = left.each_char.to_a
    b = right.each_char.to_a
    return false if a == b || (a.length - b.length).abs > 1

    i = 0
    j = 0
    edits = 0
    while i < a.length && j < b.length
      if a[i] == b[j]
        i += 1
        j += 1
      else
        edits += 1
        return false if edits > 1

        if a.length == b.length
          i += 1
          j += 1
        elsif a.length > b.length
          i += 1
        else
          j += 1
        end
      end
    end
    edits += 1 if i < a.length || j < b.length
    edits == 1
  end

  def self.normalized_key(name)
    name.downcase.gsub(/[\s._-]/, '').sub(/(?:ing|s)\z/, '')
  end

  def self.pair_key(a, b)
    [a, b].sort
  end

  def self.auto_similar(tags)
    tags.combination(2).filter_map do |a, b|
      same_key = normalized_key(a) == normalized_key(b)
      edit_distance_one = a.each_char.count >= 6 && b.each_char.count >= 6 && levenshtein_one?(a, b)
      pair_key(a, b) if same_key || edit_distance_one
    end.uniq.sort
  end

  class Generator < Jekyll::Generator
    safe true
    priority :lowest

    def generate(site)
      config = TagTree.configuration(site)
      TagTree.validate!(config)
      alias_map = TagTree.aliases(config)
      posts = site.posts.docs.select do |post|
        post.data['published'] != false &&
          post.data['visibility'] != 'private' &&
          post.date &&
          !Array(post.data['tags']).empty?
      end
      posts.sort_by! { |post| [-post.date.to_f, post.url.to_s] }

      post_tags = posts.map do |post|
        Array(post.data['tags']).map(&:to_s).uniq
      end
      counts = Hash.new(0)
      post_tags.each { |tags| tags.each { |tag| counts[tag] += 1 } }
      tag_names = counts.keys.sort
      tag_ids = tag_names.each_with_index.to_h

      covered = []
      tree = Array(config['tree']).filter_map do |node|
        build_node(node, '', tag_ids, covered)
      end
      unplaced = tag_names.reject { |tag| covered.include?(tag) }
      unless unplaced.empty?
        if tree.any? { |node| node['n'] == '其他' }
          raise Jekyll::Errors::FatalException, 'tag_tree.yml: 其他 is reserved for tags not placed in the configured tree'
        end
        leaves = unplaced.sort_by { |tag| [-counts[tag], tag] }.map { |tag| tag_ids.fetch(tag) }
        tree << { 'n' => '其他', 'p' => '其他', 'c' => [], 'l' => leaves }
        covered.concat(unplaced)
      end
      assert_tree_coverage!(tag_names, covered)

      urls = posts.map { |post| File.join(site.baseurl.to_s, post.url) }
      post_rows = posts.each_with_index.map do |post, index|
        ids = post_tags[index].map { |tag| tag_ids.fetch(tag) }.sort
        [urls[index], post.data['title'].to_s, post.date.strftime('%Y-%m-%d'), ids]
      end
      site.data['tag_browser'] = {
        'tags' => tag_names,
        'counts' => tag_names.map { |tag| counts[tag] },
        'posts' => post_rows,
        'tree' => tree,
        'aliases' => alias_map
      }
      site.data['tag_audit'] = audit_data(config, tag_names, counts, posts, post_tags, urls, tree, unplaced)

      auto = TagTree.auto_similar(tag_names)
      Jekyll.logger.info('TagTree:', "auto similar candidates: #{auto.map { |pair| pair.join(' ↔ ') }.join('; ')}")
    end

    private

    def build_node(node, parent_path, tag_ids, covered)
      name = node['name'].to_s
      path = parent_path.empty? ? name : "#{parent_path}/#{name}"
      own_id = tag_ids[name]
      leaf_ids = Array(node['tags']).filter_map do |tag|
        id = tag_ids[tag.to_s]
        covered << tag.to_s if id
        id
      end
      covered << name if own_id
      children = Array(node['children']).filter_map do |child|
        build_node(child, path, tag_ids, covered)
      end
      subtree_ids = leaf_ids.dup
      subtree_ids << own_id unless own_id.nil?
      children.each { |child| subtree_ids.concat(node_tag_ids(child)) }
      return if subtree_ids.empty?

      row = { 'n' => name, 'p' => path, 'c' => children, 'l' => leaf_ids }
      row['t'] = own_id unless own_id.nil?
      row
    end

    def node_tag_ids(node)
      ids = node['l'].dup
      ids << node['t'] unless node['t'].nil?
      node['c'].each { |child| ids.concat(node_tag_ids(child)) }
      ids
    end

    def assert_tree_coverage!(tag_names, covered)
      counts = covered.tally
      duplicates = counts.select { |_tag, count| count != 1 }
      missing = tag_names - counts.keys
      extra = counts.keys - tag_names
      return if duplicates.empty? && missing.empty? && extra.empty?

      raise Jekyll::Errors::FatalException,
            "tag_tree.yml: output tree coverage invalid (missing: #{missing.join(', ')}; duplicated: #{duplicates.keys.join(', ')}; unknown: #{extra.join(', ')})"
    end

    def audit_data(config, tag_names, counts, posts, post_tags, urls, tree, unplaced)
      locations = {}
      register_locations = lambda do |nodes|
        nodes.each do |node|
          locations[tag_names[node['t']]] = node['p'] unless node['t'].nil?
          node['l'].each { |id| locations[tag_names[id]] = node['p'] }
          register_locations.call(node['c'])
        end
      end
      register_locations.call(tree)
      unplaced.each { |tag| locations[tag] = '其他' }

      single = tag_names.filter_map do |tag|
        next unless counts[tag] == 1

        post_id = post_tags.index { |tags| tags.include?(tag) }
        post = posts[post_id]
        {
          'tag' => tag,
          'group_path' => locations[tag],
          'title' => post.data['title'].to_s,
          'url' => urls[post_id]
        }
      end

      curated = TagTree.pairs(config, 'similar').filter_map do |a, b|
        next unless counts.key?(a) && counts.key?(b)

        TagTree.pair_key(a, b)
      end
      distinct = TagTree.pairs(config, 'distinct').map { |a, b| TagTree.pair_key(a, b) }
      aliases = TagTree.aliases(config).map { |a, b| TagTree.pair_key(a, b) }
      candidates = (curated + TagTree.auto_similar(tag_names)).uniq - distinct - aliases
      similar = candidates.sort.map do |a, b|
        { 'a' => a, 'a_count' => counts[a].to_i, 'b' => b, 'b_count' => counts[b].to_i }
      end

      {
        'unplaced' => unplaced.sort_by { |tag| [-counts[tag], tag] }.map { |tag| { 'tag' => tag, 'count' => counts[tag] } },
        'single' => single,
        'similar' => similar
      }
    end
  end
end

Jekyll::Hooks.register :site, :post_read, priority: :high do |site|
  config = TagTree.configuration(site)
  TagTree.validate!(config)
  TagTree.normalize_documents(site, TagTree.aliases(config))
end
