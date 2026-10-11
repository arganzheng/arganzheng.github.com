Jekyll::Hooks.register :site, :post_read, priority: :high do |site|
  private_posts = site.posts.docs.select { |doc| doc.data['visibility'] == 'private' }
  next if private_posts.empty?

  collection = site.collections['private_posts'] ||= Jekyll::Collection.new(site, 'private_posts')
  private_posts.each do |doc|
    doc.data['sitemap'] = false
    doc.data['noindex'] = true
    site.posts.docs.delete(doc)
    collection.docs << doc
  end
end
