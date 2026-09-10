# Keep the home page (jekyll-paginate) a pure tech flow: jekyll-paginate 1.x
# drops posts with `hidden: true` from the paginator (and only from there —
# site.posts / archive / tags / feed still see them). We set that flag for
# `category: life` (they live on /life/) and for `pinned: true` (rendered
# separately at the top of page 1), so pages never have gaps.
# `:site, :post_read` runs after front matter is loaded and before generators.
Jekyll::Hooks.register :site, :post_read do |site|
  site.posts.docs.each do |post|
    post.data['hidden'] = true if post.data['category'] == 'life' || post.data['pinned']
  end
end
