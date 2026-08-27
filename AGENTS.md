# AGENTS.md

## Build / preview

Jekyll 4.4.1 is installed against Homebrew's Ruby, which is **not** on `PATH`
(the system `ruby` is 2.6 and there is no `Gemfile`):

```bash
/opt/homebrew/lib/ruby/gems/*/bin/jekyll build      # -> _site/
/opt/homebrew/lib/ruby/gems/*/bin/jekyll serve      # http://localhost:4000
```

The pre-existing `Conflict: ... the-productive-programmer-on-windows.html`
warning is harmless.

`Gruntfile.js` compiles `less/ -> css/argan-blog{,.min}.css` and minifies
`js/hux-blog.js`. Edit the `.less`/`.js` sources, not the generated CSS:

```bash
npm install && npx grunt        # or `npx grunt watch`
```

**WARNING — CSS drift:** `npx grunt` currently fails (grunt-cli is not
installed and the Gruntfile expects a non-existent `less/argan-blog.less`;
the real entry point is `less/blog.less`). More importantly,
`css/argan-blog{,.min}.css` contain ~485 lines of hand-appended rules
(from `.page-header .title` onwards) that are NOT in the `.less` sources,
so a full recompile destroys them. To change styles, edit the `.less`
source for reference, then append the corresponding compiled CSS to the
END of both `css/argan-blog.css` and `css/argan-blog.min.css` by hand
(compile a fragment with `node_modules/.bin/lessc` if helpful).

## Layout

- `_posts/` — blog posts, `layout: post`, permalink `/:title.html`
- `slides/` — reveal.js decks, `layout: slides` (or set in front matter),
  URL `/slides/:name.html`, indexed by `slides.html` (`/slides/`)
- `_includes/rich-content.html` — Mermaid + KaTeX loaders, shared by
  `_includes/head.html` and `_layouts/slides.html`. Both renderers are lazy:
  they only fetch their bundle if the page actually contains a diagram/formula,
  and they only look inside `.post-container` and `.reveal .slides`.
- `_includes/analytics.html` — GA + Baidu Tongji, shared by `footer.html` and
  `_layouts/slides.html`

## Writing a deck

Decks are ordinary Markdown; kramdown renders the file and `_layouts/slides.html`
splits the HTML on every `<hr>` into reveal.js `<section>`s.

- Create `slides/my-talk.md` with `layout: slides` and `permalink: /slides/my-talk.html` in the front matter.
- Separate slides with `---` and **always leave a blank line before it**,
  otherwise Markdown reads it as a setext `<h2>` underline and the slide is
  not split.
- `<!-- v -->` inside a slide creates vertical (nested) sub-slides.
- `<!-- .slide: data-background="#1c1f26" -->` puts reveal.js attributes on the
  current `<section>`.
- `{: .fragments}` on a list reveals it one item at a time; `{: .fragment}`
  reveals any single element.
- Speaker notes: `<aside class="notes" markdown="1">...</aside>`, shown with `S`.
- Export: append `?print-pdf` and print from the browser.

`slides/reveal-demo.md` is a live demo of all of the above.
