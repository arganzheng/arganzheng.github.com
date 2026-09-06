# AGENTS.md

## Build / preview

Jekyll 4.4.1 is installed against Homebrew's Ruby.
Homebrew Ruby and Gem paths are configured in `~/.zshrc`:
`/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin`

A `Gemfile` is also present at repo root:

```bash
jekyll build            # or bundle exec jekyll build -> _site/
jekyll serve            # or bundle exec jekyll serve -> http://localhost:4000
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

## Writing AI-Infra series posts

- **Version/date rule:** a post may only cite software versions (and facts about
  them) released *before* the post's front-matter date. Check tag dates with
  `git log -1 --format=%cs <tag>` before pinning a version; pin an explicit tag
  (e.g. "NCCL 2.28.9", "PyTorch 2.12", "vLLM v0.23.0"), never "近期主线" or a
  local checkout. Pinned checkouts used so far live next to this repo:
  `../nccl` (v2.28.9-1), `../nccl-tests` (v2.18.3), `../pytorch-v2.12.0`,
  `../vllm-v0.23.0` (git worktrees of `../pytorch` / `../vllm`).
- Cite source as path + function/class name, never line numbers.
- Length is not a target; rigor and organisation are. Structure: nav quote →
  intro with the post's core question → `## 一、总览` (ending with 本文的章节安排)
  → body (`##` Chinese numerals, `###` Arabic) → `## N、本文小结` → `## 下一篇`.
- Series are independent: no links to posts of other series.
- `{%`/`{{` inside code (PTX asm, printf formats, regexes) must be wrapped in
  `{% raw %}` … `{% endraw %}` or the Liquid pass fails the build.
