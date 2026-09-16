#!/usr/bin/env bash
# The post-page scripts (all IIFEs, order = the old <script> order in footer.html)
# -> one minified js/blog.min.js (+ source map) loaded by _includes/footer.html.
#
#   npm run js            # build
#   npm run js -- --check # exit 1 if js/blog.min.js is not what js/*.js produce
#
# Kept separate on purpose: search.js (head, every page),
# wechat-export.js (lazy, author only), dashboard.js + feedback-brief.js (/admin/).
set -euo pipefail
cd "$(dirname "$0")/.."
UGLIFY=node_modules/.bin/uglifyjs
[ -x "$UGLIFY" ] || { echo "run: npm install" >&2; exit 1; }

SRC=(
  js/argan-blog.js        # theme: responsive tables/embeds, navbar on scroll, side catalog pin
  js/toc.js               # [TOC] + floating side catalog
  js/diagram-zoom.js      # lightbox for Mermaid diagrams / images
  js/code-copy.js         # copy button on code blocks, diagrams, tables
  js/code-tabs.js         # Python / Java tabs on grouped code blocks
  js/figures.js           # captions + feedback handle (before annotations.js: the caption is what gets underlined)
  js/code-tokens.js       # member-access colouring rouge cannot express
  js/inline-popups.js     # footnote / external-link popups
  js/vendor/approx-string-match.js
  js/annotations.js       # comments, 划线评论, reactions, views
  js/share.js             # action bar, share menu, .post-stats strip
)

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$UGLIFY" "${SRC[@]}" -c -m --comments '/^!/' \
  --source-map "url='blog.min.js.map',includeSources" \
  -o "$tmp/blog.min.js"

if [ "${1:-}" = "--check" ]; then
  if cmp -s "$tmp/blog.min.js" js/blog.min.js; then echo "js/blog.min.js matches js/*.js"
  else echo "js/blog.min.js is stale — run: npm run js"; exit 1; fi
else
  cp "$tmp/blog.min.js" "$tmp/blog.min.js.map" js/
  ls -la js/blog.min.js js/blog.min.js.map | awk '{print $5, $9}'
fi
