#!/usr/bin/env bash
# less/argan-blog.less -> css/argan-blog.css + css/argan-blog.min.css
#
#   npm run css            # build
#   npm run css -- --check # exit 1 if the committed CSS is not what the Less produces
#
# Every style lives in less/ (see less/argan-blog.less for the import order);
# the two css files are build output and must never be edited by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
LESSC=node_modules/.bin/lessc
CLEAN=node_modules/.bin/cleancss
[ -x "$LESSC" ] && [ -x "$CLEAN" ] || { echo "run: npm install" >&2; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$LESSC" --no-color less/argan-blog.less "$tmp/argan-blog.css"
"$CLEAN" -O1 --format keep-breaks=false -o "$tmp/argan-blog.min.css" "$tmp/argan-blog.css"

if [ "${1:-}" = "--check" ]; then
  ok=1
  for f in argan-blog.css argan-blog.min.css; do
    cmp -s "$tmp/$f" "css/$f" || { echo "css/$f is stale — run: npm run css"; ok=0; }
  done
  [ $ok = 1 ] && echo "css/ matches less/"
  [ $ok = 1 ]
else
  cp "$tmp/argan-blog.css" "$tmp/argan-blog.min.css" css/
  ls -la css/argan-blog.css css/argan-blog.min.css | awk '{print $5, $9}'
fi
