#!/usr/bin/env bash
# Local mirror of .github/workflows/check.yml — run before pushing.
#
#   tools/check.sh            # everything below
#   SKIP_LINKS=1 tools/check.sh   # skip lychee (the slow part)
#   npm run check             # same thing
#
# Installed as the git pre-push hook by `npm run hooks` (core.hooksPath=.githooks).
#
# 1. Liquid-looking {{ / {% inside code blocks not wrapped in {% raw %}
#    (one such draft aborts the whole build, and _site silently stays stale under jekyll serve)
# 2. css/argan-blog{,.min}.css are exactly what less/ compiles to (npm run css)
# 3. js/blog.min.js is exactly what js/*.js bundle to (npm run js)
# 4. jekyll build --future --unpublished --strict_front_matter -> _site-check (must print "done in";
#    unpublished too, so a hidden post cannot park a Liquid error that bites when it is published)
# 5. Font Awesome subset covers every icon in use
# 6. lychee offline: every internal link / image / #fragment in _site-check resolves
# 7. git diff --check (whitespace errors)
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '\033[32mok\033[0m  %s\n' "$1"; }
bad()  { printf '\033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

step "Liquid in code blocks"
python3 tools/liquid-scan.py && ok "no unescaped {{ / {% in code blocks" || bad "wrap the block in {% raw %} … {% endraw %}"

step "css/ is built from less/"
tools/build-css.sh --check >/dev/null 2>&1 && ok "css/argan-blog{,.min}.css match less/" || bad "css/ is stale or hand-edited — run: npm run css"

step "js/blog.min.js is built from js/*.js"
tools/build-js.sh --check >/dev/null 2>&1 && ok "js/blog.min.js matches its sources" || bad "js/blog.min.js is stale — run: npm run js"

step "jekyll build --future --unpublished --strict_front_matter"
out=$(jekyll build --future --unpublished --strict_front_matter -d _site-check 2>&1)
if printf '%s' "$out" | grep -q 'done in'; then ok "$(printf '%s' "$out" | grep -o 'done in .*')"
else printf '%s\n' "$out" | tail -20; bad "jekyll build did not finish"; fi

step "Font Awesome subset"
python3 tools/fa-subset.py --check >/dev/null 2>&1 && ok "subset covers every icon" || bad "icons missing from subset — run tools/fa-subset.py"

step "Internal links (lychee --offline)"
if [ "${SKIP_LINKS:-}" = 1 ]; then echo "skipped (SKIP_LINKS=1)"
elif ! command -v lychee >/dev/null; then echo "skipped (brew install lychee)"
else
  lout=$(lychee --offline --root-dir "$PWD/_site-check" --include-fragments --exclude '/tags/?#' --exclude-path _site-check/slides --no-progress '_site-check/**/*.html' 2>&1)
  if [ $? -eq 0 ]; then ok "$(printf '%s' "$lout" | grep -E '^[0-9]+ Total' || echo 'all internal links resolve')"
  else printf '%s\n' "$lout" | grep -vE '^\s*$' | tail -30; bad "broken internal links"; fi
fi

step "git diff --check"
git diff --check && git diff --cached --check && ok "no whitespace errors" || bad "whitespace errors"

echo
[ $fail = 0 ] && printf '\033[32mall checks passed\033[0m\n' || printf '\033[31msome checks failed\033[0m\n'
exit $fail
