#!/usr/bin/env bash
# Run every tools/*.test.rb (Minitest suites for the _plugins/ text transforms).
# Used by tools/check.sh (= npm run check, the pre-push hook) and by
# .github/workflows/check.yml, so a plugin change cannot ship with its tests red.
#
#   tools/run-tests.sh                 # all suites
#   tools/run-tests.sh tools/x.test.rb # one suite
#
# The suites `require 'jekyll'`, which lives in the bundle, plus `minitest`
# and (two of them) `benchmark` — bundled gems of Ruby itself that are not in
# the Gemfile, so `bundle exec` alone cannot see them. Their lib directories
# are added with -I; add a name to EXTRA when a new suite needs another one.
set -uo pipefail
cd "$(dirname "$0")/.."

EXTRA="minitest benchmark"
includes=()
for gem in $EXTRA; do
  lib=$(ruby -e 'print Gem::Specification.find_by_name(ARGV[0]).full_require_paths.first' "$gem" 2>/dev/null) || {
    echo "gem '$gem' not found (it ships with Ruby: gem install $gem)"; exit 1; }
  includes+=(-I "$lib")
done

files=("$@")
[ ${#files[@]} -eq 0 ] && files=(tools/*.test.rb)

failed=0
for t in "${files[@]}"; do
  if out=$(bundle exec ruby "${includes[@]}" "$t" 2>&1); then
    printf '  %-30s %s\n' "$(basename "$t")" "$(printf '%s\n' "$out" | tail -1)"
  else
    printf '%s\n' "$out" | tail -25
    printf '  %-30s FAILED\n' "$(basename "$t")"
    failed=1
  fi
done
exit $failed
