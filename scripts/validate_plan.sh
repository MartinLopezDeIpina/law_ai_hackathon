#!/usr/bin/env bash
# Validate that every file path referenced in tasks/plan.md either exists or is marked NEW/DELETE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAN="$ROOT/tasks/plan.md"
ERRORS=0

while IFS= read -r line; do
    # Match table rows like: | path/to/file.py | STATUS |
    if [[ "$line" =~ \|[[:space:]]*(\`?[a-zA-Z0-9_./-]+\`?)[[:space:]]*\|[[:space:]]*(NEW|MODIFY|DELETE)[[:space:]]*\| ]]; then
        raw="${BASH_REMATCH[1]}"
        status="${BASH_REMATCH[2]}"
        path="${raw//\`/}"
        full="$ROOT/$path"
        if [[ "$status" == "NEW" ]]; then
            echo "  NEW    $path (will be created)"
        elif [[ "$status" == "DELETE" ]]; then
            if [[ ! -f "$full" ]]; then
                echo "  WARN   DELETE target missing (already gone?): $path"
            else
                echo "  DELETE $path"
            fi
        elif [[ "$status" == "MODIFY" ]]; then
            if [[ ! -f "$full" ]]; then
                echo "  ERROR  MODIFY target does not exist: $path"
                ERRORS=$((ERRORS + 1))
            else
                echo "  OK     $path"
            fi
        fi
    fi
done < "$PLAN"

if [[ $ERRORS -gt 0 ]]; then
    echo "validate_plan: $ERRORS error(s) found" >&2
    exit 1
fi
echo "validate_plan: OK"
