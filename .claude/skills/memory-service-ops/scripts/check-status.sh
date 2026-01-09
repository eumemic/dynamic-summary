#!/bin/bash
# Check memory service status for a Railway environment
# Usage: ./check-status.sh [production|pr-NUMBER]

set -e

PROJECT_ID="9d168ba6-ac78-4739-a53c-7ca04e211678"

# Determine environment
if [[ "$1" == "production" ]]; then
    ENV_NAME="production"
    DB_SERVICE="pgvector"
elif [[ "$1" =~ ^pr-([0-9]+)$ ]]; then
    PR_NUM="${BASH_REMATCH[1]}"
    ENV_NAME="dynamic-summary-pr-${PR_NUM}"
    DB_SERVICE="pgvector-rW-f"
else
    echo "Usage: $0 [production|pr-NUMBER]"
    echo "  production  - Check production environment"
    echo "  pr-NUMBER   - Check PR environment (e.g., pr-335)"
    exit 1
fi

echo "=== Checking $ENV_NAME ==="
echo ""

# Link to environment
echo "Linking to Railway environment..."
railway link -p "$PROJECT_ID" -e "$ENV_NAME" 2>&1 | grep -v "Select"

# Verify database service by checking dynamic-summary variables
echo ""
echo "Verifying database connection..."
INTERNAL_URL=$(railway variables --service dynamic-summary --json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('RAGZOOM_DATABASE_URL',''))")

if [[ "$INTERNAL_URL" == *"pgvector-rw-f"* ]]; then
    ACTUAL_DB_SERVICE="pgvector-rW-f"
elif [[ "$INTERNAL_URL" == *"pgvector"* ]]; then
    ACTUAL_DB_SERVICE="pgvector"
else
    echo "ERROR: Could not determine database service from RAGZOOM_DATABASE_URL"
    exit 1
fi

if [[ "$ACTUAL_DB_SERVICE" != "$DB_SERVICE" ]]; then
    echo "WARNING: Expected $DB_SERVICE but found $ACTUAL_DB_SERVICE"
    DB_SERVICE="$ACTUAL_DB_SERVICE"
fi

echo "Using database service: $DB_SERVICE"

# Get public database URL
echo ""
echo "Getting public database URL..."
DB_URL=$(railway variables --service "$DB_SERVICE" --kv 2>/dev/null | grep DATABASE_PUBLIC_URL | cut -d= -f2-)

if [[ -z "$DB_URL" ]]; then
    echo "ERROR: Could not get DATABASE_PUBLIC_URL from $DB_SERVICE"
    exit 1
fi

# Run status command
echo ""
echo "=== Memory Service Status ==="
echo ""
RAGZOOM_DATABASE_URL="$DB_URL" python -m memory_service.admin status
