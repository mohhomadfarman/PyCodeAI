#!/bin/bash

# List of URLs to crawl
URLS=(

# -------------------------
# React Documentation
# -------------------------
"https://react.dev/learn"
"https://react.dev/learn/describing-the-ui"
"https://react.dev/learn/adding-interactivity"
"https://react.dev/learn/managing-state"
"https://react.dev/learn/escape-hatches"
"https://react.dev/learn/thinking-in-react"
"https://react.dev/learn/sharing-state-between-components"
"https://react.dev/learn/passing-props-to-a-component"
"https://react.dev/learn/referencing-values-with-refs"
"https://react.dev/learn/lifecycle-of-reactive-effects"
"https://react.dev/reference/react"
"https://react.dev/reference/react/useState"
"https://react.dev/reference/react/useEffect"
"https://react.dev/reference/react/useContext"
"https://react.dev/reference/react/useReducer"
"https://react.dev/reference/react/useRef"
"https://react.dev/reference/react/useMemo"
"https://react.dev/reference/react/useCallback"

# -------------------------
# MDN JavaScript Guide
# -------------------------
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Introduction"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Grammar_and_types"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Control_flow_and_error_handling"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Loops_and_iteration"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Functions"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Expressions_and_operators"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Numbers_and_dates"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Text_formatting"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Indexed_collections"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Keyed_collections"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Working_with_objects"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_classes"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Iterators_and_generators"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Promises"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Modules"
"https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference"
"https://developer.mozilla.org/en-US/docs/Web/API"

# -------------------------
# TypeScript
# -------------------------
"https://www.typescriptlang.org/docs/"
"https://www.typescriptlang.org/docs/handbook/intro.html"
"https://www.typescriptlang.org/docs/handbook/basic-types.html"
"https://www.typescriptlang.org/docs/handbook/interfaces.html"
"https://www.typescriptlang.org/docs/handbook/functions.html"
"https://www.typescriptlang.org/docs/handbook/generics.html"
"https://www.typescriptlang.org/docs/handbook/advanced-types.html"
"https://www.typescriptlang.org/docs/handbook/utility-types.html"
"https://www.typescriptlang.org/docs/handbook/decorators.html"

# -------------------------
# Next.js
# -------------------------
"https://nextjs.org/docs"
"https://nextjs.org/docs/app"
"https://nextjs.org/docs/pages"
"https://nextjs.org/docs/app/building-your-application/routing"
"https://nextjs.org/docs/app/building-your-application/data-fetching"
"https://nextjs.org/docs/api-reference"
"https://nextjs.org/docs/authentication"
"https://nextjs.org/docs/deployment"

# -------------------------
# Node.js
# -------------------------
"https://nodejs.org/en/docs"
"https://nodejs.org/api/"
"https://nodejs.org/api/fs.html"
"https://nodejs.org/api/http.html"
"https://nodejs.org/api/path.html"
"https://nodejs.org/api/stream.html"
"https://nodejs.org/api/events.html"
"https://nodejs.org/api/crypto.html"
"https://nodejs.org/api/process.html"

# -------------------------
# Express
# -------------------------
"https://expressjs.com/"
"https://expressjs.com/en/guide/routing.html"
"https://expressjs.com/en/guide/middleware.html"
"https://expressjs.com/en/guide/error-handling.html"
"https://expressjs.com/en/guide/using-middleware.html"
"https://expressjs.com/en/api.html"

# -------------------------
# PostgreSQL
# -------------------------
"https://www.postgresql.org/docs/"
"https://www.postgresql.org/docs/current/tutorial.html"
"https://www.postgresql.org/docs/current/sql-select.html"
"https://www.postgresql.org/docs/current/sql-insert.html"
"https://www.postgresql.org/docs/current/sql-update.html"
"https://www.postgresql.org/docs/current/sql-delete.html"
"https://www.postgresql.org/docs/current/functions.html"
"https://www.postgresql.org/docs/current/indexes.html"

# -------------------------
# Docker
# -------------------------
"https://docs.docker.com/"
"https://docs.docker.com/get-started/"
"https://docs.docker.com/engine/"
"https://docs.docker.com/compose/"
"https://docs.docker.com/network/"
"https://docs.docker.com/storage/"

# -------------------------
# Kubernetes
# -------------------------
"https://kubernetes.io/docs/home/"
"https://kubernetes.io/docs/concepts/"
"https://kubernetes.io/docs/tasks/"
"https://kubernetes.io/docs/reference/"
"https://kubernetes.io/docs/setup/"

# -------------------------
# Testing
# -------------------------
"https://jestjs.io/docs/getting-started"
"https://jestjs.io/docs/mock-functions"
"https://testing-library.com/docs/react-testing-library/intro/"
"https://playwright.dev/docs/intro"
"https://playwright.dev/docs/test-assertions"
"https://www.cypress.io/documentation"

# -------------------------
# Git
# -------------------------
"https://git-scm.com/docs"
"https://git-scm.com/docs/git-clone"
"https://git-scm.com/docs/git-commit"
"https://git-scm.com/docs/git-branch"
"https://git-scm.com/docs/git-rebase"
"https://git-scm.com/docs/git-merge"

# -------------------------
# GitHub
# -------------------------
"https://docs.github.com/"
"https://docs.github.com/en/actions"
"https://docs.github.com/en/repositories"
"https://docs.github.com/en/copilot"
"https://docs.github.com/en/rest"

# -------------------------
# Tailwind CSS
# -------------------------
"https://tailwindcss.com/docs"
"https://tailwindcss.com/docs/installation"
"https://tailwindcss.com/docs/responsive-design"
"https://tailwindcss.com/docs/hover-focus-and-other-states"
"https://tailwindcss.com/docs/theme"

)


echo "🕷️  Starting bulk crawl of ${#URLS[@]} articles..."

# No cd needed if run from project root
for url in "${URLS[@]}"; do
    echo "----------------------------------------"
    python cli.py crawl-web "$url"
    sleep 2
done

echo "----------------------------------------"
echo "✅ Bulk crawl finished!"
