#!/usr/bin/env bash
# Writes overview_page.html, the long-page layout of overview.html (the slide deck). Edit overview.html only, then run this.
set -eu
cd "$(dirname "$0")"
sed -e 's|<meta name="layout" content="slides">|<meta name="layout" content="page">|' \
    -e 's|<title>Thor 共置研究导览</title>|<title>Thor 共置研究文档</title>|' overview.html > overview_page.html
grep -q 'content="page"' overview_page.html
