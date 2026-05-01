<!-- markdownlint-disable-file -->
# M.Tech AI/ML — Study Hub

A free, open study companion for the M.Tech AI &amp; Machine Learning curriculum.
Module-by-module notes, hand-picked free resources, and quick-revise cards.

**Status:** Semester 1 available · Semester 2 planned · Semesters 3–4 not started yet.

## What this is

- A static, **zero-build** website (HTML + CSS + vanilla JS).
- Hosted on GitHub Pages.
- Designed for fast, exam-window revision — by the maintainer, with the cohort
  and any public learner who finds it useful.

## Run it locally

```powershell
python -m http.server 8000
# open http://localhost:8000
```

(Browsers block `fetch()` on `file://` URLs, so opening `index.html` by
double-click leaves notes panels stuck on "Loading…". Use the local server.)

## Project layout

```text
index.html                     overview / home
courses/<slug>.html            one HTML shell per subject
data/subjects/<slug>.js        subject metadata (modules, resources)
notes/<slug>/NN-*.md           per-module Quick-Revise notes
assets/                        styles, JS, favicon, OG card, analytics
docs/                          PRD, smoke checklist, content review tracker
```

## Add a module / fix a typo

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Verify before release

Walk through [`docs/SMOKE-CHECKLIST.md`](docs/SMOKE-CHECKLIST.md).

## Why these decisions look the way they do

The product brief — primary users, success metric, scope edges, and the explicit
list of things this iteration **does not** do — lives in
[`docs/prds/study-hub-mvp-baseline.md`](docs/prds/study-hub-mvp-baseline.md).

## Tech stack

- Markdown rendering: [marked](https://marked.js.org/)
- HTML sanitization: [DOMPurify](https://github.com/cure53/DOMPurify)
- Math: [KaTeX](https://katex.org/)
- Diagrams: [Mermaid](https://mermaid.js.org/)
- Analytics: [GoatCounter](https://www.goatcounter.com/) (cookieless)

All loaded via CDN with pinned versions. No build step. No package manager.
