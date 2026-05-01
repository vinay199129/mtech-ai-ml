<!-- markdownlint-disable-file -->
# Contributing to the M.Tech AI/ML Study Hub

Thanks for considering a contribution. This is a small, opinionated project
maintained by a single author. The bar is not "polished open-source library" —
it is "trustworthy study companion that loads in any browser without a build
step." Contributions that fit that bar are welcome.

## TL;DR

- **Found an error in the notes?** Open an issue with the file path and a one-line description, or send a PR.
- **Want to add a resource to a module?** Edit `data/subjects/<subject>.js` and add an entry to that module's `resources` array.
- **Want to write a Quick-Revise notes file for a module that doesn't have one?** Add a Markdown file at `notes/<subject>/NN-<slug>.md` and reference it from the module's `notesFile`.

No build step. No package install. Just edit and reload `index.html` (served via `python -m http.server`).

## Run the site locally

```powershell
# from the project root
python -m http.server 8000
# then open http://localhost:8000
```

Browsers block `fetch()` against `file://` URLs, so opening `index.html` by
double-click will leave notes panels stuck on "Loading…". Always serve through
the local HTTP server.

## Project layout

```text
index.html                     entry / overview page
courses/<slug>.html            one HTML shell per subject
data/subjects/<slug>.js        subject metadata (modules, resources, refs)
notes/<slug>/NN-<topic>.md     per-module Quick-Revise notes (Markdown)
assets/styles.css              all styles
assets/index-page.js           overview-page rendering
assets/course-page.js          subject-page rendering (a11y + sanitization)
assets/goatcounter.js          cookieless analytics bootstrap (FR-009)
docs/                          PRD, smoke checklist, Lighthouse baselines
```

## How to add a new module to an existing subject

1. Open `data/subjects/<subject>.js`.
2. Add a new entry to that subject's `modules` array:

   ```js
   {
     n: 14,                          // module number (next in sequence)
     title: 'Backpropagation through time',
     ref: 'Goodfellow §10.2',        // optional textbook reference
     subtopics: [
       'Unrolling the recurrence',
       'Vanishing / exploding gradients',
       'Truncated BPTT'
     ],
     notesFile: '14-bptt.md',        // optional; relative to notes/<subject>/
     resources: [
       { type: 'video',   title: '...', desc: '...', url: 'https://...' },
       { type: 'book',    title: '...', desc: '...', url: 'https://...' },
       { type: 'article', title: '...', desc: '...', url: 'https://...' }
     ]
   }
   ```

3. (Optional) Create `notes/<subject>/14-bptt.md` with a Quick-Revise card.
4. Reload the page. No build step.

### Resource `type` values

Allowed values (each gets a colored badge): `video`, `book`, `article`, `docs`, `course`. Anything else renders as an unstyled badge.

## Notes file conventions

- Markdown is rendered with [`marked`](https://marked.js.org/) and sanitized with [DOMPurify](https://github.com/cure53/DOMPurify) before insertion. **Do not** rely on raw `<script>`, `<iframe>`, or `on*=` attributes — they will be stripped.
- Math is rendered with [KaTeX](https://katex.org/). Use `$...$` for inline and `$$...$$` for display math. Backslash sequences inside math are protected from Markdown processing.
- Diagrams use [Mermaid](https://mermaid.js.org/). Wrap in a fenced block:

  ````markdown
  ```mermaid
  graph LR
    A --> B
  ```
  ````

- Callouts use the existing CSS classes (`intuition`, `tip`, `warn`, `example`, `defn`):

  ```html
  <div class="callout intuition">
    <span class="callout-title">Intuition</span>
    <p>Why this works in plain words.</p>
  </div>
  ```

- Symbol definition lists:

  ```html
  <dl class="symbols">
    <dt>x</dt><dd>input vector</dd>
    <dt>w</dt><dd>weight vector</dd>
  </dl>
  ```

## What NOT to put in notes or data files

- No raw `<script>` tags.
- No inline event handlers (`onclick=`, `onerror=`, etc.).
- No `<iframe>`, `<embed>`, or `<object>` tags.
- No tracking pixels or third-party analytics snippets — analytics goes through `assets/goatcounter.js` only.
- No paywalled or affiliate links in `resources`. Free, freely-accessible material only.

## Issue / PR conventions

- **Issues:** include the file path (e.g. `notes/machine-learning/07-svm.md`) and a one-line description. Screenshots welcome.
- **PRs:**
  - Keep them small. One module per PR is ideal.
  - Run through the relevant items in [`docs/SMOKE-CHECKLIST.md`](SMOKE-CHECKLIST.md) before opening the PR.
  - Do not bundle unrelated changes (e.g. don't fix typos and add a module in the same PR).

## Style

- 2-space indentation in HTML / CSS / JS.
- Keep CSS class names existing — `course-page.js` and `index-page.js` query by class.
- Prefer plain DOM construction (`document.createElement` + `textContent`) over template strings with interpolation. See `elText()` in `course-page.js`.

## License / use

This is a personal study hub shared openly. Notes and curated resource lists
may be reused for non-commercial study with attribution. If you're an
instructor or cohort lead and want to fork this for your own program, please do.

## Code of conduct

Be respectful in issues and PRs. The maintainer reserves the right to close
unconstructive threads without comment.
