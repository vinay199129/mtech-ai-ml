<!-- markdownlint-disable-file -->
# Smoke Checklist — M.Tech AI/ML Study Hub

Run through this list before every release (any change merged to `main` that's
about to ship to GitHub Pages). Desktop only — mobile is intentionally out of
scope per PRD §4.

**Browsers:** at minimum Chrome + one of {Edge, Firefox, Safari} on desktop.

## 1. Per-page smoke (run on each URL)

URLs to exercise:

- `/` (home)
- `/courses/math-foundations.html`
- `/courses/statistical-methods.html`
- `/courses/machine-learning.html`
- `/courses/deep-neural-networks.html`

For each URL:

- [ ] Page loads with no console errors (open DevTools first).
- [ ] Favicon renders (no 404 in Network panel for `/assets/favicon.svg`).
- [ ] OG card renders when URL is pasted into Slack / Discord / a chat preview.
- [ ] Header brand-mark and nav links work.
- [ ] Skip-to-content link appears on first Tab and lands focus on `#main-content`.
- [ ] No CSP violations in console.

## 2. Course page interactivity (run on each course URL)

- [ ] Tab through the page; every interactive element receives a visible focus ring.
- [ ] Tab to a `.module-head`; press **Enter** — module opens, chevron rotates.
- [ ] Press **Space** on a focused module head — module toggles closed.
- [ ] Open a module that has notes — Quick Revise panel renders KaTeX math correctly.
- [ ] Open a module with a Mermaid block — diagram renders (light theme).
- [ ] Click a curated resource — opens in new tab with no warnings.
- [ ] Use the search box — filters modules; URL hash updates with `?q=...`.
- [ ] Click "Expand all" then "Collapse all" — both work; `aria-expanded` flips.
- [ ] Reload with `#m=3,5` in the URL — modules 3 and 5 auto-open and scroll into view.

## 3. Security smoke (FR-004 + FR-005)

- [ ] Add a temporary `notes/<subject>/00-xss-test.md` with payload:
      ```
      # XSS test
      <script>alert('xss-script')</script>
      <img src=x onerror="alert('xss-img')">
      ```
      Wire it temporarily into a module's `notesFile`. Open the module — **no alert fires**, payload is rendered as inert text or stripped. Revert before commit.
- [ ] Temporarily set a module resource title to `<img src=x onerror=alert(1)>` in a `data/subjects/*.js` file. Reload — title renders as literal text, no alert. Revert before commit.

## 4. Accessibility smoke (NFR-001)

- [ ] Run axe DevTools on home + 1 course page — zero serious / critical issues.
- [ ] Walk the home page using **keyboard only** (no mouse) — every link, button, and module is reachable; nothing is trapped.

## 5. Analytics (FR-009)

- [ ] If GoatCounter is configured (`window.SITE_GC_URL` in `assets/goatcounter.js` no longer contains `YOUR-SITE`), open DevTools → Network → reload home — confirm a request to `*.goatcounter.com/count` succeeds with 200 / 204.
- [ ] Open a module on a course page — confirm a second request fires with `path=module_open/<subject>/<n>`.
- [ ] Confirm **no cookies are set** by the analytics request (Application → Cookies).

## 6. Performance baseline (NFR-003)

- [ ] Run Lighthouse (desktop preset) on home + 1 course page (any one).
- [ ] Save HTML reports as `docs/lighthouse-baseline/home-YYYY-MM-DD.html` and `docs/lighthouse-baseline/<subject>-YYYY-MM-DD.html`.
- [ ] On the next release, re-run and confirm no individual category regresses by more than 5 points vs. the most recent baseline.

## 7. Positioning (FR-006)

- [ ] Home page semester sections show: Sem 1 = `Available`, Sem 2 = `Planned`, Sem 3/4 = `Not started`.
- [ ] Footer Roadmap mirrors the same labels (no `coming soon`).
- [ ] `<meta name="description">` mentions "evolving over 4 semesters, Semester 1 available now".

## 8. Pre-merge sanity

- [ ] No merge conflict markers anywhere.
- [ ] `git status` clean after build (no accidental tracked artifacts).
- [ ] `python -m http.server 8000` from project root serves the site without errors.

---

**Known limitations** (not blockers; tracked separately):

- OG card is SVG. Some platforms (Slack older clients, some email previews) may not render SVG OG images. Convert to PNG (1200×630) when a build step exists.
- `Content-Security-Policy` is set via `<meta http-equiv>` only; GitHub Pages cannot send HTTP CSP headers. This is the available mechanism on this host.
- Mobile is out of scope this iteration. Do not add mobile checks here.
