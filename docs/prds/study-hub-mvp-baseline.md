<!-- markdownlint-disable-file -->
<!-- markdown-table-prettify-ignore-start -->
# M.Tech AI/ML Study Hub — MVP Baseline PRD
Version 0.2 | Status Approved for build | Owner Maintainer (vinay199129) | Team Solo | Target MVP declared 2026-05-01 (Phase 1–3 roll forward) | Lifecycle MVP

## Progress Tracker
| Phase | Done | Gaps | Updated |
|-------|------|------|---------|
| Context | done | — | 2026-05-01 |
| Problem & Users | done | persona validation through cohort feedback (post-launch) | 2026-05-01 |
| Scope | done | — | 2026-05-01 |
| Requirements | done | acceptance criteria locked | 2026-05-01 |
| Metrics & Risks | done | analytics baseline captured after FR-009 ships | 2026-05-01 |
| Operationalization | done | GitHub Pages + GoatCounter | 2026-05-01 |
| Finalization | in progress | implementation pending | 2026-05-01 |
Unresolved Critical Questions: 0 | TBDs: 1 (cohort feedback channel)

## 1. Executive Summary
### Context
A free, zero-build static study hub (HTML/CSS/JS hosted on **GitHub Pages**) for the M.Tech AI/ML curriculum. Today it ships Semester 1 content (4 subjects, ~54 modules, ~140 curated free resources) with quick-revise notes rendered from local Markdown; Semesters 2–4 are planned and will evolve incrementally. The site was just re-themed to match the A365 starter look-and-feel and the navigation simplified. Recent design + PM reviews surfaced a backlog mixing **correctness/conformance fixes**, **speculative scaling work**, and **visual preferences**.
### Core Opportunity
Convert the current "themed but unvalidated" build into a **defensible MVP for the maintainer + study cohort + public discoverers** by (a) closing the small set of correctness/accessibility/security gaps that block trustworthy use, (b) keeping the 4-semester roadmap visible but **honest** with milestone framing on Sem-2/3/4 placeholders, and (c) deferring desktop/scaling work until content, cohort feedback, or contributors actually demand it. Mobile is explicitly out of scope for this iteration.
### Goals
| Goal ID | Statement | Type | Baseline | Target | Timeframe | Priority |
|---------|-----------|------|----------|--------|-----------|----------|
| G-001 | Site is usable end-to-end via keyboard alone (every interactive element reachable + visible focus) on **desktop**. | Quality | Modules not keyboard-reachable; no `:focus-visible` ring. | WCAG 2.1 AA pass on 2.1.1 + 2.4.7 across all course pages. | This iteration | Must |
| G-002 | Notes rendering pipeline cannot execute author-supplied scripts. | Security | Raw `marked` output + `innerHTML` for resource fields. | Markdown sanitized via DOMPurify; user-content fields use `textContent`. | This iteration | Must |
| G-003 | Home-page positioning is **honest about the 4-semester roadmap** — Sem-2/3/4 placeholders carry an explicit status (planned / in progress / not started). | Trust | 3 "coming soon" semester placeholders shown with no status. | Each placeholder shows status + (where applicable) milestone. | This iteration | Must |
| G-004 | Maintainer revises Sem-1 material faster using the site than from raw notes. | UX (self) | Anecdotal only. | Maintainer self-report after one revision session. | Within 30 days of MVP | Must |
| G-005 | Site quality is measurable, not opinion-based. | Process | No analytics, no Lighthouse baseline, no smoke checklist. | Lighthouse baseline captured; cookieless page-view + module-open tracking via GoatCounter; 1-page manual smoke checklist in repo. | This iteration | Must |
| G-006 | Public learners can discover and share the site. | Reach | No OG cards, no favicon, no analytics on referrals. | Shareable links render preview cards (FR-008); GoatCounter referrer report enabled. | This iteration | Should |
| G-007 | Cohort can give structured feedback. | UX (cohort) | No feedback channel. | A documented feedback channel (GitHub Issues link in footer + cohort comms note). | This iteration | Should |

### Objectives (Optional)
| Objective | Key Result | Priority | Owner |
|-----------|------------|----------|-------|
| Make MVP defensible for self + cohort + public | All Must-priority Goals complete; smoke checklist passes on 4 course pages on desktop | Must | Maintainer |
| Enable measurable iteration | GoatCounter live; Lighthouse baseline captured; GitHub Issues open as feedback channel | Must | Maintainer |
| Avoid speculative scaling work | Backlog explicitly defers A2/A5/D1/D4 until trigger event (contributor / Sem-2 start / measured friction) | Should | Maintainer |

## 2. Problem Definition
### Current Situation
- Single-author repo, no contributor docs, no analytics, no test plan.
- Recent theme migration to A365 palette is visually consistent; nav was simplified to scale.
- Design review (this conversation) enumerated ~30 issues spanning IA, visual, a11y, code, perf, SEO, content.
- PM review flagged that the design backlog conflates four workstreams and is not anchored to a user / success metric.

### Problem Statement
The site looks polished and themes well, but a small number of correctness gaps (keyboard accessibility, notes-pipeline XSS surface) and one positioning ambiguity ("4 semesters but only 1 shipped") undermine trust and usability for the actual user. Without an explicit MVP definition, the next round of work risks burning time on speculative scaling and visual preferences instead.

### Root Causes
* No written product brief — primary user, top jobs-to-be-done, and success metric are undefined.
* Design review treated correctness, polish, and refactors as one backlog.
* No instrumentation, so prioritization is opinion-based.
* Positioning ("4-semester study hub") was inherited from file-tree layout, not a deliberate scope decision.

### Impact of Inaction
* Inaccessible to keyboard-only users → fails WCAG 2.1 AA → blocks any institutional adoption.
* Notes pipeline becomes a security liability the moment external PRs are accepted.
* Ongoing churn on visual/structural changes with no feedback loop on whether they help.
* "Coming soon × 3" reads as abandoned, lowering perceived quality of the Sem-1 content that *is* good.

## 3. Users & Personas
Three co-primary personas. The site must serve all three; design tie-breaks favor **Maintainer (self)** since that is the only voice with verified feedback today.

| Persona | Goals | Pain Points | Impact |
|---------|-------|------------|--------|
| **P1 — Maintainer / M.Tech student (self)** | Quick revision before Sem-1 exams; find a topic by name; work through a module's notes + 2–3 trusted resources | Modules not keyboard-reachable; no quick way to jump to a topic | Drives G-004 (revise faster); blocks own use until G-001 ships |
| **P2 — Study cohort** | Use the site as shared revision material; flag errors / suggest additions | No feedback channel; no shareable section anchors beyond `#m=N` | Drives G-007 (feedback channel) and FR-011 (CONTRIBUTING.md) |
| **P3 — Public learner discovering the site** | Browse curriculum, decide whether it's worth bookmarking, share with peers | No OG cards, missing favicon, unclear roadmap status | Drives G-006 (reach), FR-006 (honest roadmap), FR-007/FR-008 |
| **Deferred — future external contributor** | Add notes/resources without breaking the build | Raw HTML allowed in notes (until FR-004) | In scope: minimal `CONTRIBUTING.md` (FR-011); full contributor flow deferred |

### Journeys (Optional)
- **P1 revision journey**: Home → Sem-1 → Subject card → expand module → read notes → click a resource. Target: ≤ 3 interactions to module body, fully keyboard-operable.
- **P3 discovery journey**: External link (shared by P2) → preview card renders (FR-008) → lands on subject page → scans modules → bookmarks home.
- **P2 feedback journey**: Spot an error → footer "Report an issue" link → opens GitHub Issues with prefilled template (deferred; MVP ships plain link).

## 4. Scope
### In Scope (MVP)
* **Quality bundle** — keyboard accessibility (G-001), `:focus-visible` ring, ARIA on collapsible modules, sanitize notes output (G-002), escape user-content fields, fix favicon 404.
* **Positioning bundle** — keep 4-semester structure; add explicit status badges (planned / in progress / not started) to Sem-2/3/4 placeholders (FR-006). Tagline stays "M.Tech AI/ML — Study Hub."
* **Measurement bundle** — capture Lighthouse desktop baseline; add cookieless page-view + module-open tracking via **GoatCounter**; commit 1-page manual smoke checklist; add `<meta>` OpenGraph tags so shared links render.
* **Documentation bundle** — short `README.md` (what / who / run locally / add a module) and `CONTRIBUTING.md` (FR-011) for cohort + future contributors.
* **Content review bundle** — proofread all `notes/**/*.md` for proof / formula / typo correctness before MVP declared (FR-012).

### Out of Scope (deferred to a later release; revisit on trigger event)
* **Mobile UX** — explicitly deferred this iteration. All a11y / Lighthouse / smoke checks target desktop. Revisit when cohort or analytics show mobile traffic > 20%.
* **A2** Sticky docs-style sidebar TOC for course pages → defer until Sem-2 begins OR analytics show navigation friction.
* **A5** Global cross-subject search → defer until > 1 semester of content.
* **D1** Migrate `window.COURSES` to JSON manifest → defer until a contributor exists.
* **D4** Collapse 4 course HTML files into a single template → defer; current cost is low and would invalidate bookmarks.
* **D6** Stylelint/Prettier/CI tooling → defer until a contributor exists.
* **B6** "Notes" badge on module heads → defer; conflicts with "chrome stays constant as content grows" principle.
* **E1** Lazy-load mermaid/KaTeX → defer until measured perf problem.
* **Issue templates** beyond a single "Report a content error" label — deferred until first external contributor appears.

### Assumptions
* Hosted on **GitHub Pages** as a static site. No backend, no auth, no user accounts.
* Solo maintainer; cohort + public consume read-only; contributor PRs welcome but not advertised.
* Target user has a modern evergreen browser on a **desktop / laptop**. Mobile not supported this iteration.
* Notes content is authored by trusted maintainer(s); FR-004 sanitization protects against future contributor risk.

### Constraints
* **Zero-build by design** — any new tooling must be opt-in (no build step required to view the site).
* **No backend** — analytics/tracking must be client-only and **cookieless** (GoatCounter selected).
* **Bookmark stability** — existing `courses/<slug>.html` URLs and `#m=…` hash deep-links must keep working.
* **GitHub Pages constraints** — no custom server headers; CSP/security headers must be set via `<meta http-equiv>` where possible, or accepted as best-effort.
* **Solo maintainer time** — bias toward small, reversible changes; avoid refactors without a triggering need.

## 5. Product Overview
### Value Proposition
A trustworthy, accessible, free study companion for M.Tech AI/ML — module-by-module notes plus the best 2–3 free resources per topic, optimized for exam-window revision.

### Differentiators (Optional)
* Curated free resources (no affiliate / no paywall friction).
* Quick-revise notes with KaTeX + Mermaid built in.
* Zero-build, hostable on any static-file server in seconds.
* Bookmark-friendly hash state (`#m=3,5&q=svm`).

### UX / UI (Conditional)
Theme + layout already migrated to A365-inspired light palette in prior session. UX status: shipped, awaiting cohort validation. No further visual changes in this PRD unless they support a Goal. Mobile responsiveness is explicitly **not** addressed in this iteration.

## 6. Functional Requirements
| FR ID | Title | Description | Goals | Personas | Priority | Acceptance | Notes |
|-------|-------|------------|-------|----------|----------|-----------|-------|
| FR-001 | Keyboard-operable modules | Every `.module-head` is reachable via Tab and toggles open/closed via Enter and Space. | G-001 | Primary | Must | Manual: Tab through a course page; every module receives focus with a visible ring; Enter and Space toggle; `aria-expanded` updates correctly. | Convert `<div>` to `<button>` or add `role="button"` + `tabindex="0"` + key handlers + `aria-expanded`/`aria-controls`. |
| FR-002 | Visible focus indicator | A `:focus-visible` outline appears on every interactive element. | G-001 | Primary, Secondary | Must | Tab through home + 1 course page; outline visible against both white and blue (hero) backgrounds. | Single global rule in `assets/styles.css`. |
| FR-003 | Skip-to-content link | Keyboard users can skip past the site header. | G-001 | Primary | Should | Tab once on any page; "Skip to main content" link appears and focuses `#course-root` / main on activation. | Standard pattern; ~10 lines of CSS + HTML. |
| FR-004 | Sanitize rendered Markdown | Notes Markdown rendered to HTML is passed through DOMPurify before injection. | G-002 | All | Must | Author a notes file with `<script>alert(1)</script>` and an `onerror` image; both are stripped after render. | Add DOMPurify CDN script to course pages; wrap `marked.parse(...)` output. |
| FR-005 | Escape resource fields | Resource title and description are inserted via `textContent`, not `innerHTML`. | G-002 | All | Must | Set a resource title containing `<img src=x onerror=alert(1)>`; renders as literal text. | Refactor `course-page.js` resource builder. |
| FR-006 | Honest 4-semester roadmap on home page | Home keeps all four semester sections; Sem-2/3/4 placeholders carry an explicit status badge (`Planned` / `In progress` / `Not started`) and short note. Tagline stays "M.Tech AI/ML — Study Hub." | G-003 | P3 | Must | Visual inspection: each of Sem-2/3/4 cards renders a status pill; no "coming soon" without a label; meta description mentions "evolving over 4 semesters, Semester 1 available now." | Status pill styled with existing palette; no new color tokens. |
| FR-007 | Favicon present | No 404 for `/favicon.ico`. | G-005 | All | Must | Browser network panel shows 200 for favicon on every page. | Reuse "AI" brand-mark style as a 32×32 SVG favicon. |
| FR-008 | OpenGraph + Twitter card tags | All four course pages + home have `og:title`, `og:description`, `og:image`. | G-006 | P3 | Must | Paste a course URL into a chat app preview; renders title + description + image. | Single image asset reused; absolute URLs to GitHub Pages domain. |
| FR-009 | Cookieless analytics via GoatCounter | Page views and module-open events captured via GoatCounter — no cookies, no PII. | G-005, G-006 | Maintainer | Must | Maintainer can see "module X of subject Y was opened N times this week" in the GoatCounter dashboard; referrer report shows where public traffic came from. | Use GoatCounter `count()` JS API for `module_open`; site code is the standard `<script data-goatcounter="..." async src="//gc.zgo.at/count.js">` snippet. |
| FR-010 | Smoke checklist in repo | Repo contains a one-page manual checklist exercised before each release. | G-005 | Maintainer | Must | `docs/SMOKE-CHECKLIST.md` exists with: 4 course pages × {open, expand a module, deep-link `#m=N`, keyboard-only nav, axe DevTools clean}. **Desktop only.** | Checklist is markdown only — no automation in this PRD. |
| FR-011 | CONTRIBUTING.md | Repo contains `CONTRIBUTING.md` describing how cohort + future contributors can suggest content fixes or add modules. | G-007 | P2 | Must | File exists and covers: how to file an issue, how to propose a notes edit (PR flow), Markdown conventions for notes, KaTeX/Mermaid usage, what NOT to put in notes (no raw `<script>`, no inline `<iframe>`). | Footer link "Contribute" points to it. |
| FR-012 | Content correctness review | Every Markdown file under `notes/**/*.md` is proofread by the maintainer before MVP declared — proofs, formulas, definitions, broken links. | G-004 | P1 | Must | Tracking checklist (one row per file) committed at `docs/CONTENT-REVIEW.md`; every row checked before Phase 4 closes. | Findings captured as PR commits, not as separate issues. |

### Feature Hierarchy (Optional)
```plain
Quality bundle           → FR-001, FR-002, FR-003, FR-004, FR-005
Positioning bundle       → FR-006
Measurement bundle       → FR-007, FR-008, FR-009, FR-010
Documentation bundle     → FR-011
Content review bundle    → FR-012
```

## 7. Non-Functional Requirements
| NFR ID | Category | Requirement | Metric/Target | Priority | Validation | Notes |
|--------|----------|------------|--------------|----------|-----------|-------|
| NFR-001 | Accessibility | WCAG 2.1 AA conformance for the success criteria touched by FR-001..FR-003 — **desktop scope**. | Manual + axe-core: zero serious/critical issues on home + all 4 course pages. | Must | axe DevTools + keyboard-only walkthrough. | Mobile a11y deferred. |
| NFR-002 | Security | No script execution from author-supplied notes content. | DOMPurify enabled with default profile; `<meta http-equiv="Content-Security-Policy">` set with a least-privilege policy that allows only the CDN origins in use (jsDelivr/cdn for KaTeX, marked, mermaid, DOMPurify, GoatCounter). | Must | Manual XSS test cases pass per FR-004 acceptance; browser console shows no CSP violations on legitimate pages. | GitHub Pages cannot set HTTP headers; meta-CSP is the available mechanism. |
| NFR-003 | Performance | Lighthouse **desktop** score baseline captured; no regression > 5 points after this iteration. | Lighthouse desktop run on home + 1 course page, before and after. | Must | Screenshots stored at `docs/lighthouse-baseline/`. | Mobile Lighthouse deferred. Optimization (E1/E4) explicitly out of scope. |
| NFR-004 | Compatibility | Works on latest Chrome, Edge, Firefox, Safari — **desktop only**. | Manual smoke per FR-010 on at least 2 browsers. | Must | Smoke checklist. | No mobile, no legacy browser support. |
| NFR-005 | Maintainability | All MVP changes are reversible by file revert; no new build step. | `git revert` of any single PR restores prior behavior. | Must | Code review. | Reinforces zero-build constraint. |
| NFR-006 | Privacy | Analytics is cookieless and contains no PII. | GoatCounter docs: no cookies, IPs hashed and discarded after session window. | Must | GoatCounter selection recorded in §15. | Vendor selected. |

## 8. Data & Analytics (Conditional)
### Inputs
None from the user (no forms, no auth). Author-supplied content: `data/subjects/*.js` + `notes/<subject>/*.md`.

### Outputs / Events
| Event | Trigger | Payload | Purpose | Owner |
|-------|---------|--------|---------|-------|
| `page_view` | Page load | path, referrer | Identify hot/cold pages | Maintainer |
| `module_open` | User opens a `.module` | subject slug, module number | Validate "navigation friction" hypothesis before building sidebar | Maintainer |
| `notes_load_error` | `loadNotes()` catch branch | subject slug, module number, error message | Detect broken `notesFile` references | Maintainer |

### Instrumentation Plan
GoatCounter selected. Standard count snippet on every page (home + 4 course pages); explicit `goatcounter.count({path: 'module_open/<subject>/<n>'})` call inside the module-open handler in `assets/course-page.js`.

### Metrics & Success Criteria
| Metric | Type | Baseline | Target | Window | Source |
|--------|------|----------|--------|--------|--------|
| WCAG 2.1 AA conformance (touched criteria, desktop) | Quality | Fail (2.1.1, 2.4.7) | Pass | This iteration | Manual + axe |
| XSS smoke tests (FR-004 + FR-005) | Security | n/a | All pass | This iteration | Manual |
| Lighthouse desktop (home + 1 course) | Performance | TBD (capture before changes) | No regression > 5 pts | This iteration | Lighthouse |
| GitHub stars | Reach (G-006) | Capture on launch day | +N over 30 days (target TBD after baseline) | 30 days post-MVP | GitHub |
| Cohort feedback items | UX (G-007) | 0 | ≥ 3 actionable items received | 30 days post-MVP | GitHub Issues |
| Maintainer self-report: "revised faster" | UX (G-004) | Anecdotal | One revision session completed using site as primary source | 30 days post-MVP | Self-report in changelog |
| Module-open events / week | UX | 0 | Captured | After FR-009 ships | GoatCounter |

## 9. Dependencies
| Dependency | Type | Criticality | Owner | Risk | Mitigation |
|-----------|------|------------|-------|------|-----------|
| DOMPurify (CDN) | External library | High | Maintainer | CDN outage breaks notes rendering | Pin version + SRI hash; consider self-hosting the file in `assets/`. |
| GoatCounter | External service | Medium | Maintainer | Service outage = no analytics (no user impact) | Async script tag; failures silent. |
| GitHub Pages | Infra | High | Maintainer | Hosting change invalidates URLs | Pin to GitHub Pages for MVP; use repo's `gh-pages` or `main` branch convention. |
| KaTeX / marked / mermaid CDNs | External library | Medium | Maintainer | CDN outage breaks math/markdown/diagrams | Pin versions + SRI; rendering failures degrade gracefully. |

## 10. Risks & Mitigations
| Risk ID | Description | Severity | Likelihood | Mitigation | Owner | Status |
|---------|-------------|---------|-----------|-----------|-------|--------|
| R-001 | Building scaling features (sidebar, search, JSON manifest) before Sem-2 content exists wastes effort and bloats the codebase. | High | High (default trajectory of design review) | Explicitly defer items A2/A5/D1/D4/D6/B6 in §4 Out of Scope; document trigger event. | Maintainer | Open |
| R-002 | Accepting external PRs before FR-004/FR-005 land introduces XSS. | High | Low today, High once contributors invited | Do not advertise contributions until quality bundle ships. | Maintainer | Open |
| R-003 | "Coming soon × 3" framing erodes trust in the Sem-1 content that does ship. | Medium | High (current state) | Resolve via FR-006 in this iteration. | Maintainer | Open |
| R-004 | Optimizing perf without baseline (E1/E4) breaks math/diagram rendering. | Medium | Medium | Capture Lighthouse baseline (NFR-003) before any perf work; defer perf changes from this PRD. | Maintainer | Mitigated (deferred) |
| R-005 | Prioritization remains opinion-based without analytics. | Medium | High | FR-009 + FR-010 land in this iteration to enable next iteration to be data-informed. | Maintainer | Open |
| R-006 | Solo-maintainer bus factor — no contributor docs, all context in one head. | Medium | Medium | Minimal `README.md` in this iteration; full contributor flow deferred. | Maintainer | Open |

## 11. Privacy, Security & Compliance
### Data Classification
No user data collected beyond pseudonymous page-view events (FR-009).

### PII Handling
No PII collected. Analytics vendor must be cookieless and IP-anonymizing (NFR-006).

### Threat Considerations
* **XSS via notes content** — addressed by FR-004 + FR-005.
* **CDN tampering** — pin all CDN assets to specific versions with SRI hashes (already done for KaTeX; extend to marked, mermaid, DOMPurify).
* **Clickjacking** — low risk for read-only static site; recommend `X-Frame-Options: DENY` at deploy time.

### Regulatory / Compliance (Conditional)
| Regulation | Applicability | Action | Owner | Status |
|-----------|--------------|--------|-------|--------|
| WCAG 2.1 AA | Aspirational; mandatory if institutionally adopted | FR-001..FR-003 + NFR-001 | Maintainer | In progress |
| GDPR / privacy | Triggered if EU traffic + analytics | Cookieless analytics (NFR-006) avoids most obligations | Maintainer | Mitigated by design |

## 12. Operational Considerations
| Aspect | Requirement | Notes |
|--------|------------|-------|
| Deployment | **GitHub Pages** from the repo's default branch (or `gh-pages`). Pushes auto-deploy. | No CI step required. |
| Rollback | Git revert + push; Pages redeploys automatically. | Trivial. |
| Monitoring | GoatCounter dashboard reviewed weekly during exam window. | Lightweight. |
| Alerting | None in MVP. | Out of scope. |
| Support | GitHub Issues on the repo; link in footer. | No SLA. |
| Capacity Planning | Static; GitHub Pages handles scale. | n/a |

## 13. Rollout & Launch Plan
User asked for "today" (2026-05-01) as target. Interpreted as: **PRD baseline locked today; implementation phases roll forward immediately, MVP declared when all Must FRs land**. If literal same-day shipping is required, scope must drop to Quality + Positioning bundles only — flag in changelog if that path is chosen.

### Phases / Milestones
| Phase | Date | Gate Criteria | Owner |
|-------|------|--------------|-------|
| 0. PRD baseline locked | 2026-05-01 | This document approved | Maintainer |
| 1. Quality bundle | +1 to +3 days | FR-001..FR-005 merged; manual a11y + XSS smoke passes | Maintainer |
| 2. Positioning bundle | +0 to +1 day after Phase 1 | FR-006 applied: status pills on Sem-2/3/4; meta description updated | Maintainer |
| 3. Measurement bundle | +1 to +2 days after Phase 2 | FR-007..FR-010 merged; Lighthouse desktop baseline captured; GoatCounter live | Maintainer |
| 4. Documentation + Content review | +2 to +5 days after Phase 3 | FR-011 (`CONTRIBUTING.md`) + FR-012 (every notes file proofread) complete | Maintainer |
| 5. MVP declared | When Phase 4 closes | All Must FRs + NFRs satisfied; smoke checklist passes; cohort + public link shared | Maintainer |

### Feature Flags (Conditional)
None. Static site.

### Communication Plan (Optional)
On MVP declared: share GitHub Pages URL with cohort (channel TBD by maintainer); pin a "Feedback welcome — file an issue" note on the repo README. No public/social announcement required.

## 14. Open Questions
| Q ID | Question | Owner | Deadline | Status |
|------|----------|-------|---------|--------|
| Q-001 | Who is the **primary** user? | User | — | **Resolved 2026-05-01**: Co-primary — maintainer (self) + study cohort + public learners. Tie-breaks favor maintainer. |
| Q-002 | Single success metric? | User | — | **Resolved 2026-05-01**: Composite — (1.a) maintainer revises faster + (1.b) cohort gives positive feedback + (1.d) GitHub stars/shares grow. Reflected in G-004, G-006, G-007 + §8 metrics table. |
| Q-003 | Positioning: Sem-1-only or 4-semester? | User | — | **Resolved 2026-05-01**: Keep all 4 semesters; add status pills on Sem-2/3/4. See FR-006. |
| Q-004 | Mobile UX in scope? | User | — | **Resolved 2026-05-01**: Out of scope this iteration. Desktop only. See §4 Out of Scope. |
| Q-005 | Deployment target? | User | — | **Resolved 2026-05-01**: GitHub Pages. See §12. |
| Q-006 | Analytics vendor? | User | — | **Resolved 2026-05-01**: GoatCounter (free, cookieless, hosted). See FR-009 + NFR-006. |
| Q-007 | CONTRIBUTING.md in MVP? | User | — | **Resolved 2026-05-01**: Yes — FR-011. |
| Q-008 | Target release date? | User | — | **Resolved 2026-05-01**: "Today" (2026-05-01) interpreted as PRD lock + immediate Phase 1 start. See §13 note. |
| Q-009 | Content correctness review in scope? | User | — | **Resolved 2026-05-01**: Yes — FR-012. |
| Q-010 | What channel does the cohort use for feedback (GitHub Issues only, or also a chat/email channel)? | User | Before Phase 5 | **Open** — defaults to GitHub Issues link in footer if unanswered. |

## 15. Changelog
| Version | Date | Author | Summary | Type |
|---------|------|-------|---------|------|
| 0.1 | 2026-05-01 | PRD Builder + maintainer | Initial draft from design + PM review of the just-themed site. | Initial |
| 0.2 | 2026-05-01 | PRD Builder + maintainer | Locked: 3 co-primary personas (self/cohort/public), composite success metric (G-004/G-006/G-007), 4-semester roadmap with status pills (FR-006), mobile out of scope, GitHub Pages + GoatCounter ops picks, added FR-011 (CONTRIBUTING.md) + FR-012 (content review), 8 of 9 open questions resolved. Status moved Draft → Approved for build. | Revision |

## 16. References & Provenance
| Ref ID | Type | Source | Summary | Conflict Resolution |
|--------|------|--------|---------|--------------------|
| REF-001 | Conversation | Design review (this session) | 30+ items spanning IA, visual, a11y, code, perf, SEO, content. | Items reclassified into in-scope (must-fix) vs. deferred. |
| REF-002 | Conversation | PM review (this session) | Flagged scope conflation, missing user/metric, security under-prioritized, "coming soon" trust risk. | Drove the §4 Out-of-Scope list and §10 risk register. |
| REF-003 | Codebase | `assets/course-page.js`, `assets/styles.css`, `index.html`, `courses/*.html` | Current implementation baseline. | Source of truth for FR acceptance phrasing. |
| REF-004 | External | A365 Governed Pro-Code Agent Starter — github.com/vinay199129/a365-governed-procode-agent-starter | Theme/layout reference applied in prior session. | Visual baseline; no further dependency. |

### Citation Usage
References are conversational + codebase only; no external research integrated yet.

## 17. Appendices (Optional)
### Glossary
| Term | Definition |
|------|-----------|
| Quick-revise card | Self-contained, exam-ready Markdown notes file rendered into a module body. |
| WCAG | Web Content Accessibility Guidelines. |
| DOMPurify | XSS sanitizer for HTML/SVG/MathML injected into the DOM. |
| Cookieless analytics | Page-view tracking that does not set cookies and does not require a consent banner under GDPR. |

### Additional Notes
This PRD intentionally **excludes** items that the design review listed but the PM review flagged as speculative or premature. Those items live in §4 Out of Scope with an explicit trigger event for revisit.

Generated 2026-05-01 by PRD Builder (mode: full, v0.2)
<!-- markdown-table-prettify-ignore-end -->
