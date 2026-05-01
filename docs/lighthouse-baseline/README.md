<!-- markdownlint-disable-file -->
# Lighthouse baselines

Per **NFR-003**, capture a Lighthouse desktop report for the home page and at
least one course page before each release. Save reports here with the naming
convention:

```text
home-YYYY-MM-DD.html
math-foundations-YYYY-MM-DD.html
statistical-methods-YYYY-MM-DD.html
machine-learning-YYYY-MM-DD.html
deep-neural-networks-YYYY-MM-DD.html
```

**Regression rule:** no individual category (Performance, Accessibility, Best
Practices, SEO) may drop by more than **5 points** vs. the most recent baseline.

## How to capture

1. Run the site locally: `python -m http.server 8000` from the project root.
2. Open Chrome → DevTools → **Lighthouse** tab.
3. Mode: **Navigation**, Device: **Desktop**, Categories: all four.
4. Click **Analyze page load** for the home URL.
5. In the Lighthouse report, click the three-dot menu → **Save as HTML** → save
   into this folder using the naming convention above.
6. Repeat for one course page.

## Baselines

| Date | Page | Performance | Accessibility | Best Practices | SEO | Notes |
|------|------|-------------|---------------|----------------|-----|-------|
| _pending_ | home | — | — | — | — | Capture before MVP declared. |
| _pending_ | math-foundations | — | — | — | — | Capture before MVP declared. |
