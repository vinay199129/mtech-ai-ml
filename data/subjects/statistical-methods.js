/* Introduction to Statistical Methods */
(window.COURSES = window.COURSES || []).push({
  slug: "statistical-methods",
  name: "Statistical Methods",
  semester: 1,
  desc: "Probability fundamentals, conditional probability and Bayes, random variables and distributions, hypothesis testing and confidence intervals, MLE, ANOVA, regression, time series (AR/MA/ARIMA/SARIMA/VAR), and Gaussian Mixture Models with EM.",
  textbooks: [
    "Kaptein et al. — Statistics for Data Scientists (Springer 2022)",
    "Jay Devore — Probability and Statistics for Engineering & the Sciences, 8e",
    "Brockwell & Davis — Introduction to Time Series and Forecasting, 2e"
  ],
  modules: [
    {
      n: 1, title: "Descriptive Statistics & Probability Basics",
      subtopics: ["Mean, median, mode (central tendency)", "Variance, std-dev, IQR (variability)", "5-point summary, outliers, skewness", "Sample space, events, axioms of probability"],
      ref: "T1, T2",
      notesFile: "01-descriptive-prob.md",
      resources: [
        { type: "video", title: "StatQuest — Descriptive vs Inferential Statistics", url: "https://www.youtube.com/watch?v=SzZ6GpcfoQY", desc: "Frame everything that follows in 5 min." },
        { type: "course", title: "Khan Academy — Statistics & Probability (full track)", url: "https://www.khanacademy.org/math/statistics-probability", desc: "Self-paced from zero — start with 'Analyzing categorical data' onward." },
        { type: "article", title: "Seeing Theory — Visual probability", url: "https://seeing-theory.brown.edu/basic-probability/index.html", desc: "Interactive Brown University site for axioms and basic probability." }
      ]
    },
    {
      n: 2, title: "Mutually Exclusive, Independent Events & Problem Solving",
      subtopics: ["Union, intersection, complement", "Mutually exclusive vs independent", "Inclusion–exclusion", "Counting: permutations & combinations"],
      ref: "T1, T2",
      notesFile: "02-events.md",
      resources: [
        { type: "video", title: "3Blue1Brown — Bayes' theorem (intuition for events)", url: "https://www.youtube.com/watch?v=HZGCoVF3YvM", desc: "Sets up dependence/independence visually." },
        { type: "article", title: "Devore Ch. 2 worked problems (online solution sets)", url: "https://www.probabilitycourse.com/chapter1/1_4_0_conditional_probability.php", desc: "Problem-driven walkthrough of independence vs disjoint." }
      ]
    },
    {
      n: 3, title: "Conditional Probability & Total Probability",
      subtopics: ["P(A|B) definition", "Multiplication rule", "Law of total probability", "Tree diagrams"],
      ref: "T1, T2",
      notesFile: "03-conditional-total.md",
      resources: [
        { type: "video", title: "StatQuest — Conditional probability", url: "https://www.youtube.com/watch?v=_IgyaD7vOOA", desc: "Builds the rule from Venn diagrams." },
        { type: "article", title: "ProbabilityCourse — Conditional probability (chap 1.4)", url: "https://www.probabilitycourse.com/chapter1/1_4_0_conditional_probability.php", desc: "Free textbook with solved examples." }
      ]
    },
    {
      n: 4, title: "Bayes' Theorem & Naïve Bayes",
      subtopics: ["Bayes' rule with proof", "Posterior, prior, likelihood, evidence", "Naïve Bayes classifier (Gaussian, multinomial, Bernoulli)", "Laplace smoothing"],
      ref: "T1, T2 + online",
      notesFile: "04-bayes-naive-bayes.md",
      resources: [
        { type: "video", title: "3Blue1Brown — Bayes' theorem, the geometry of changing beliefs", url: "https://www.youtube.com/watch?v=HZGCoVF3YvM", desc: "Best intuition for what Bayes really says." },
        { type: "article", title: "scikit-learn — Naive Bayes user guide", url: "https://scikit-learn.org/stable/modules/naive_bayes.html", desc: "Practical: variants, when each works, code." },
        { type: "video", title: "StatQuest — Naive Bayes, clearly explained", url: "https://www.youtube.com/watch?v=O2L2Uv9pdDA", desc: "End-to-end with multinomial example." }
      ]
    },
    {
      n: 5, title: "Random Variables — Discrete & Continuous",
      subtopics: ["Discrete vs continuous RV", "PMF, PDF, CDF", "Expectation, variance, covariance", "Joint, marginal, conditional distributions", "Transformation of RVs"],
      ref: "T1, T2",
      notesFile: "05-random-vars.md",
      resources: [
        { type: "video", title: "MIT 6.041 — Probability (J. Tsitsiklis), Lec 5–8", url: "https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/", desc: "Definitive course; lectures on discrete/continuous RVs and joint distributions." },
        { type: "article", title: "Seeing Theory — Probability distributions", url: "https://seeing-theory.brown.edu/probability-distributions/index.html", desc: "Interactive PDF/CDF playgrounds." }
      ]
    },
    {
      n: 6, title: "Common Distributions",
      subtopics: ["Bernoulli, Binomial, Poisson", "Normal (Gaussian)", "Student's t, F, Chi-square", "When to use which (sample size, variance known/unknown)"],
      ref: "T1, T2",
      notesFile: "06-distributions.md",
      resources: [
        { type: "video", title: "StatQuest — Probability distributions playlist", url: "https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9", desc: "One short video per distribution with clear story." },
        { type: "article", title: "Common probability distributions cheatsheet (Sean Owen)", url: "https://medium.com/@srowen/common-probability-distributions-the-data-scientists-crib-sheet-2c5d4c01d44a", desc: "Map every distribution to a use case." }
      ]
    },
    {
      n: 7, title: "Sampling, CLT, Estimation, Confidence Intervals",
      subtopics: ["Random vs stratified sampling", "Sampling distribution", "Central Limit Theorem (CLT)", "Point estimate vs interval estimate", "Confidence intervals (z, t)"],
      ref: "T1, T2",
      notesFile: "07-clt-ci.md",
      resources: [
        { type: "video", title: "StatQuest — The Central Limit Theorem", url: "https://www.youtube.com/watch?v=YAlJCEDH2uY", desc: "Why CLT matters in 8 minutes." },
        { type: "article", title: "PennState STAT 500 — CIs and sampling distributions", url: "https://online.stat.psu.edu/stat500/lesson/4", desc: "Walks through every CI formula with worked examples." }
      ]
    },
    {
      n: 8, title: "Hypothesis Testing — Means & Proportions",
      subtopics: ["Null vs alternative hypothesis", "Type I/II errors, power", "Z-test, t-test (one & two sample)", "One/two proportion tests", "p-value interpretation"],
      ref: "T1, T2",
      notesFile: "08-hypothesis-testing.md",
      resources: [
        { type: "video", title: "StatQuest — Hypothesis testing & p-values", url: "https://www.youtube.com/watch?v=0oc49DyA3hU", desc: "Removes p-value confusion completely." },
        { type: "article", title: "Hypothesis testing cheat sheet (PennState STAT 500 Lesson 6–8)", url: "https://online.stat.psu.edu/stat500/lesson/6", desc: "All test variants in one organized lesson tree." }
      ]
    },
    {
      n: 9, title: "MLE & ANOVA",
      subtopics: ["Likelihood function, log-likelihood", "Maximum Likelihood Estimation", "ANOVA — one-way (single factor)", "Two-way ANOVA", "F-statistic"],
      ref: "T1, T2",
      notesFile: "09-mle-anova.md",
      resources: [
        { type: "video", title: "StatQuest — Maximum Likelihood, clearly explained", url: "https://www.youtube.com/watch?v=XepXtl9YKwc", desc: "Worked example of fitting a normal." },
        { type: "video", title: "StatQuest — ANOVA, clearly explained", url: "https://www.youtube.com/watch?v=oOuu8IBd-yo", desc: "Why ANOVA is just an F-test in disguise." },
        { type: "article", title: "PennState STAT 502 — ANOVA full course", url: "https://online.stat.psu.edu/stat502/", desc: "Single & dual factor with derivations." }
      ]
    },
    {
      n: 10, title: "Correlation & Regression",
      subtopics: ["Pearson & Spearman correlation", "Simple & multiple linear regression", "OLS estimation", "R², residual analysis, assumptions"],
      ref: "T1, T2",
      notesFile: "10-correlation-regression.md",
      resources: [
        { type: "video", title: "StatQuest — Linear regression, fitting a line to data", url: "https://www.youtube.com/watch?v=nk2CQITm_eo", desc: "Derives least squares end-to-end." },
        { type: "article", title: "An Introduction to Statistical Learning — Ch. 3 (free PDF)", url: "https://www.statlearning.com/", desc: "ISLR ch 3 is the canonical regression chapter." }
      ]
    },
    {
      n: 11, title: "Time Series — Basics & MA Models",
      subtopics: ["Components: trend, seasonality, cyclic, residual", "Stationarity (weak/strong)", "Moving average (basic & weighted)", "ACF / PACF"],
      ref: "T3",
      notesFile: "11-ts-basics-ma.md",
      resources: [
        { type: "book", title: "Hyndman & Athanasopoulos — Forecasting: Principles and Practice (FREE)", url: "https://otexts.com/fpp3/", desc: "Open textbook. Read chapters 2, 3, 8 first — best on the planet." },
        { type: "video", title: "Ritvik Math — Time series intro", url: "https://www.youtube.com/watch?v=GE3JOFwTWVM", desc: "Visual breakdown of components and stationarity." }
      ]
    },
    {
      n: 12, title: "AR, ARMA, ARIMA",
      subtopics: ["Autoregressive (AR(p)) model", "Moving Average residual model", "ARMA, ARIMA(p,d,q)", "Box–Jenkins methodology"],
      ref: "T3",
      notesFile: "12-ar-arma-arima.md",
      resources: [
        { type: "book", title: "FPP3 — Chapter 9 (ARIMA models)", url: "https://otexts.com/fpp3/arima.html", desc: "ARIMA with R/Python examples and interpretation." },
        { type: "video", title: "Ritvik Math — ARIMA explained", url: "https://www.youtube.com/watch?v=3UmyHed0iYE", desc: "Step-by-step for a real series." }
      ]
    },
    {
      n: 13, title: "SARIMA, SARIMAX, VAR, VARMAX, Exponential Smoothing",
      subtopics: ["Seasonal ARIMA (SARIMA)", "SARIMAX (with exogenous regressors)", "Vector AR (VAR), VARMAX (multivariate)", "Simple/Holt/Holt-Winters exponential smoothing"],
      ref: "T3",
      notesFile: "13-sarima-var-ets.md",
      resources: [
        { type: "book", title: "FPP3 — Chapter 8 (Exponential smoothing) & §9.9 (Seasonal ARIMA)", url: "https://otexts.com/fpp3/expsmooth.html", desc: "Companion chapters covering all of session 14." },
        { type: "docs", title: "statsmodels — SARIMAX & VAR documentation", url: "https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html", desc: "Practical Python implementation with worked notebooks." }
      ]
    },
    {
      n: 14, title: "Gaussian Mixture Models & EM",
      subtopics: ["Mixture model intuition", "Latent variables", "E-step (responsibilities), M-step (parameter update)", "GMM vs k-means", "Convergence & log-likelihood monotonicity"],
      ref: "Class notes",
      notesFile: "14-gmm-em.md",
      resources: [
        { type: "video", title: "Victor Lavrenko — EM algorithm playlist", url: "https://www.youtube.com/playlist?list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt", desc: "Six short videos derive EM end-to-end." },
        { type: "article", title: "scikit-learn — Gaussian Mixture user guide", url: "https://scikit-learn.org/stable/modules/mixture.html", desc: "Practical use, BIC/AIC selection, full vs diag covariance." },
        { type: "article", title: "Bishop PRML — Chapter 9 (free PDF)", url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/", desc: "The definitive treatment of mixture models and EM." }
      ]
    }
  ]
});
