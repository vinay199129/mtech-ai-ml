/* Machine Learning */
(window.COURSES = window.COURSES || []).push({
  slug: "machine-learning",
  name: "Machine Learning",
  semester: 1,
  handout: "semesters/sem-1/machine-learning/handout.docx",
  desc: "Foundations of ML: workflow & data prep; linear/logistic regression; decision trees; instance-based learning (kNN, LWR, RBF); SVMs; Bayesian learning (MLE/MAP, Naïve Bayes); ensembles (bagging, RF, boosting, AdaBoost, GBM, XGBoost); unsupervised learning (k-means, GMM/EM); model evaluation, fairness, interpretability.",
  textbooks: [
    "Tom Mitchell — Machine Learning (McGraw-Hill, 1997)",
    "Bishop — Pattern Recognition & Machine Learning (Springer, 2006) — free PDF on author page",
    "Tan, Steinbach, Kumar — Introduction to Data Mining, 2e"
  ],
  modules: [
    {
      n: 1, title: "Introduction to ML & Designing a Learning System",
      subtopics: ["What is ML, applications", "Supervised, unsupervised, reinforcement", "Designing a learning system (Mitchell's framework)", "Issues in ML: generalization, overfitting, bias"],
      ref: "T1 Ch. 1",
      notesFile: "01-intro-ml.md",
      resources: [
        { type: "course", title: "Andrew Ng — Machine Learning Specialization (Coursera, free audit)", url: "https://www.coursera.org/specializations/machine-learning-introduction", desc: "Modern replacement for the legendary CS229 intro." },
        { type: "video", title: "MIT 6.034 — Intro to Learning (Patrick Winston)", url: "https://www.youtube.com/watch?v=sQO5fZW9SUQ", desc: "Classic lecture motivating ML problem framing." },
        { type: "article", title: "Mitchell Ch. 1 summary (Carnegie Mellon notes)", url: "http://www.cs.cmu.edu/~tom/mlbook.html", desc: "Author's companion page for the textbook." }
      ]
    },
    {
      n: 2, title: "Math Preliminaries & ML Workflow",
      subtopics: ["LA, calculus, probability, info theory recap", "Data pipeline: ingestion → cleaning → feature engineering", "Train/val/test split, k-fold CV", "Performance metrics: accuracy, precision, recall, F1, ROC-AUC"],
      ref: "R2 Ch. 2–3 + lecture notes",
      notesFile: "02-math-workflow.md",
      resources: [
        { type: "article", title: "scikit-learn — A complete ML workflow tutorial", url: "https://scikit-learn.org/stable/tutorial/basic/tutorial.html", desc: "Hands-on canonical workflow." },
        { type: "video", title: "StatQuest — Confusion matrix, ROC, AUC", url: "https://www.youtube.com/watch?v=4jRBRDbJemM", desc: "All evaluation metrics explained simply." },
        { type: "article", title: "Google ML Crash Course — Framing, Data, Generalization", url: "https://developers.google.com/machine-learning/crash-course", desc: "Free, polished, exam-aligned." }
      ]
    },
    {
      n: 3, title: "Linear Regression",
      subtopics: ["Direct (closed-form) solution: normal equation", "Gradient descent: batch / SGD / mini-batch", "Linear basis function models (polynomial, RBF features)", "Bias–variance decomposition"],
      ref: "R1 Ch. 3",
      notesFile: "03-linear-regression.md",
      resources: [
        { type: "video", title: "StatQuest — Linear Regression and Bias-Variance", url: "https://www.youtube.com/watch?v=EuBBz3bI-aA", desc: "Two short videos that nail both topics." },
        { type: "article", title: "Bishop PRML — Chapter 3 (linear models for regression)", url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/", desc: "Definitive textbook chapter — free PDF." },
        { type: "course", title: "Andrew Ng — Linear regression with one/multiple variables", url: "https://www.coursera.org/learn/machine-learning", desc: "Original Coursera lectures, still the cleanest intro." }
      ]
    },
    {
      n: 4, title: "Logistic Regression & Linear Classification",
      subtopics: ["Discriminant functions, decision boundary", "Probabilistic discriminative classifiers", "Sigmoid, softmax", "Logloss / cross-entropy", "Multi-class classification"],
      ref: "R1 Ch. 3–4",
      notesFile: "04-logistic.md",
      resources: [
        { type: "video", title: "StatQuest — Logistic Regression playlist", url: "https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe", desc: "Maximum likelihood, log loss, ROC — all of it." },
        { type: "article", title: "CS229 — Logistic Regression notes (Andrew Ng)", url: "https://cs229.stanford.edu/notes2022fall/main_notes.pdf", desc: "Stanford notes derive logistic regression from MLE." }
      ]
    },
    {
      n: 5, title: "Decision Trees",
      subtopics: ["Information gain, entropy, Gini", "ID3 / C4.5 / CART", "Avoiding overfitting: pruning", "Minimum Description Length", "Continuous attributes & missing values"],
      ref: "T1 Ch. 3, R2 Ch. 3",
      notesFile: "05-decision-trees.md",
      resources: [
        { type: "video", title: "StatQuest — Decision Trees, clearly explained", url: "https://www.youtube.com/watch?v=_L39rN6gz7Y", desc: "Best 17-minute intro." },
        { type: "article", title: "Tan, Steinbach, Kumar — Ch. 3 (decision tree induction)", url: "https://www-users.cse.umn.edu/~kumar001/dmbook/index.php", desc: "Companion site for textbook with sample chapter." },
        { type: "docs", title: "scikit-learn — Decision Trees user guide", url: "https://scikit-learn.org/stable/modules/tree.html", desc: "Practical: hyperparameters, pruning, cost-complexity." }
      ]
    },
    {
      n: 6, title: "Instance-Based Learning",
      subtopics: ["k-Nearest Neighbors", "Distance metrics", "Locally Weighted Regression (LWR)", "Radial Basis Functions", "Curse of dimensionality"],
      ref: "T1 Ch. 8",
      notesFile: "06-instance-based.md",
      resources: [
        { type: "video", title: "StatQuest — k-Nearest Neighbors, clearly explained", url: "https://www.youtube.com/watch?v=HVXime0nQeI", desc: "Classification + regression with kNN." },
        { type: "article", title: "Mitchell Ch. 8 — Instance-based learning notes", url: "http://www.cs.cmu.edu/~tom/mlbook-chapter-slides.html", desc: "Mitchell's official slides for Ch. 8." }
      ]
    },
    {
      n: 7, title: "Support Vector Machines",
      subtopics: ["Maximum margin classifier", "Linear SVM (soft margin)", "Kernel trick (RBF, polynomial)", "Mercer's theorem", "Multi-class SVM"],
      ref: "R2 Ch. 4 + Burges tutorial",
      notesFile: "07-svm.md",
      resources: [
        { type: "video", title: "StatQuest — SVM (Part 1, 2, 3)", url: "https://www.youtube.com/watch?v=efR1C6CvhmE", desc: "Three videos: intuition → polynomial → RBF." },
        { type: "article", title: "Burges — Tutorial on SVM (Microsoft Research PDF)", url: "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/svmtutorial.pdf", desc: "Cited in handout — read sections 1–4." }
      ]
    },
    {
      n: 8, title: "Bayesian Learning — MLE, MAP, Bayes Rule, Optimal Bayes",
      subtopics: ["Likelihood, prior, posterior", "MLE vs MAP", "Bayes optimal classifier", "Brute force MAP learner"],
      ref: "T1 Ch. 6",
      notesFile: "08-bayesian-learning.md",
      resources: [
        { type: "video", title: "StatQuest — MLE & MAP", url: "https://www.youtube.com/watch?v=p3T-_LMrvBc", desc: "Two short videos on parameter estimation." },
        { type: "article", title: "Mitchell Ch. 6 — Bayesian Learning slides", url: "http://www.cs.cmu.edu/~tom/mlbook-chapter-slides.html", desc: "Author's official slide deck." }
      ]
    },
    {
      n: 9, title: "Naïve Bayes & Generative Classifiers",
      subtopics: ["Naïve Bayes assumption", "Gaussian / Multinomial / Bernoulli NB", "Generative vs discriminative", "Bayesian linear regression"],
      ref: "T1 Ch. 6, R1 Ch. 4",
      notesFile: "09-naive-bayes.md",
      resources: [
        { type: "video", title: "StatQuest — Naive Bayes (Multinomial & Gaussian)", url: "https://www.youtube.com/watch?v=O2L2Uv9pdDA", desc: "Both versions in two short videos." },
        { type: "docs", title: "scikit-learn — Naive Bayes guide", url: "https://scikit-learn.org/stable/modules/naive_bayes.html", desc: "Practical recipes and when each variant fits." }
      ]
    },
    {
      n: 10, title: "Ensemble Learning — Bagging & Random Forest",
      subtopics: ["Bias–variance for ensembles", "Bootstrap aggregation (bagging)", "Random Forest (feature subsetting)", "Out-of-bag error"],
      ref: "R2 Ch. 4",
      notesFile: "10-ensemble-bagging.md",
      resources: [
        { type: "video", title: "StatQuest — Random Forests, clearly explained", url: "https://www.youtube.com/watch?v=J4Wdy0Wc_xQ", desc: "Bootstrap + bagging + RF in one go." },
        { type: "article", title: "Breiman — Random Forests (original paper)", url: "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf", desc: "Short, foundational paper." }
      ]
    },
    {
      n: 11, title: "Boosting — AdaBoost, Gradient Boosting, XGBoost",
      subtopics: ["Boosting idea: weak learners → strong learner", "AdaBoost: re-weighting", "Gradient Boosting (functional gradient descent)", "XGBoost: regularization, tree pruning, parallelism"],
      ref: "R2 Ch. 4 + lecture notes",
      notesFile: "11-boosting.md",
      resources: [
        { type: "video", title: "StatQuest — Gradient Boosting playlist (Pt 1–4)", url: "https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF", desc: "From AdaBoost through XGBoost." },
        { type: "article", title: "XGBoost docs — Introduction to Boosted Trees", url: "https://xgboost.readthedocs.io/en/stable/tutorials/model.html", desc: "Math + practice from the library authors." }
      ]
    },
    {
      n: 12, title: "Unsupervised Learning — k-Means, GMM, EM",
      subtopics: ["k-Means algorithm & variants (k-means++)", "Choosing k (elbow, silhouette)", "Mixture models", "EM algorithm", "Soft vs hard clustering"],
      ref: "T1 Ch. 6",
      notesFile: "12-unsupervised.md",
      resources: [
        { type: "video", title: "StatQuest — k-means clustering & GMM", url: "https://www.youtube.com/watch?v=4b5d3muPQmA", desc: "Pair with the EM playlist linked in ZC418 §14." },
        { type: "docs", title: "scikit-learn — Clustering user guide", url: "https://scikit-learn.org/stable/modules/clustering.html", desc: "Full comparison table of clustering algorithms." }
      ]
    },
    {
      n: 13, title: "Model Evaluation, Comparison, Fairness, Interpretability",
      subtopics: ["Cross-validation properly done", "Statistical comparison of models (paired t-test, McNemar)", "Bias/fairness metrics", "Interpretability: SHAP, LIME, feature importance"],
      ref: "T1 Ch. 5 + lecture notes",
      notesFile: "13-evaluation.md",
      resources: [
        { type: "article", title: "Christoph Molnar — Interpretable ML (free book)", url: "https://christophm.github.io/interpretable-ml-book/", desc: "Definitive open book covering SHAP, LIME, partial dependence." },
        { type: "article", title: "Google — Fairness in ML (ML Crash Course)", url: "https://developers.google.com/machine-learning/crash-course/fairness/video-lecture", desc: "Concise primer + case studies." },
        { type: "video", title: "StatQuest — Cross-validation, clearly explained", url: "https://www.youtube.com/watch?v=fSytzGwwBVw", desc: "Get CV right before model comparison." }
      ]
    }
  ]
});
