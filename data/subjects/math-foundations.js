/* Math Foundations for ML */
(window.COURSES = window.COURSES || []).push({
  slug: "math-foundations",
  name: "Math Foundations for ML",
  semester: 1,
  handout: "semesters/sem-1/math-foundations/handout.pdf",
  desc: "Linear algebra, matrix decompositions, vector calculus, multivariate Taylor series, gradient descent (incl. SGD/Adam family), constrained/convex optimization, dimensionality reduction (PCA), and SVM optimization.",
  textbooks: [
    "Deisenroth, Faisal, Ong — Mathematics for Machine Learning (Cambridge, 2020) — FREE PDF: mml-book.com",
    "Charu Aggarwal — Linear Algebra and Optimization for ML (Springer, 2020)"
  ],
  modules: [
    {
      n: 1, title: "Solution of Linear Systems",
      subtopics: ["Systems of linear equations", "Matrix form Ax=b", "Gaussian elimination, row-echelon, rank", "Existence & uniqueness of solutions"],
      ref: "T1: §2.1–2.3",
      notesFile: "01-linear-systems.md",
      resources: [
        { type: "video", title: "3Blue1Brown — Essence of Linear Algebra (Ch. 1–8)", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab", desc: "The single best visual intuition for vectors, span, linear systems, and matrices." },
        { type: "book",  title: "MML Book — Chapter 2 (free PDF)", url: "https://mml-book.github.io/", desc: "Official textbook, fully open. Read §2.1–2.3 directly." },
        { type: "article", title: "Gilbert Strang — Solving Ax=b (MIT 18.06 notes)", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/video-lectures/lecture-2-elimination-with-matrices/", desc: "Strang's elimination lecture — gold standard." }
      ]
    },
    {
      n: 2, title: "Vector Spaces, Bases, Norms, Inner Products",
      subtopics: ["Linear independence, basis, rank", "Norms (L1, L2, L∞)", "Inner products, lengths, distances", "Angles, orthogonality, orthonormal basis", "Gram–Schmidt"],
      ref: "T1: §2.4–2.8, §3.1–3.5",
      notesFile: "02-vector-spaces.md",
      resources: [
        { type: "video", title: "3Blue1Brown — Inner products & duality", url: "https://www.youtube.com/watch?v=LyGKycYT2v0", desc: "Geometric intuition for dot product and orthogonality." },
        { type: "video", title: "Strang — Orthogonal vectors & subspaces (MIT 18.06)", url: "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/video-lectures/lecture-14-orthogonal-vectors-and-subspaces/", desc: "Bases, orthogonality, projections in one lecture." },
        { type: "article", title: "Gram–Schmidt explained step-by-step", url: "https://textbooks.math.gatech.edu/ila/gram-schmidt.html", desc: "Worked numerical examples from Georgia Tech ILA." }
      ]
    },
    {
      n: 3, title: "Matrix Decomposition I — Determinant, Trace, Eigen, Cholesky",
      subtopics: ["Determinant & trace properties", "Eigenvalues, eigenvectors, characteristic polynomial", "Symmetric positive-definite matrices", "Cholesky decomposition"],
      ref: "T1: §4.1–4.3",
      notesFile: "03-matrix-decomp-1.md",
      resources: [
        { type: "video", title: "3Blue1Brown — Eigenvectors & eigenvalues", url: "https://www.youtube.com/watch?v=PFDu9oVAE-g", desc: "Visual derivation of eigenstuff in 17 minutes." },
        { type: "book", title: "MML Book — Chapter 4", url: "https://mml-book.github.io/", desc: "Reads exactly with the syllabus references." },
        { type: "article", title: "Cholesky Decomposition — Rosetta Code & derivation", url: "https://en.wikipedia.org/wiki/Cholesky_decomposition", desc: "Formal definition, numerical recipe, worked example." }
      ]
    },
    {
      n: 4, title: "Matrix Decomposition II — Eigendecomposition, SVD, Low-Rank Approx",
      subtopics: ["Diagonalization", "Singular Value Decomposition (SVD)", "Truncated SVD, Eckart–Young theorem", "Matrix approximation / low-rank"],
      ref: "T1: §4.4–4.6",
      notesFile: "04-svd-low-rank.md",
      resources: [
        { type: "video", title: "Steve Brunton — SVD (full playlist)", url: "https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv", desc: "Best modern SVD course on YouTube — intuition, math, code." },
        { type: "video", title: "3Blue1Brown — Change of basis & similar matrices", url: "https://www.youtube.com/watch?v=P2LTAUO1TdA", desc: "Why diagonalization works geometrically." },
        { type: "article", title: "Gentle intro to SVD (Jeremy Kun)", url: "https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/", desc: "Clean prose derivation tying SVD to projections and PCA." }
      ]
    },
    {
      n: 5, title: "Vector Calculus I — Gradients (scalar & vector fields)",
      subtopics: ["Differentiation of univariate functions (refresher)", "Partial derivatives & gradient ∇f", "Jacobian for vector-valued functions", "Chain rule (multivariate)"],
      ref: "T1: §5.1–5.3",
      notesFile: "05-vec-calc-1.md",
      resources: [
        { type: "video", title: "Khan Academy — Multivariable Calculus", url: "https://www.khanacademy.org/math/multivariable-calculus", desc: "From gradient through Jacobian — work through 'Derivatives of multivariable functions'." },
        { type: "book", title: "MML Book — Chapter 5", url: "https://mml-book.github.io/", desc: "Concise, self-contained chapter aligned with handout." },
        { type: "article", title: "The Matrix Calculus You Need For Deep Learning (Parr & Howard)", url: "https://explained.ai/matrix-calculus/", desc: "Single best free reference for Jacobians and matrix derivatives." }
      ]
    },
    {
      n: 6, title: "Vector Calculus II — Matrix gradients, Backprop, Autodiff",
      subtopics: ["Gradients of matrices wrt matrices", "Useful identities (trace tricks, ∂/∂X tr(AX))", "Backpropagation as chain rule", "Forward & reverse-mode automatic differentiation"],
      ref: "T1: §5.4–5.6",
      notesFile: "06-vec-calc-2.md",
      resources: [
        { type: "article", title: "Matrix Cookbook (Petersen & Pedersen)", url: "https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf", desc: "Cheat-sheet of every matrix-derivative identity you'll ever need." },
        { type: "article", title: "Karpathy — 'Yes you should understand backprop'", url: "https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b", desc: "Gotchas + intuition from someone who teaches it." },
        { type: "video", title: "3Blue1Brown — Backpropagation calculus", url: "https://www.youtube.com/watch?v=tIeHLnjs5U8", desc: "Derives backprop from chain rule visually." }
      ]
    },
    {
      n: 7, title: "Vector Calculus III — Higher-order derivatives, Taylor, Hessian",
      subtopics: ["Higher-order partial derivatives", "Linearization", "Multivariate Taylor series", "Computing maxima/minima (unconstrained), Hessian test"],
      ref: "T1: §5.7–5.8 + class notes",
      notesFile: "07-vec-calc-3.md",
      resources: [
        { type: "video", title: "Multivariate Taylor series (Visually Explained)", url: "https://www.youtube.com/watch?v=3d6DsjIBzJ4", desc: "Geometric picture of 2nd-order approximation." },
        { type: "article", title: "Hessian matrix & convexity (Wolfram MathWorld)", url: "https://mathworld.wolfram.com/Hessian.html", desc: "Compact reference with worked examples." },
        { type: "book", title: "Boyd & Vandenberghe — Convex Optimization (free)", url: "https://web.stanford.edu/~boyd/cvxbook/", desc: "Chapter 2–3 are the gold standard for convexity & 2nd-order conditions." }
      ]
    },
    {
      n: 8, title: "Continuous Optimization — Gradient Descent & Lagrangians",
      subtopics: ["Gradient descent — vanilla, line search", "Constrained optimization with Lagrange multipliers", "KKT conditions", "Convex optimization basics"],
      ref: "T1: §7.1–7.3",
      notesFile: "08-cont-opt.md",
      resources: [
        { type: "video", title: "Stanford CS229 — Convex Optimization Overview", url: "https://www.youtube.com/watch?v=McLq1hEq3UY", desc: "Compact 1-hour primer aligned with ML use cases." },
        { type: "course", title: "Boyd — Convex Optimization I (Stanford EE364a)", url: "https://www.youtube.com/playlist?list=PL3940DD956CDF0622", desc: "Full Stanford lectures, freely available." },
        { type: "article", title: "Sebastian Ruder — An overview of gradient descent algorithms", url: "https://www.ruder.io/optimizing-gradient-descent/", desc: "Definitive blog post mapping every GD variant." }
      ]
    },
    {
      n: 9, title: "Nonlinear Optimization I — Practical SGD",
      subtopics: ["Learning rate decay, initialization", "SGD vs batch", "Hyperparameter tuning", "Feature preprocessing & normalization"],
      ref: "T2: §4.4–4.5",
      notesFile: "09-sgd.md",
      resources: [
        { type: "article", title: "Distill — Why Momentum Really Works", url: "https://distill.pub/2017/momentum/", desc: "Interactive visualization — best resource on optimization dynamics." },
        { type: "article", title: "Goodfellow et al. — Deep Learning, Ch. 8 (free)", url: "https://www.deeplearningbook.org/contents/optimization.html", desc: "Standard reference for SGD and its variants." },
        { type: "video", title: "Andrew Ng — Optimization algorithms (deeplearning.ai)", url: "https://www.youtube.com/watch?v=lAq96T8FkTw", desc: "Concise lecture covering mini-batch, momentum, learning rate decay." }
      ]
    },
    {
      n: 10, title: "Nonlinear Optimization II — Adam family & difficult landscapes",
      subtopics: ["Local minima, saddle points, cliffs, plateaus", "Momentum, Nesterov", "AdaGrad, RMSProp, Adam, AdaDelta"],
      ref: "T2: §5.2–5.3",
      notesFile: "10-adam.md",
      resources: [
        { type: "article", title: "Sebastian Ruder — Gradient descent variants (deep dive)", url: "https://www.ruder.io/optimizing-gradient-descent/", desc: "Adam, RMSProp, AdaGrad with formulas and references." },
        { type: "article", title: "Adam: A Method for Stochastic Optimization (Kingma & Ba)", url: "https://arxiv.org/abs/1412.6980", desc: "Original Adam paper — short and readable." },
        { type: "video", title: "Optimization for Deep Learning (Stanford CS231n L7)", url: "https://www.youtube.com/watch?v=_JB0AO7QxSA", desc: "Justin Johnson walks through every modern optimizer." }
      ]
    },
    {
      n: 11, title: "Dimensionality Reduction & PCA",
      subtopics: ["Maximum variance perspective", "Projection perspective", "Eigenvector & low-rank approximations", "PCA in high dimensions, latent variable view"],
      ref: "T1: §10.1–10.7",
      notesFile: "11-pca.md",
      resources: [
        { type: "video", title: "StatQuest — Principal Component Analysis (PCA), step-by-step", url: "https://www.youtube.com/watch?v=FgakZw6K1QQ", desc: "Crystal-clear intuition before diving into math." },
        { type: "book", title: "MML Book — Chapter 10 (free)", url: "https://mml-book.github.io/", desc: "All four perspectives covered with derivations." },
        { type: "article", title: "Setosa — PCA explained visually", url: "http://setosa.io/ev/principal-component-analysis/", desc: "Interactive visualization of PCA." }
      ]
    },
    {
      n: 12, title: "Support Vector Machines — Primal/Dual & Kernels",
      subtopics: ["KKT conditions for SVM", "Primal & dual formulations (linear SVM)", "Hard vs soft margin", "Kernel trick — polynomial, RBF"],
      ref: "T1: §12.1–12.6, T2: §6.4",
      notesFile: "12-svm.md",
      resources: [
        { type: "video", title: "Caltech ML — SVM (Yaser Abu-Mostafa, L14–15)", url: "https://www.youtube.com/watch?v=eHsErlPJWUU", desc: "Best lecture on SVM derivation including the dual." },
        { type: "article", title: "Burges — A Tutorial on SVM for Pattern Recognition", url: "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/svmtutorial.pdf", desc: "The classic tutorial cited in your handout." },
        { type: "video", title: "StatQuest — SVM main ideas + kernels", url: "https://www.youtube.com/watch?v=efR1C6CvhmE", desc: "Friendly intuition for primal, dual, and kernel trick." }
      ]
    }
  ]
});
