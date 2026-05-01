## Boosting — AdaBoost, Gradient Boosting, XGBoost

<div class="callout intuition"><span class="callout-title">Big picture (no jargon)</span>

Bagging trains models in **parallel**, all equal voters. **Boosting** trains models **sequentially**, each new one focused on what the previous ones got wrong. The result: a single strong learner built from a weighted sum of many weak ones, often the best off-the-shelf algorithm for tabular data.

Three flavours:
- **AdaBoost** — reweight *training samples* after each round (wrong samples get more weight).
- **Gradient Boosting** — fit each new model to the **negative gradient** of the loss (the "pseudo-residuals") instead of reweighting samples.
- **XGBoost** — production-grade gradient boosting with a 2nd-order Taylor expansion + explicit tree-complexity regularisation + huge engineering optimisations.

**Real-world analogy.** Studying for an exam by repeatedly taking practice tests: after each test, focus extra hard on the questions you got wrong. By the final test, you've drilled exactly the topics where you were weakest. Boosting is precisely this strategy applied to learners.

</div>

### Vocabulary — every term, defined plainly

- **Boosting** — sequentially train models, each correcting the previous ensemble's mistakes.
- **Weak learner** — base model with error slightly better than random (often a depth-1 tree, a "stump").
- **AdaBoost** (Adaptive Boosting) — reweights *samples*, with hard exponential weights on wrong predictions.
- **Sample weight $w_i^{(t)}$** — importance of sample $i$ at round $t$. Misclassified samples get up-weighted.
- **Vote weight $\alpha_t$** — how much weak learner $t$ contributes to the final ensemble.
- **Weighted error $\epsilon_t$** — sum of weights on misclassified samples in round $t$.
- **Gradient Boosting (GBM)** — generalises AdaBoost; treats boosting as gradient descent in *function* space.
- **Pseudo-residual** — negative gradient of the loss with respect to the current ensemble's prediction. For squared loss, equals the ordinary residual $y_i - F_{t-1}(\mathbf x_i)$.
- **Learning rate / shrinkage $\nu$** — multiplier on each new weak learner; smaller $\nu$ → more rounds needed but better generalisation.
- **XGBoost / LightGBM / CatBoost** — production GBM libraries.
- **2nd-order Taylor approximation** — XGBoost's trick: use both gradient $g_i$ and Hessian $h_i$ for sharper splits.
- **Tree complexity regulariser** — $\gamma T + \tfrac12 \lambda \sum w_j^2$ in XGBoost penalises number of leaves and large leaf weights.
- **Early stopping** — halt boosting when validation error stops improving.

### Picture it

```mermaid
flowchart LR
  D["Data with weights wᵢ"] --> M1["Train weak learner h₁"]
  M1 --> E1["Errors"]
  E1 -->|"reweight: up wrong, down right"| D2["New weights"]
  D2 --> M2["Train h₂"]
  M2 --> E2["Errors"] --> D3["New weights"]
  D3 --> M3["Train h₃ ..."]
  M3 --> Final["F = Σ αₜ · hₜ"]
```

### Build the idea — AdaBoost (binary, $y \in \{-1, +1\}$)

Initialise sample weights $w_i^{(1)} = 1/n$. For $t = 1, 2, \dots, T$:

1. Train weak learner $h_t$ on the current weighted data; compute weighted error
   $$
   \epsilon_t \;=\; \sum_{i=1}^n w_i^{(t)}\,\mathbf 1\{h_t(\mathbf x_i) \ne y_i\}.
   $$
2. Compute the learner's vote weight
   $$
   \alpha_t \;=\; \tfrac12 \ln \frac{1 - \epsilon_t}{\epsilon_t}.
   $$
   ($\alpha_t > 0$ when $\epsilon_t < 0.5$; $\alpha_t \to \infty$ when $\epsilon_t \to 0$; $\alpha_t < 0$ flips a worse-than-random learner.)
3. Update example weights and **renormalise** so they sum to 1:
   $$
   w_i^{(t+1)} \;\propto\; w_i^{(t)} \,\exp\!\big(-\alpha_t\,y_i\,h_t(\mathbf x_i)\big).
   $$
   Wrong samples ($y_i h_t(\mathbf x_i) = -1$) get multiplied by $e^{\alpha_t} > 1$ → up-weighted. Right samples by $e^{-\alpha_t} < 1$ → down-weighted.

Final classifier:

$$
F(\mathbf x) \;=\; \operatorname{sign}\!\left(\sum_{t=1}^T \alpha_t\,h_t(\mathbf x)\right).
$$

### Build the idea — Gradient Boosting (GBM)

Treat boosting as **gradient descent in function space**. After $t-1$ rounds we have ensemble $F_{t-1}$. Compute pseudo-residuals (negative gradient of loss $L$ wrt prediction):

$$
r_i^{(t)} \;=\; -\,\frac{\partial L(y_i, F)}{\partial F}\bigg|_{F = F_{t-1}(\mathbf x_i)}.
$$

For squared loss $L = \tfrac12 (y - F)^2$, this is just $r_i^{(t)} = y_i - F_{t-1}(\mathbf x_i)$ — the **ordinary residual**. So GBM with squared loss = "fit a tree to the residuals; repeat".

Fit weak learner $h_t$ to predict $r_i^{(t)}$, then update with shrinkage $\nu$:

$$
F_t(\mathbf x) \;=\; F_{t-1}(\mathbf x) \;+\; \nu\, h_t(\mathbf x).
$$

For other losses (log-loss for classification, Huber for robust regression, etc.), the procedure is identical — just plug in the appropriate gradient.

### Build the idea — XGBoost (production-grade GBM)

XGBoost adds three things on top of GBM:

**1. Second-order Taylor approximation** of the loss around the current prediction:

$$
\mathcal L^{(t)} \;\approx\; \sum_{i=1}^n \Big[\,g_i\,h_t(\mathbf x_i) \;+\; \tfrac12 h_i\,h_t(\mathbf x_i)^2\,\Big] \;+\; \Omega(h_t),
$$

where $g_i = \partial L / \partial F$ and $h_i = \partial^2 L / \partial F^2$ at $F_{t-1}$.

**2. Explicit tree-complexity regularisation:**

$$
\Omega(h_t) \;=\; \gamma\,T \;+\; \tfrac12 \lambda \sum_{j=1}^T w_j^2,
$$

where $T$ is the number of leaves and $w_j$ is the weight (output) of leaf $j$.

**3. Engineering:** histogram-based split finding, sparse-aware splits (handles missing data natively), parallelism, GPU support, cache-friendly layouts.

### Build the idea — comparison table

| | AdaBoost | GBM | XGBoost |
|---|---|---|---|
| Loss | Exponential (binary) | Any differentiable | Any (with $h_i$) |
| Reweighting | Sample weights | Function-space gradient | Gradient + Hessian |
| Regularisation | Implicit | LR + tree depth | + $L_1, L_2, \gamma T$ |
| Missing data | No native | No native | Native (default direction) |
| Speed | OK | OK | Very fast |
| Modern dominance | Niche | Common | Production standard (with LightGBM, CatBoost) |

<dl class="symbols">
  <dt>$\epsilon_t$</dt><dd>weighted training error of learner $h_t$ (must be $<0.5$ for useful learners)</dd>
  <dt>$\alpha_t$</dt><dd>vote weight of $h_t$; high when $\epsilon_t$ small</dd>
  <dt>$w_i^{(t)}$</dt><dd>weight on sample $i$ at round $t$ — wrong samples up-weighted</dd>
  <dt>$g_i, h_i$</dt><dd>1st and 2nd derivatives of loss wrt prediction at sample $i$</dd>
  <dt>$T$</dt><dd>number of leaves in the new tree (XGBoost)</dd>
  <dt>$w_j$</dt><dd>weight (output value) of leaf $j$</dd>
  <dt>$\nu$</dt><dd>learning rate / shrinkage; smaller = more rounds needed, better generalisation</dd>
  <dt>$\gamma, \lambda$</dt><dd>complexity regularisation strengths</dd>
</dl>

### Worked example — fully expanded

<div class="callout example"><span class="callout-title">Worked example: round 1 of AdaBoost on 5 samples</span>

**Setup.** $n = 5$ samples, initial weights $w_i^{(1)} = 0.2$ each. The chosen weak learner (a stump) misclassifies samples 2 and 4.

**Step 1 — weighted error.**

$\epsilon_1 = w_2^{(1)} + w_4^{(1)} = 0.2 + 0.2 = 0.4$.

**Step 2 — vote weight.**

$\alpha_1 = \tfrac12 \ln\!\left(\dfrac{1 - 0.4}{0.4}\right) = \tfrac12 \ln(0.6 / 0.4) = \tfrac12 \ln(1.5) \approx \tfrac12 \cdot 0.4055 \approx 0.203$.

**Step 3 — unnormalised reweighting.**

For wrong samples (2 and 4): $w \cdot e^{\alpha_1} = 0.2 \cdot e^{0.203} = 0.2 \cdot 1.225 \approx 0.245$.

For right samples (1, 3, 5): $w \cdot e^{-\alpha_1} = 0.2 \cdot e^{-0.203} = 0.2 \cdot 0.816 \approx 0.163$.

Unnormalised totals: $3 \cdot 0.163 + 2 \cdot 0.245 = 0.490 + 0.490 = 0.980$.

**Step 4 — renormalise so weights sum to 1.**

Each right sample: $0.163 / 0.980 \approx 0.166$. Each wrong sample: $0.245 / 0.980 \approx 0.250$.

Final round-2 weights: $(0.166, 0.250, 0.166, 0.250, 0.166)$. Sum $= 0.998 \approx 1$ ✓ (rounding).

**Step 5 — sanity check.** The two misclassified samples each got their weight bumped from 0.20 to 0.25 (×1.25), while correctly classified samples dropped from 0.20 to 0.166 (×0.83). The next weak learner will be trained on this reweighted distribution — it will work harder on samples 2 and 4.

**Step 6 — what happens to the final score?** Sample 2 (wrong this round) contributes $-\alpha_1 = -0.203$ to its current ensemble score. If the next two rounds correctly classify it with vote weights $\alpha_2 = 0.4, \alpha_3 = 0.5$, its cumulative score becomes $-0.203 + 0.4 + 0.5 = 0.697 > 0$ → predicted +1, correctly. The boosting fixed it.

</div>

### How to think about it

<div class="callout intuition"><span class="callout-title">Mental model — a relentless complaint loop</span>

After each round, the algorithm "complains" about the still-misclassified examples and forces the next weak learner to focus there. The surprise: even with thousands of rounds, test error often *keeps dropping* well past zero training error (Schapire's margin theory: boosting maximises a margin distribution, similar in spirit to SVM).

Gradient Boosting is the conceptual breakthrough: rather than reweighting samples, fit each new learner to the **negative gradient** of the loss at the current ensemble's predictions. For squared loss this is exactly the residuals — "fit residuals, repeat" is the simplest boosting idea you can implement in 10 lines. For other losses, plug in the right gradient and you get classification, ranking, Poisson regression, etc.

**When this comes up in ML.** Boosting (especially XGBoost / LightGBM / CatBoost) dominates **tabular** ML — Kaggle competitions, finance, click-through-rate prediction, anything where features are mixed numeric / categorical and $N \le 10^7$. It's the answer to "what's the strongest off-the-shelf model for tabular data?" until the dataset becomes truly huge.

</div>

<div class="callout warn"><span class="callout-title">Watch out — common traps</span>

- **AdaBoost is sensitive to noisy labels and outliers** — they get up-weighted forever, dominating later rounds. Use Gradient Boosting with a robust loss (Huber) when noise is heavy.
- **Boosting can overfit** if you push $T$ very large with no regularisation. Use **early stopping** on a validation set.
- **Use small learning rate $\nu$** (0.01–0.1) + many trees, *not* large $\nu$ + few trees. Slower but better generalisation.
- **XGBoost has many knobs** (`max_depth`, `eta` = $\nu$, `subsample`, `colsample_bytree`, `gamma`, `lambda`, `min_child_weight`); tune via grid / random search or Bayesian optimisation.
- **Sequential by nature** — boosting cannot be parallelised across rounds (only within a round). For huge datasets, prefer LightGBM (histogram-based, fast) or CatBoost (handles categoricals natively).
- **Class imbalance.** Boosting with cross-entropy can suffer; use `scale_pos_weight` in XGBoost, or focal loss.

</div>

<div class="callout tip"><span class="callout-title">Exam tip</span>

Memorise the **three AdaBoost lines** (weighted error → $\alpha_t = \tfrac12 \ln((1-\epsilon)/\epsilon)$ → exponential reweighting). Be able to **trace round 1 numerically** for a small (4–5 sample) example — the calculation above is the canonical exam question. Be able to **explain "GBM is gradient descent in function space"** in one paragraph and write the **second-order Taylor expansion** that XGBoost uses.

</div>
