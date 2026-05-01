<!-- markdownlint-disable-file -->
# Content correctness review tracker

Per **FR-012** in [the PRD](prds/study-hub-mvp-baseline.md), every Markdown
notes file must be proofread by the maintainer before MVP is declared. Check
each row when:

1. Proofs and derivations are mathematically correct.
2. Formulas (KaTeX) render without errors and match the surrounding text.
3. Definitions agree with the cited textbooks listed in the subject's
   `data/subjects/<slug>.js`.
4. External resource links (`url:` fields in the resources array of the
   matching module) return 200 and are still on-topic.
5. No typos in headings, callout titles, or symbol-definition lists.

Findings are committed as edits to the underlying file (PR commits), not as
separate issues.

## math-foundations (12 files)

| Status | File | Reviewer | Notes |
|--------|------|----------|-------|
| [ ] | `notes/math-foundations/01-linear-systems.md` | maintainer | |
| [ ] | `notes/math-foundations/02-vector-spaces.md` | maintainer | |
| [ ] | `notes/math-foundations/03-matrix-decomp-1.md` | maintainer | |
| [ ] | `notes/math-foundations/04-svd-low-rank.md` | maintainer | |
| [ ] | `notes/math-foundations/05-vec-calc-1.md` | maintainer | |
| [ ] | `notes/math-foundations/06-vec-calc-2.md` | maintainer | |
| [ ] | `notes/math-foundations/07-vec-calc-3.md` | maintainer | |
| [ ] | `notes/math-foundations/08-cont-opt.md` | maintainer | |
| [ ] | `notes/math-foundations/09-sgd.md` | maintainer | |
| [ ] | `notes/math-foundations/10-adam.md` | maintainer | |
| [ ] | `notes/math-foundations/11-pca.md` | maintainer | |
| [ ] | `notes/math-foundations/12-svm.md` | maintainer | |

## statistical-methods (14 files)

| Status | File | Reviewer | Notes |
|--------|------|----------|-------|
| [ ] | `notes/statistical-methods/01-descriptive-prob.md` | maintainer | |
| [ ] | `notes/statistical-methods/02-events.md` | maintainer | |
| [ ] | `notes/statistical-methods/03-conditional-total.md` | maintainer | |
| [ ] | `notes/statistical-methods/04-bayes-naive-bayes.md` | maintainer | |
| [ ] | `notes/statistical-methods/05-random-vars.md` | maintainer | |
| [ ] | `notes/statistical-methods/06-distributions.md` | maintainer | |
| [ ] | `notes/statistical-methods/07-clt-ci.md` | maintainer | |
| [ ] | `notes/statistical-methods/08-hypothesis-testing.md` | maintainer | |
| [ ] | `notes/statistical-methods/09-mle-anova.md` | maintainer | |
| [ ] | `notes/statistical-methods/10-correlation-regression.md` | maintainer | |
| [ ] | `notes/statistical-methods/11-ts-basics-ma.md` | maintainer | |
| [ ] | `notes/statistical-methods/12-ar-arma-arima.md` | maintainer | |
| [ ] | `notes/statistical-methods/13-sarima-var-ets.md` | maintainer | |
| [ ] | `notes/statistical-methods/14-gmm-em.md` | maintainer | |

## machine-learning (13 files)

| Status | File | Reviewer | Notes |
|--------|------|----------|-------|
| [ ] | `notes/machine-learning/01-intro-ml.md` | maintainer | |
| [ ] | `notes/machine-learning/02-math-workflow.md` | maintainer | |
| [ ] | `notes/machine-learning/03-linear-regression.md` | maintainer | |
| [ ] | `notes/machine-learning/04-logistic.md` | maintainer | |
| [ ] | `notes/machine-learning/05-decision-trees.md` | maintainer | |
| [ ] | `notes/machine-learning/06-instance-based.md` | maintainer | |
| [ ] | `notes/machine-learning/07-svm.md` | maintainer | |
| [ ] | `notes/machine-learning/08-bayesian-learning.md` | maintainer | |
| [ ] | `notes/machine-learning/09-naive-bayes.md` | maintainer | |
| [ ] | `notes/machine-learning/10-ensemble-bagging.md` | maintainer | |
| [ ] | `notes/machine-learning/11-boosting.md` | maintainer | |
| [ ] | `notes/machine-learning/12-unsupervised.md` | maintainer | |
| [ ] | `notes/machine-learning/13-evaluation.md` | maintainer | |

## deep-neural-networks (15 files)

| Status | File | Reviewer | Notes |
|--------|------|----------|-------|
| [ ] | `notes/deep-neural-networks/01-intro.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/02-perceptron.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/03-linear-nn-regression.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/04-linear-nn-classification.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/05-mlp-backprop.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/06-cnn.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/07-cnn-architectures.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/08-transfer-learning.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/09-rnn.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/10-lstm-gru.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/11-attention.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/12-transformer.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/13-vit-pretraining.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/14-deep-optimization.md` | maintainer | |
| [ ] | `notes/deep-neural-networks/15-deep-regularization.md` | maintainer | |

## Roll-up

| Subject | Files | Reviewed | % |
|---------|------:|---------:|--:|
| math-foundations | 12 | 0 | 0% |
| statistical-methods | 14 | 0 | 0% |
| machine-learning | 13 | 0 | 0% |
| deep-neural-networks | 15 | 0 | 0% |
| **Total** | **54** | **0** | **0%** |

MVP gate: all 54 rows checked.
