# PRD template for research / ML projects

Derived from a general product-PRD reference. Kept: executive summary, problem, scope, requirements, risks, timeline, open questions. Removed: market sizing, go-to-market, pricing, budget, wireframes. Added: data specification, modeling and evaluation, execution environment, privacy, acceptance checklist.

Copy this file to `docs/<project>_PRD.md` (or `docs/PRD.md` if the repository holds a single project) and delete what does not apply. Keep section numbers stable because plans and prompts refer to them.

## Header
Status, date, authors, advisor, privacy rule (placeholders only).

## 1. Executive summary
1.1 Problem (2-3 sentences). 1.2 Proposed solution. 1.3 Expected impact (3 bullets). 1.4 Resources (people, compute, data). 1.5 Success metrics (3-5; for research, evidence- and process-based rather than a promised number). 1.6 Definition of Done.

## 2. Problem definition
2.1 Users. 2.2 Context. 2.3 What the data actually looks like (counts, missingness, confounds, with an audit command that reproduces them). 2.4 Cost of not solving.

## 3. Scope
3.1 In scope (ID, item, priority P0-P2). 3.2 Out of scope. 3.3 MVP. 3.4 Learning goals (the questions the project answers).

## 4. Requirements
4.1 User stories with acceptance criteria. 4.2 Functional requirements. 4.3 Non-functional: reproducibility, privacy, portability, performance, robustness, code style.

## 5. Data specification
Input contract, eligible rows, feature groups (included, excluded with reasons), missing-value rules, transforms, labels and class filtering.

## 6. Modeling and evaluation
Stages, baselines, models, **validation design that matches how the model will be used** (group by batch/date/subject to prevent leakage), diagnostics (trivial baselines, permutation test, spread), metrics, ablations, final model, pre-trained model decision, optional experiments.

## 7. Technical specification
Code layout, CLI, configuration, outputs and artifacts, dependencies.

## 8. Execution environment
Primary and fallback machines, data movement, retrieving results; runbook in an appendix.

## 9. Testing and QA
9.1 Approach and the verification loop. 9.2 Edge-case catalog (every row is a required task). 9.3 Acceptance checklist.

## 10. Phases, gates and exit criteria
Table of phases with exit criteria and review gates.

## 11. Risks and mitigations
Table: risk, probability, impact, mitigation.

## 12. Open questions
Table: question, default if unanswered.

## Appendices
Glossary; runbook; any data audit tables (aggregated, with placeholders).
