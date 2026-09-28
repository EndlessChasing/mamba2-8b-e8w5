# Readapted8B MK baseline: output error types

## Scope

This CPU-only analysis classifies the **existing**384 normal baseline outputs
per model. It does not execute a model, inspect the live adapter training run,
fit parameters, change a recipe, or advance a candidate. The baseline used
full-vocabulary greedy generation with at most12 new tokens. Target-removed
controls are outside this classification; their original false-recall counts
remain0/384 for both models.

Input: `reports/readapted_mk_baseline_v1.json`, SHA256
`a88f8fe75e82b992b8fc9cebeb6f681847e87aa36ea8bb1771edf979c47d551d`. The embedded frozen prompts, paired token digests,
parsed records, predicted first numbers and correctness flags were checked.
Every classification and all68 loss/12 gain IDs are retained in
[the JSON receipt](../reports/readapted_mk_baseline_error_types.json).

## Fixed five categories

Use the original matcher `(?<!\d)\d{6}(?!\d)` and its first match. Classify in
order: exact queried answer; another value present in the records; echoed
queried key; other six-digit number; no six-digit number. Key/value ranges are
disjoint in all384 prompts, so the value/key categories do not overlap.

| Model | Exact answer | Wrong existing value | Queried-key echo | Other6-digit | No6-digit |
|---|---:|---:|---:|---:|---:|
| Original FP16 | 167 | 213 | 0 | 4 | 0 |
| Current readapted | 111 | 242 | 0 | 31 | 0 |

Both models produced a standalone six-digit number on **all384 normal cases**.
The current model's decline is therefore not explained by missing six-digit
output under this fixed scorer. Of its273 errors,242 (88.64%) select a different
record's value;31 produce a number absent from the value list. The original
model has217 errors,213 of which (98.16%) select another record's value.

## N and template breakdown

Each row contains64 prompts. Templates are0=`key: value`,1=`Key ... has value ...`,
2=`key -> value`. Echo/no-number counts are zero in every cell. The JSON also
contains separate per-N and per-template aggregates.

| N | Template | Source exact | Source wrong value | Source other | Current exact | Current wrong value | Current other |
|---|---|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 26 | 38 | 0 | 18 | 46 | 0 |
| 16 | 1 | 30 | 34 | 0 | 17 | 47 | 0 |
| 16 | 2 | 64 | 0 | 0 | 55 | 9 | 0 |
| 64 | 0 | 3 | 58 | 3 | 1 | 45 | 18 |
| 64 | 1 | 7 | 56 | 1 | 3 | 53 | 8 |
| 64 | 2 | 37 | 27 | 0 | 17 | 42 | 5 |

At N16, source/current correct counts are120/192 and90/192. At N64 they are
47/192 and21/192. All31 current absent-value outputs occur at N64. These are
descriptive associations; the same cases were already observed development data.

## The68 losses and12 gains

Of68 source-correct/current-wrong cases, **65 (95.59%) became another existing
record's value**, and3 became an absent six-digit value. All12 gains corrected
a source answer that had selected another existing record's value.

| N | Template | Losses | Gains |
|---|---|---:|---:|
| 16 | 0 | 13 | 5 |
| 16 | 1 | 17 | 4 |
| 16 | 2 | 9 | 0 |
| 64 | 0 | 2 | 0 |
| 64 | 1 | 5 | 1 |
| 64 | 2 | 22 | 2 |

The complete transition counts are: exact→exact99, exact→wrong-value65,
exact→other3, wrong-value→exact12, wrong-value→wrong-value173,
wrong-value→other28, and other→wrong-value4. They sum to384 and reproduce
167→111 correct (68 losses minus12 gains =56 net losses).

## Interpretation and limits

The predominant visible failure is a **wrong association among supplied record
values**, with an additional increase in absent-value numbers at N64. This is
consistent with a binding-selection problem at the output, but does **not**
establish whether internal state retention, readout, cross-layer processing or
numeric decoding caused it. A generated number can match a prompt value without
proving the model internally copied that record. No internal activations or
causal interventions were measured here.

The first-match rule and generation budget stay fixed; later numbers are not
rescored. This analysis neither predicts Resurface success nor alters the frozen
training/evaluation protocol. The active adapter must still independently pass
recall controls and the same-enabled-model PPL nonregression requirement.
