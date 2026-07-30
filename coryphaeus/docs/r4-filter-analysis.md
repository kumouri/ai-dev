# r4 filter analysis — why 56% of steps bought nothing, and the v2 filter

**Sources:** `runs/calibrated-gsm8k.jsonl` (213 kept of 800 probed) × r4's own telemetry
(60 complete groups, 240 rollouts, 638 worker calls — the real pipeline, the real policy).
r4 is a labeled difficulty dataset we already paid for; this report mines it before any new spend.

## Hypothesis 1 — probe/training config mismatch: **minor, not the bug**

Probe and training workers both run at temperature 0.3 with `max_tokens=1024`. The prompts differ
structurally (solo `\boxed` prompt vs conductor-written subtasks) — irreducible, since the
conductor's decomposition can't be known pre-run. Config mismatch is not what happened.

## Hypothesis 2 — single-sample probes: **confirmed, exactly, 22 for 22**

Of r4's 22 unanimous-correct groups (the too-easy waste), **all 22** were questions the weak worker
had "failed" in its *single* calibration probe. That is the entire mechanism: one sample of a
worker that solves a question 70–90% of the time reads "fail" 10–30% of the time, the question gets
kept as "discriminative," and in training every rollout sails through it. The v1 filter was built
on a coin flipped once per question.

Empirical outcomes of the 60 attempted (all but one were probe-"disagree"):

| group outcome | n | share |
|---|---|---|
| **mixed (carries gradient)** | 33 | 55% |
| unanimous-correct | 22 | 37% |
| unanimous-wrong | 5 | 8% |

The v1 filter's yield (55% useful) barely beats what r3 got on raw GSM8K — the single-sample noise
ate most of its value. Honest verdict: v1 was directionally right and statistically underpowered.

## Hypothesis 3 — drift: **real but second-order tonight**

Parse-fail rate fell 32% → 20% within r4's 59 steps (first half vs second half of rollouts —
**visible learning**, the first direct evidence GRPO is bending the thing it should bend first).
As parsing improves, easier questions will migrate into unanimous-correct. Not worth mid-run
recalibration yet; the 20-step gate re-measures reality at every run start, which bounds the decay.

## The v2 filter

Use every piece of evidence we own, cheapest first:

1. **Measured-mixed (33 questions): keep.** Empirically contested under the real policy — gold.
2. **Measured-unanimous (27): drop.** 22 proved too easy, 5 too hard, by direct measurement.
3. **Unattempted (~153): re-probe with n=6 samples** on the weak worker at the training temperature,
   keep empirical pass-rate in **[0.17, 0.83]** (1–5 of 6). Six samples cost ~38 min at measured
   latency and de-quantize the estimate the single sample couldn't provide.

Expected v2 set: ~70–110 questions. Smaller than v1 — deliberately: 200 steps over ~90 contested
questions (~2 epochs) beats 200 steps where half are dead weight. Random sampling re-rolls groups
on revisit, so repeats still carry gradient.

**Gate (per the brief):** 20-step probe run; ship the remaining 180 only if zero-variance groups
come in **under 20%** (r4: 45–56% depending on the accounting). The probe's checkpoint resumes
into the full run, so a passed gate costs nothing.

## The step_time tail, decomposed — it was not the network, and not rambling

| suspect | evidence | verdict |
|---|---|---|
| remote tail | median 8.6s, p95 29.1s, max **71.2s**, retried 1% (60s cap + transient retry working) | tamed |
| rambling to the token cap | emission chars median 492, p95 890 — well under the cap; natural termination | no |
| **VRAM squeeze at train start** | steps 1–3 ≈ 20 min each while a daemon model sat on the card; healthy steps 60–110s | **this** |

The 11s→1282s spread is environmental, not architectural: the 1282s class is the squeezed phase.
Consequence for r5: **evict immediately before the train phase** (r4's chain evicted at chain
start, three hours before training began), and expect ~60–110s/step healthy — so a 6h deadline
buys ~180–200 steps when nothing goes wrong and self-sizes down when something does.

## Bonus finding

The 1.5B writes real multi-step workflows: n_steps distribution {2: 32, 3: 49, 4: 49, 5: 46} with
26% parse failures (falling). The prompted 4B averaged 1.1 steps. The trained policy is exploring
the part of the space the paper's recursion story actually lives in.
