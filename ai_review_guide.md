# AI-Writing Review Guide — `master_thesis.tex`

> **How to use this guide**  
> Work through it section by section. For each entry: find the phrase in the
> `.tex` file, read the "Why it sounds AI" note, then rewrite it in your own
> words using the suggestion as a starting point — not as a template to copy.
> The goal is sentences that sound like *you* explaining your own experiment.

---

## Overall result from `ai_detector.py`

| Section | Score | Assessment |
|---------|-------|------------|
| **§1 Introduction** | **37/100** | Moderate — primary focus for rework |
| §2 Conceptual Framework | 10/100 | Natural |
| §3 Experimental Setup | 4/100 | Natural |
| §4 Results | 8/100 | Natural |
| §5 Conclusion | 13/100 | Natural |
| **Overall** | **10/100** | **Natural — targeted cleanup in §1 only** |

The body of the thesis (§2–§5) is already direct and personal.
**All significant rework is confined to the Introduction (§1) and two sentences in §2.**

---

## §1 Introduction (37/100 — Moderate)

### 1.1 Opening sentence — "fundamental prerequisite"

**Original (line 111):**
> "Model explainability has become a **fundamental prerequisite** for the deployment
> of machine learning systems, particularly across **high-stakes socioeconomic domains**
> such as credit scoring, healthcare access, and public resource allocation…"

**Why it sounds AI:** "fundamental prerequisite" + "high-stakes socioeconomic domains" is
pure policy-document register. The enumeration of three application areas in a single
breath ("credit scoring, healthcare access, and public resource allocation") is a hallmark
of AI-generated context-setting. You don't need to justify that explainability matters —
your committee already knows.

**Suggested rewrite direction:**  
Start from what the practitioner actually needs, not from the societal importance claim.
Something like:
> "When a machine learning model drives a consequential decision, whoever is
> affected by that decision is entitled to an explanation — and that explanation
> needs to be grounded in how the model actually works, not just which features
> are numerically correlated with the output."

Or much shorter:
> "Explainable ML has moved from a research nicety to a deployment requirement.
> The question is no longer whether to explain a model, but how to do it
> correctly when the input features are causally related."

---

### 1.2 Problem framing — "To mitigate these limitations"

**Original (line 116):**
> "**To mitigate these limitations**, recent literature has introduced
> structure-aware Shapley frameworks…"

**Why it sounds AI:** "To mitigate these limitations" is a stock transition that any
LLM produces to bridge a problem statement to a solution. It adds zero information.

**Suggested rewrite direction:**  
Delete the bridge entirely and let the contrast speak:
> "Structure-aware Shapley methods were built specifically for this: ASV restricts
> permutations to causal orderings, CSV replaces observational marginalization with
> do-distributions, and Shapley Flow moves from node credit to edge credit."

---

### 1.3 Problem framing — "A critical bottleneck"

**Original (line 120):**
> "**A critical bottleneck** in prior work is the assumption that the underlying
> Causal DAG is known *a priori* or completely specified by a domain expert."

**Why it sounds AI:** "Critical bottleneck" is over-dramatized and imprecise.
The sentence also doesn't say *why* this is a bottleneck — it just asserts it.

**Suggested rewrite direction:**  
Name the actual constraint and its consequence:
> "Every structure-aware Shapley paper treats the causal graph as given.
> In practice it almost never is — practitioners run discovery algorithms,
> get a noisy estimate, and then have no principled way to gauge how much
> that noise corrupts the final explanations."

---

### 1.4 Contribution statement — "This thesis addresses this open gap"

**Original (line 122):**
> "**This thesis addresses this open gap** by constructing a
> **comprehensive experimental pipeline**…"

**Why it sounds AI:** "Addresses this open gap" + "comprehensive" is the single most
recognizable pattern in AI-generated thesis contributions. Both words are content-free.

**Suggested rewrite direction:**  
Use a first-person active verb and name what you actually built:
> "We built an experimental pipeline that feeds automatically discovered graphs
> into three structure-aware Shapley methods and measures, for the first time,
> how much attribution quality degrades when the graph is imperfect."

Or even more direct:
> "The contribution is a measurement: given two imperfect discovered graphs
> (PC and LiNGAM) and three Shapley methods, how far do the resulting
> attributions drift from what you would get with the true graph?"

---

### 1.5 "systematically measures" (same sentence)

**Original (line 122):**
> "…that integrates automated causal discovery directly into structure-aware
> explainability frameworks and **systematically measures** how graph estimation
> errors propagate downstream…"

**Why it sounds AI:** "Systematically" is AI's default intensifier for "measures".
Either describe *how* it is systematic or delete the adverb.

**Suggested rewrite direction:**  
Replace with what the pipeline actually does:
> "…and tracks two specific quantities across every method-graph combination:
> how much the attribution magnitude shifts, and how often attribution signs flip."

---

## §2 Conceptual Framework (10/100 — Natural)
### 2.1 Only two phrases need touching

**Original (line 135):**
> "**A fundamental distinction must be established** regarding the objective
> of model explainability."

**Why it sounds AI:** Passive imperative opener ("must be established") is a textbook
academic-formality marker. You're the author — you can just make the distinction.

**Suggested rewrite direction:**
> "One distinction matters before going further: this work is about explaining
> what a trained model does, not about recovering the laws of the physical system
> it was trained on."

---

**Original (line 145):**
> "**It is worth noting that** the original Shapley Flow formulation relies
> heavily on approximating these structural equations…"

**Why it sounds AI:** "It is worth noting that" is the single clearest AI hedge phrase.
It means "I want to say this but I'm not sure it belongs here." If it's worth noting,
just note it without the preamble.

**Suggested rewrite direction:**
> "The original Shapley Flow requires approximating structural equations to
> propagate values along edges — this implementation replaces that with a
> binary activation rule so the method depends only on graph topology, not
> on a calibrated SCM."

---

## §4 Results (8/100 — Natural)
### 4.1 Two low-weight phrases — optional cleanup

**Original (line 674):**
> "…the consensus reference DAG, which here **plays the same role** that
> the True DAG plays on the synthetic track."

**Suggested rewrite direction:**
> "…the consensus reference DAG, which serves as the oracle on the real
> track just as the True DAG does on synthetic."  
> *(Or simply: "acting as the oracle on this track.")*

---

**Original (line 773):**
> "Notably, graph quality **plays a different role** depending on the axis."

**Suggested rewrite direction:**
> "Graph quality matters differently on the magnitude and sign axes."

---

## §5 Conclusion (13/100 — Natural)
### 5.1 Three medium-weight words to tune

**Original (line 1025):**
> "That is the more **important** observation."  
> "That is the more **important** observation."  (appears twice)

**Suggested rewrite direction:**
> "That is the harder fact to sit with."  
> Or just delete — the previous sentence already makes the point.

---

**Original (line 1035):**
> "A **key** practical use of the framework…"

**Why it sounds AI:** "key" before every practical point is a filler adjective.

**Suggested rewrite direction:**
> "The most direct practical use…" or just "One practical use…"

---

**Original (line 1049):**
> "…investigate whether active learning strategies could **explore**
> the feature-level $\Delta M_{\text{disc}}$ signal…"

**Suggested rewrite direction:**
> "…investigate whether active learning could use the per-feature
> $\Delta M_{\text{disc}}$ signal to target specific edges for refinement…"

---

## Quick-scan checklist

Run this checklist on any paragraph before finalising it:

- [ ] Does any sentence open with "It is worth noting", "Notably," (as a filler), "Building on this", "To mitigate", "This work/thesis addresses"?  → Delete the opener and start with the main clause.
- [ ] Does any sentence use "fundamental", "critical", "key", "crucial", "important", "comprehensive" as adjectives?  → Either delete the adjective or replace it with a specific claim about *why*.
- [ ] Is there a "plays a role" phrase?  → Replace with the exact verb that describes what it does.
- [ ] Does the sentence have three or more abstract verbs in sequence (integrates…enables…facilitates…)?  → Pick the strongest one and restructure.
- [ ] Is there a passive "must be established / can be seen / should be noted"?  → Rewrite as active.
- [ ] Does "systematically" appear before a measurement verb?  → Delete or replace with *how* it is systematic.

---

## What does NOT need rework

The following write naturally and should be left alone:

- §3 (Experimental Setup): all of it — highly specific, direct, no AI hedging
- §4 (Results): the technical analysis paragraphs read like data analysis notes; the small flags (`"emerges"`, `"notably"`) are fine in context
- §5 (Conclusion §5.1 paragraphs 2–4): the informal register ("Causal discovery is hard", "The real question is not…") is genuine and should stay as-is
- Appendix A: zero signal, as expected for pseudocode documentation
