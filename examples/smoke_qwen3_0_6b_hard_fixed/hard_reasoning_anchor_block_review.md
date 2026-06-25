# Hard Reasoning RCPC Candidate Review

This document checks two stages of the current RCPC design on harder reasoning tasks:

1. Candidate micro-action anchoring by robust-normalized action entropy.
2. Local contiguous block aggregation around entropy peaks.

The model used here is `Qwen/Qwen3-0.6B`; correctness is not the target. We inspect whether high-entropy spans and peak-centered blocks look like plausible reasoning decision points.

## Sample 1: `aime24-76` (AIME24)

**Problem**

```text
Let $\triangle ABC$ have circumcenter $O$ and incenter $I$ with $\overline{IA}\perp\overline{OI}$, circumradius $13$, and inradius $6$. Find $AB\cdot AC$.
```

**Gold answer:** `468`

**Original Reasoning Text**

```text
<think>
Okay, so I need to find AB multiplied by AC in triangle ABC where the circumradius is 13, inradius is 6, and the incenter I is such that IA is perpendicular to OI. Hmm, okay. Let me start by recalling some formulas related to triangle centers and their relationships.

First, the circumradius R is given as 13. The inradius r is 6. The formula for the inradius is r = 4R sin(A/2) sin(B/2) sin(C/2). Wait, is that right? Let me confirm. Alternatively, r = (a + b - c)/2 * something? No, maybe I should recall the formula for inradius in terms of area and semiperimeter. Yes, r = Δ/s, where Δ is the area and s is the semiperimeter. Also, the formula for the circumradius is R = abc/(4
```

**Top-12 Candidate Action Spans**

1. `robust_E=1.310` · `raw_E=0.897` · `tokens=4` · `action_id=9`
   - Let me confirm.
2. `robust_E=1.036` · `raw_E=0.825` · `tokens=15` · `action_id=11`
   - r = (a + b - c)/2 * something? No,
3. `robust_E=0.964` · `raw_E=0.807` · `tokens=24` · `action_id=12`
   - maybe I should recall the formula for inradius in terms of area and semiperimeter. Yes, r = Δ/s
4. `robust_E=0.687` · `raw_E=0.735` · `tokens=12` · `action_id=15`
   - the formula for the circumradius is R = abc/(4
5. `robust_E=0.603` · `raw_E=0.713` · `tokens=22` · `action_id=6`
   - 6. The formula for the inradius is r = 4R sin(A/2) sin(B/
6. `robust_E=0.115` · `raw_E=0.586` · `tokens=24` · `action_id=1`
   - so I need to find AB multiplied by AC in triangle ABC where the circumradius is 13, inradius is
7. `robust_E=0.091` · `raw_E=0.579` · `tokens=2` · `action_id=10`
   - Alternatively,
8. `robust_E=0.037` · `raw_E=0.565` · `tokens=22` · `action_id=2`
   - 6, and the incenter I is such that IA is perpendicular to OI. Hmm, okay.
9. `robust_E=-0.037` · `raw_E=0.546` · `tokens=17` · `action_id=3`
   - Let me start by recalling some formulas related to triangle centers and their relationships. /  / First,
10. `robust_E=-0.134` · `raw_E=0.521` · `tokens=9` · `action_id=4`
   - the circumradius R is given as 1
11. `robust_E=-1.310` · `raw_E=0.214` · `tokens=7` · `action_id=5`
   - 3. The inradius r is
12. `robust_E=-1.352` · `raw_E=0.204` · `tokens=9` · `action_id=7`
   - 2) sin(C/2). Wait,

**Peak-Centered Candidate Blocks**

1. `anchor_action=9` · `actions=[9, 10, 11, 12]` · `anchor_robust_E=1.310` · `mean_robust_E=0.850` · `tokens=45`
   - Let me confirm. Alternatively, r = (a + b - c)/2 * something? No, maybe I should recall the formula for inradius in terms of area and semiperimeter. Yes, r = Δ/s
2. `anchor_action=11` · `actions=[9, 10, 11, 12]` · `anchor_robust_E=1.036` · `mean_robust_E=0.850` · `tokens=45`
   - Let me confirm. Alternatively, r = (a + b - c)/2 * something? No, maybe I should recall the formula for inradius in terms of area and semiperimeter. Yes, r = Δ/s
3. `anchor_action=15` · `actions=[15]` · `anchor_robust_E=0.687` · `mean_robust_E=0.687` · `tokens=12`
   - the formula for the circumradius is R = abc/(4
4. `anchor_action=6` · `actions=[6]` · `anchor_robust_E=0.603` · `mean_robust_E=0.603` · `tokens=22`
   - 6. The formula for the inradius is r = 4R sin(A/2) sin(B/

