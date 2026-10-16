# Stage 1 analysis plan (frozen before the analysis was executed)

Written 2026-10-16T13:34:50+08:00. HEAD `4239b15eeb861e644c4d10da0ea8b928888e866c`. Script `analyze_horizon_oracle.py` sha256 `a94148fd78e76d959724302aa9027cca88aa646e2a28b844839efaabf063f8d2`.
No Phase-B per-scene cross-arm statistic had been computed before this file was written; the
aggregate per-arm rates and the AAC-vs-fixed paired table in `results/aac/phaseB_coast/report.md` were known.

1. **Gate (from the research brief, unchanged).** Headroom = per-scene oracle over h ∈ {2,4,8,12,16}
   minus the strongest single fixed h, on scenes matched by task + episode seed + identical
   scene_hash and model_hash in all five arms. < 3 pp → LOW; 3–5 pp → BORDERLINE; ≥ 5 pp with
   meaningful within-task heterogeneity → HIGH. The raw gate is reported exactly as specified.
2. **Validity rule added before execution (not a threshold change).** Each arm is one stochastic rollout
   per scene (running RNG). A per-scene "any arm succeeds" oracle therefore includes rollout noise that no
   router can harvest. The temporal intervention is called HIGH PRIORITY only if the horizon oracle also
   beats a same-horizon replicate oracle: on the 4 tasks × 50 seeds where Phase A-U re-ran fixed8/fixed16/AAC
   with the same protocol, the 2-arm oracle gain of {fixed16, fixed8} must exceed the 2-arm gain of
   {fixed16 run B, fixed16 run A-U}, with the task-stratified bootstrap 95% CI of the difference excluding 0.
   If not, the raw verdict is reported together with "not distinguishable from rollout noise".
3. Within-task heterogeneity = scenes where the task's best h fails but another h succeeds; plus headroom of
   the scene oracle over the per-task best h (in-sample and even/odd cross-fitted).
4. CIs: task-stratified bootstrap (4000 resamples, seed 0). Paired tests: exact McNemar.
5. Exceptions: aborted infrastructure attempts (attempts.log, task_logs tracebacks) are separated from
   completed episodes; only completed episodes enter any statistic.
