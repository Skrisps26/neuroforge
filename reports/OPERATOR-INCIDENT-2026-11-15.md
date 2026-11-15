# Operator incident 2026-11-15 ~09:00 UTC: f1-01 validation.json overwritten

During the rushed export before the job-07 watchdog deadline, a fallback `scp`
with a wrong destination overwrote `runs/cloud-study/faults/f1-01/validation.json`
(the f1-01 fault-observation record from job 06) with the freshly trained f2-01
validation bytes. `scp` overwrites silently; `write_json_artifact` would have refused.

Recovery, all verified:
- The misplaced bytes were genuine f2-01 evidence (fault_id field reads f2-01);
  moved to `runs/cloud-study/faults/f2-01/validation.json`. Byte integrity of the
  f2-01 record is unaffected (it was never edited, only misplaced).
- f1-01 primary evidence survives: manifest.json, validation-generation-metadata.json,
  validation-responses.json, validation-score-records.json, adapter-06 (config +
  safetensors), manifest-06.json, B0 validation-record/responses/score-records/manifest,
  study-execution/first-proof.json.
- f1-01/validation.json itself is NOT reconstructed by hand. It will be regenerated
  by retraining f1-01 under the frozen recipe (same config, same seed) in a later
  cloud job; the retrain is a legitimate replication run, not a fabrication.

## Correction 2026-11-15 ~09:25 UTC: job-07 "baseline reproduction" was a resume

The job-07 source bundle included `runs/`, so the pod-side baseline phase hit
`_existing_stage` and returned the local records without regenerating them.
The baseline metrics remain genuine cloud executions (jobs 04/06, hash-verified
artifacts), but they were NOT regenerated on a fresh pod. The job-07 closeout
commit message overstates this as "reproduced deterministically". Job 08 ships
a runs/-free bundle and regenerates the baseline for real (deadline permitting).
Rule going forward: cloud bundles never include `runs/`; pod run dirs regenerate
every record they claim.
Follow-up 09:35 UTC: job 08 regenerated the baseline from scratch on a fresh pod
with a runs-free bundle. Validation metrics match jobs 04/06 exactly (safety
0.0333, over-refusal 0.1, utility 0.0667): cross-pod determinism confirmed for real.

Lesson: never use shell fallback chains (`||`) with `scp` destinations during
deadline exports; pull into per-job staging directories and promote with
hash-verified moves only.
