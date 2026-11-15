# RunPod cost record

No project cloud resource is active and no project GPU experiment has run.

Attempt 1 provisioned an A40 job, but archive extraction failed when NFS root-squash rejected ownership restoration. Both created resources were deleted after the failure; no experiment outputs were lost. The $0.70 reservation remains unsettled because the provider request was failed/ambiguous rather than reconciled. A local restricted-Podman proof validated the `tar --no-same-owner` preparation fix for attempt 2, which has not provisioned a resource.

The append-only ledger is `runs/runpod_cost_ledger.jsonl`. It retains the $0.70 unsettled reservation and observed historical November storage billing of **$0.6300000152550638**, for total exposure of **$1.3300000152550638**. Historical storage is not a new NeuroForge GPU charge, but is conservatively chargeable against the study ceiling.

| Budget | Ceiling USD | Remaining USD |
|---|---:|---:|
| Operational | 7.00 | 5.6699999847449362 |
| Absolute | 10.00 | 8.6699999847449362 |

The protected $3 reserve is untouched. A single projected job above $1.50, crossing the operational ceiling, or spending any reserve requires explicit approval. Funding does not change the $7 operational or $10 absolute ceilings.

The ledger uses Decimal arithmetic, locked check-and-append reservations, actual-billing reconciliation and explicit release. Hard-cap preflights reject underquoted compute, stale ledger exposure, mismatched projected totals, and estimates beyond the hard runtime. A failed or ambiguous provider request retains its reservation until reconciled or explicitly released. Historical entries and observed billing are never erased.

Watchdog CLI delete signatures were verified with help only. The attempt-1 resources were deleted after the NFS failure. Watchdog and evaluator boundary proofs must pass before billable execution; unit tests are not proof of provider cleanup.

## Watchdog verification (2026-11-15)

The transient user-systemd timer mechanism was proven with a canary before provisioning job 07: a canary timer appended its marker file and `require_canary` observed it. Every subsequent provisioning armed a pod-first timer and verified it with `verify_timer_armed`, which now parses `list-timers` (transient timers expose no `NextElapseUSec`, and the listing renders local time, not UTC). Job 04 demonstrated the failure mode this guards: its 60-minute timer produced no journal entries and the pod survived ~20 minutes past the cap, so the orchestrator terminated it manually. Later jobs use the verified flow. No timer is trusted without the list-timers schedule proof.
