# Five selected single-day inverse timing

Frozen design, 2026-11-05. Each date in 1, 113, 225, 337, and 450 is a
separate 10 MHz single-day workload. Day 450 is one selected date. Reuse the
ten frozen G1 consumed-n visibility segments for each date, giving 1,954,680
half-baseline samples. Each inverse computes all 201,326,592 NEST output
pixels at Nside 4096. The binaries take one explicit GPU UUID. They perform
no new visibility generation and write no full maps.

B is the corrected mature legacy direct 3D fast path (raw DCF-weighted
numerator). C is the original representative STS with grid phase and its
usual cell weighting and normalization. The shared visibility inputs match.
The output response definitions differ. Report inverse-loop cost ratios
without assuming matched image accuracy. The archived DCF table is a
fixed input, with no new 450-day computation.

Per day and method, run a complete functional gate, one separate warm-up, and
four formal paired blocks in the order BC, CB, CB, BC. Each warm-up, gate,
and formal run uses the same output domain. Preserve formal observations
in execution order. Recheck before/after the target UUID for foreign compute
contexts. Retain host/node conditions and the actual device identity.
Use the inverse wall-time boundaries of the 2026-11-01 single-4090
comparison. Cache/decode, initialization, finalization and audits remain
separate. B's CUDA event span includes planner dependencies. Sample the
same 8192 frozen IDs and scan full arrays for finiteness and identity
outside the timed inverse loop.

Numerical gates apply to every run. B must agree with its accepted
fixed-DCF raw sample numerator with relative L2 and max absolute difference
over reference peak at most 0.001. C must match the accepted production
sample numerator bit for bit and its sample W exactly. Full B raw N and C
N/W/C must contain all output pixels, pass finiteness checks, and record
their full-array hashes. Hashes are audit records. Atomic B is not held
to a bitwise repeatability requirement. A gate failure stops the batch
and preserves its attempt and raw log. No fallback output size or tolerance.

The runner verifies source/binary, all 50 cache identities, reference IDs,
device identity, and free disk before work. Boundary checks run before and
after each subprocess. Device clock locking and node exclusivity are not
assumed. Case directories commit by atomic rename following their gates.
On interruption, successful committed cases are verified and kept. A
partly completed formal day receives separately labeled restart warm-ups
before continuing. An uncommitted attempt is preserved for review.
