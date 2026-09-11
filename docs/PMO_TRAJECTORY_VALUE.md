# Target-free trajectory-value development probe

User authorization (2026-09-11): implement the proposed small, answer-informed
trajectory-learning and ordinary-proposal comparison. This is a new development
experiment, not a change to any prior contract or reference-model training gate.

Problem: identify whether ordinary COMPOSE proposals expose useful edits and
whether achieved continuation returns improve selection. Output: a learned,
budget-conditioned value potential and retained executable option trajectories.
Claim tested: short-prefix task-score improvement on seven development starts,
not unknown-winner recovery or benchmark superiority. The five public-target
compiler routes are training data. Their endpoints are not authenticated IVG
optimization trajectories. Two new perturbations are nearby-development probes,
not independent scaffold or task holdouts.

Frozen substrate: selected R_theta, existing region distribution, applicable
balanced macro/program menu, generic channel, exact executor, 1..40 active atoms,
48 persistent slots, charge-preserving achiral representation. No new ring
templates, reference training, docking or prescreen. Known endpoint/alignment,
pending-retirement masks and target-based proposal filters are absent from
evaluation proposal and decision interfaces. Controller features are fixed
Morgan graph features plus structural counts, not target similarity.

## Sequence and decisions fixed before outputs

1. Reuse the five exact saved demonstrations and all compatible existing labels.
   At each original start and its eight-edits-before-end state, audit the next
   demonstrated canonical successor against the FULL production marked law,
   aggregated by canonical product. Also measure generic-channel exposure across
   the unchanged selectable regions. A missing teacher is reported, never injected.
2. From the late states of original_root_0 and original_root_1, draw one valid
   non-self ordinary successor outside the training-state canonical identities.
   Freeze these two perturbations before fitting. No selection by score or target
   distance. Original starts plus these two states are the seven probes.
3. Score unlabelled demonstration states using the exact PMO perindopril-MPO
   oracle. Store all labels. For each state and b in {1,3,6,11,32,64}, back up the
   best score actually seen within b subsequent primitive steps, including stop.
   These are achieved-path returns, not optimal values or reference expectations.
   No unobserved alternative is assigned zero or labelled unsuccessful.
4. Fit a 128/128 ReLU sigmoid value network for exactly 600 Adam updates at .003,
   batch 64, seed 20260920, CPU float32, one thread. Half each batch samples known
   immediate scores at b=0; half samples a route uniformly then a state and budget
   uniformly. No early stopping or selection using fresh outcomes. Save model,
   optimizer, RNG, preparation hash and configuration. No uncertainty claim.
5. Generate four ordinary complete options per probe, using the existing
   WHERE/WHAT/HOW generator unchanged. Lock them, then score. Compare uniform,
   immediate score, and learned eleven-step value (floored by known stop score).
   Canonical duplicate draws aggregate their reference multiplicity. Failed
   options remain reported but cannot be selected as molecules. Zero viable
   proposals is an explicit abstention. The two guided arms use the existing
   kappa=1, epsilon=.1 tilt with the same seeded uniform variate for selection.
6. Generate four further ordinary options from each distinct selected state.
   Share identical selected states across arms. Retain and return the best scored
   completed continuation or incumbent, preserving its complete witnessed chain.
   Do not throw away the best continuation and evaluate a fresh random one.
   No refitting occurs after first-stage labels. This tests an in-loop first
   option decision; it does not install a new primitive HOW policy.

Primary readout: per-source best actual score after the selected two-option
prefix, all arms and abstentions. Report proposal exposure, first-choice changes,
predicted vs achieved continuation value, diversity, topology, intended/realized
scale, wall time, executor/law counts, and separate training/evaluation calls.
This short probe does not exhaust the 38..55-step demonstrated source routes;
failure to reach their endpoint is not a reachability conclusion.

## Compute and restart

At most 400 new physical PMO calls total, including training-state, perturbation,
ranking and final labels; none hidden as cheap preprocessing. No docking.
Five coverage/preparation workers, seven first-option workers and at most 21
continuation workers, at most 20 simultaneous CPU containers. Prior measured
four-draw workers took up to 97 seconds; allow 1200 seconds per worker and 1800
seconds for the driver as operational aborts, never silently compare incomplete
arms as matched. Expected experiment wall time 5..15 minutes excluding deployment,
based on that prior run, not guaranteed. Estimated remote cost $0.5..5, recorded
as an estimate. Cache every completed worker and exact law; heartbeat every 30
seconds. Source must be clean and committed; preflight, deploy, then durable spawn.

Prepare source/role census locally without scoring. Remote CPU preparation
publishes an immutable labelled panel before the separate fit stage consumes it;
no GPU is allocated. Reuse the already qualified inference package. Full-suite
release qualification remains pending; run focused tests and lint for this probe.
