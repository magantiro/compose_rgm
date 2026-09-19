# Integrated route/FiberControl JAK2 result

The frozen three-cell controller completed all 147 charged calls with zero
failed dockings. At 49 calls per cell, COMPOSE reached -9.6, -11.2 and -10.6
on JAK2 seeds 0, 1 and 2. The corresponding reported InVirtuoGen delta=0.6
means are -9.7, -10.4 and -10.3. Thus two cells beat the reported means and
the third finished 0.1 above its mean.

The seed-1 winner was an anchored-replacement proposal selected by the online
FiberControl model from a -9.8 parent. The seed-2 winner was a shallow proposal
selected by FiberControl from a -10.1 parent. The seed-0 incumbent came from
shallow exploration, not the value model. No final incumbent was uniquely
route-expert-derived. A route-plus-shallow endpoint did reach -10.1 on seed 2,
showing useful overlap, but the route prior has not yet earned causal or winner
credit.

This is winner-informed development evidence under one docking seed. The
-11.2 and -10.6 endpoints require fresh-seed replication before they are used
as robust headline values. The run is not held-out evidence, a full T4 result,
or a same-generator causal ablation of FiberControl.

The authoritative artifacts remain on Modal volume
`compose-t4-integrated-route-fiber-v1` under run ID
`2f223bd1210f3da8e5adc99e93679b1303bfb2922c0509f570b9b213bb96d066`.
Physical hashes and per-round trajectories are recorded in `final_result.json`.
