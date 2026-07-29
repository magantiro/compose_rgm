#!/usr/bin/env bash
# Move the RingCore-V1 scientific corpus between Modal workspaces.
#
# Modal volumes are workspace-scoped, so a single process cannot mount both sides: the transfer is
# download -> upload, driven by a local staging directory.
#
# Only the artifacts the scientific run actually reads are moved. Deliberately NOT moved:
#   mmp_packed_v1/_raw/                       ~400 MB, intermediate, regenerable during packing
#   edit_mining_full_broad_40/edit_pool_full.jsonl  410 MB, unread now that the MMP layer is packed
#   ~40 historical run directories            dead
#
# Every artifact is content-hashed (shard content_sha256 -> overlay binding -> manifest checksum), so a
# partial or corrupted copy fails LOUDLY at load rather than silently training on damaged data. The
# verify step below re-derives the manifest identity in the destination and compares.
#
# Usage:
#   scripts/transfer_workspace.sh stage    <src-profile>              # download from the source workspace
#   scripts/transfer_workspace.sh upload   <dst-profile>              # create volumes + upload
#   scripts/transfer_workspace.sh verify   <dst-profile>              # re-check counts and hashes
set -euo pipefail

STAGE_DIR="${STAGE_DIR:?set STAGE_DIR to the local staging directory}"
ARTIFACTS_VOLUME="compose-v4-artifacts"
CORPUS_VOLUME="guacamol"

BASE_B_RUN="compose-v4-stage3-flexible-graft-3k-1ac6f19-v1"
CORPUS_FILES=(guacamol_subset_500000_seed0.smiles guacamol_heldout_val_5000_seed0.smiles)
TOP_LEVEL_FILES=(UNIFIED_PACKED_MANIFEST.json REPRESENTABILITY_OVERLAY.json)

action="${1:?stage|upload|verify}"
profile="${2:?modal profile name}"

# ---- expected identities. A mismatch after transfer means the copy is not the audited corpus. --------
EXPECT_MANIFEST_CHECKSUM="5c5c254e1054c081"
EXPECT_OVERLAY_CHECKSUM="32372dc5d73139a7"
EXPECT_SCHEDULER_HASH="dafd4b5092414394"

stage() {
  mkdir -p "$STAGE_DIR"
  modal profile activate "$profile"
  modal volume get "$ARTIFACTS_VOLUME" edit_precompile_v1 "$STAGE_DIR/" --force
  modal volume get "$ARTIFACTS_VOLUME" edit_packed_v1 "$STAGE_DIR/" --force
  mkdir -p "$STAGE_DIR/mmp_packed_v1"
  for partition in train validation test; do
    modal volume get "$ARTIFACTS_VOLUME" "mmp_packed_v1/$partition" "$STAGE_DIR/mmp_packed_v1/" --force
  done
  for file in MMP_PACK_COMPLETE.json partition_manifest.json; do
    modal volume get "$ARTIFACTS_VOLUME" "mmp_packed_v1/$file" \
      "$STAGE_DIR/mmp_packed_v1/$file" --force
  done
  for file in "${TOP_LEVEL_FILES[@]}"; do
    modal volume get "$ARTIFACTS_VOLUME" "$file" "$STAGE_DIR/$file" --force
  done
  mkdir -p "$STAGE_DIR/$BASE_B_RUN"
  for file in checkpoint.best_so_far.pt manifest.json; do
    modal volume get "$ARTIFACTS_VOLUME" "$BASE_B_RUN/$file" "$STAGE_DIR/$BASE_B_RUN/$file" --force
  done
  mkdir -p "$STAGE_DIR/guacamol"
  for file in "${CORPUS_FILES[@]}"; do
    modal volume get "$CORPUS_VOLUME" "$file" "$STAGE_DIR/guacamol/$file" --force
  done
  echo "staged $(find "$STAGE_DIR" -type f | wc -l) files, $(du -sh "$STAGE_DIR" | cut -f1)"
}

upload() {
  modal profile activate "$profile"
  modal volume create "$ARTIFACTS_VOLUME" 2>/dev/null || true
  modal volume create "$CORPUS_VOLUME" 2>/dev/null || true
  for dir in edit_precompile_v1 edit_packed_v1 mmp_packed_v1 "$BASE_B_RUN"; do
    [ -d "$STAGE_DIR/$dir" ] && modal volume put "$ARTIFACTS_VOLUME" "$STAGE_DIR/$dir" "$dir" --force
  done
  for file in "${TOP_LEVEL_FILES[@]}"; do
    [ -f "$STAGE_DIR/$file" ] && modal volume put "$ARTIFACTS_VOLUME" "$STAGE_DIR/$file" "$file" --force
  done
  for file in "${CORPUS_FILES[@]}"; do
    [ -f "$STAGE_DIR/guacamol/$file" ] && \
      modal volume put "$CORPUS_VOLUME" "$STAGE_DIR/guacamol/$file" "$file" --force
  done
  echo "upload complete"
}

verify() {
  modal profile activate "$profile"
  tmp="$(mktemp -d)"
  modal volume get "$ARTIFACTS_VOLUME" UNIFIED_PACKED_MANIFEST.json "$tmp/m.json" --force
  modal volume get "$ARTIFACTS_VOLUME" REPRESENTABILITY_OVERLAY.json "$tmp/o.json" --force
  python3 - "$tmp" "$EXPECT_MANIFEST_CHECKSUM" "$EXPECT_OVERLAY_CHECKSUM" <<'PY'
import json, sys
tmp, want_manifest, want_overlay = sys.argv[1], sys.argv[2], sys.argv[3]
manifest = json.load(open(f"{tmp}/m.json"))
overlay = json.load(open(f"{tmp}/o.json"))
checks = {
    "manifest_checksum": (manifest.get("manifest_checksum"), want_manifest),
    "overlay_effective_corpus_checksum": (overlay.get("effective_corpus_checksum"), want_overlay),
    "SCIENTIFIC_TRAINING_CONTRACT": (
        (manifest.get("contract_levels") or {}).get("SCIENTIFIC_TRAINING_CONTRACT"), "PASS"),
    "total_shards": (manifest.get("totals", {}).get("shards"), 63),
    "total_entries": (manifest.get("totals", {}).get("entries"), 725671),
    "exclusions": (len(overlay.get("exclusions", [])), 3),
}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
for key, (got, want) in checks.items():
    print(f"  {'OK ' if got == want else 'BAD'}  {key}: {got!r} (want {want!r})")
if bad:
    raise SystemExit(f"TRANSFER VERIFY FAILED: {bad}")
print("TRANSFER VERIFY: PASS -- the destination holds the audited corpus")
PY
  rm -rf "$tmp"
}

case "$action" in
  stage) stage ;;
  upload) upload ;;
  verify) verify ;;
  *) echo "usage: $0 stage|upload|verify <profile>" >&2; exit 2 ;;
esac
