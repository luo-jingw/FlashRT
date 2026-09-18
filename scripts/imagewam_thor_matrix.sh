#!/usr/bin/env bash
# ImageWAM Thor configuration matrix: one fixed workload, one row per
# configuration, the table is built from the logs.
#
# Two modes, selected by whether PROFILES is set:
#
#   flag rows (PROFILES unset, the default): one row per switch set in
#   ROWS, each relative to the precision under test.
#     default          served defaults
#     vae              + native VAE encoder captured into the main graph
#     vae_trim         + text_trim
#     vae_trim_fa4bb   + FA4 on the backbone attention
#     stack            + FA4 on the ActionDiT (mot) attention too
#     stack_no_vae     stack without the native VAE   (leave-one-out)
#     stack_no_trim    stack without text_trim        (leave-one-out)
#   Leaving FA4 out of the stack is vae_trim; leaving only the mot site out
#   is vae_trim_fa4bb, so those two are ladder rows and not repeated.
#
#   profile rows (PROFILES="default fast"): one row per named profile, row
#   name `profile_<name>`, exporting PROFILE=<name> to the compare script.
#   The row is the named profile itself (its switches live in
#   config_resolver.PROFILES), not a hand-written switch set; ROWS is not
#   read in this mode. PRECS still selects every row's precision: a named
#   profile's own precision is nvfp4, so PRECS unset (nvfp4 below) is the
#   profile's own configuration and any other value overrides it. Profile
#   rows select no calibration file -- the trim/non-trim choice belongs to
#   the profile -- so a static-FP8 precision in this mode needs CALIBRATION
#   in the caller's environment.
#
# Row names and per-row logs do not collide between the two modes: a profile
# row is `profile_<name>` and its log is
# matrix_<suite>_<prec>_profile_<name>.log, and TAG carries the mode and the
# profile names, so a profile run and a flag run into the same OUT cannot
# overwrite each other's table or logs.
#
# Same environment as scripts/imagewam_thor_validation.sh (CKPT_PATH,
# FLUX2_*, QWEN3_MODEL_SPEC, DATA_ROOT, PYTHONPATH, BUNDLE, OUT).
# Optional: SUITE (libero_spatial), PRECS ("nvfp4"), ROWS (all seven),
# PROFILES (unset: the flag rows above), FULL (10 tasks x frames 0,60 x
# seeds 0,1), TAG (names the output files; default: the precisions joined by
# "+", and in profile mode those plus "profiles" and the profile names).
# Two runs into one OUT need different SUITE, PRECS, PROFILES or TAG,
# otherwise the second overwrites the first.
#
#   OUT=$HOME/thor_val/matrix BUNDLE=$HOME/thor_bundle PRECS="nvfp4 fp8_static_cutlass" \
#     bash scripts/imagewam_thor_matrix.sh
#   OUT=$HOME/thor_val/matrix_profile BUNDLE=$HOME/thor_bundle PRECS=nvfp4 \
#     PROFILES="default fast" bash scripts/imagewam_thor_matrix.sh
#
# The latency column is the e2e script's own steady-state P50, taken on a
# shared machine: compare rows of one run with each other, and record
# whether the GPU was exclusive.
set -u

: "${OUT:?set OUT to an output directory}"
: "${BUNDLE:?set BUNDLE to the copied thor_bundle directory}"
SUITE="${SUITE:-libero_spatial}"
PRECS="${PRECS:-nvfp4}"
ROWS="${ROWS:-default vae vae_trim vae_trim_fa4bb stack stack_no_vae stack_no_trim}"
PROFILES="${PROFILES:-}"
FULL="${FULL:-N_TASKS=10 FRAMES=0,60 SEEDS=0,1}"
CAL="$BUNDLE/imagewam_libero_calib_n64_v1.safetensors"
CAL_TRIM="$BUNDLE/imagewam_libero_calib_n64_trim_v2.safetensors"
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

declare -A CFG=(
  [default]=""
  [vae]="VAE_ENCODER=native VAE_GRAPH=1"
  [vae_trim]="VAE_ENCODER=native VAE_GRAPH=1 TEXT_TRIM=1"
  [vae_trim_fa4bb]="VAE_ENCODER=native VAE_GRAPH=1 TEXT_TRIM=1 FLASHRT_THOR_FA4=1"
  [stack]="VAE_ENCODER=native VAE_GRAPH=1 TEXT_TRIM=1 FLASHRT_THOR_FA4=1 FA4_MOT=1"
  [stack_no_vae]="TEXT_TRIM=1 FLASHRT_THOR_FA4=1 FA4_MOT=1"
  [stack_no_trim]="VAE_ENCODER=native VAE_GRAPH=1 FLASHRT_THOR_FA4=1 FA4_MOT=1"
)

# The rows this run walks: the named profiles, or the switch sets in ROWS.
if [ -n "$PROFILES" ]; then
  ROWLIST=""
  for p in $PROFILES; do ROWLIST="$ROWLIST profile_$p"; done
else
  ROWLIST="$ROWS"
fi

# parse_log <log>: prints the CSV fields "fr_min,fr_median,mae_median,p50,fa4,fa4_mot,fa4_fallback"
parse_log() {
  local f=$1 fr mae p50 eff fa4 mot fb
  fr=$(sed -n 's/^fr_vs_off .*min=\([-0-9.]*\) median=\([-0-9.]*\).*/\1,\2/p' "$f" | tail -1)
  mae=$(sed -n 's/^mae_fr_vs_gt .*median=\([-0-9.]*\).*/\1/p' "$f" | tail -1)
  p50=$(sed -n 's/^infer() P50=\([0-9.]*\)ms.*/\1/p' "$f" | tail -1)
  eff=$(grep '^effective_config' "$f" | tail -1)
  fa4=$(sed -n 's/.* use_fa4=\([A-Za-z]*\) .*/\1/p' <<< "$eff")
  mot=$(sed -n 's/.* use_fa4_mot=\([A-Za-z]*\) .*/\1/p' <<< "$eff")
  fb=$(sed -n 's/.* fa4_fallback_reason=\(.*\) calibration=.*/\1/p' <<< "$eff")
  fb=${fb//,/;}   # a fallback message may contain commas; keep the CSV columns intact
  echo "${fr:-NA,NA},${mae:-NA},${p50:-NA},${fa4:-NA},${mot:-NA},${fb:-NA}"
}

COMMIT=$(git rev-parse --short HEAD)
if [ -n "$PROFILES" ]; then
  TAG="${TAG:-${PRECS// /+}_profiles_${PROFILES// /+}}"
else
  TAG="${TAG:-${PRECS// /+}}"
fi
CSV="$OUT/matrix_${SUITE}_${TAG}.csv"
MD="$OUT/matrix_${SUITE}_${TAG}.md"
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state; report_jetson_clock_state()" \
  > "$OUT/matrix_clock.log" 2>&1
echo "precision,row,rc,fr_vs_off_min,fr_vs_off_median,mae_vs_gt_median,infer_p50_ms,use_fa4,use_fa4_mot,fa4_fallback,commit" > "$CSV"
{
  if [ -n "$PROFILES" ]; then
    echo "commit $COMMIT, suite $SUITE, $FULL, profiles $PROFILES"
  else
    echo "commit $COMMIT, suite $SUITE, $FULL"
  fi
  echo
  echo "| precision | row | rc | vs official min | vs official median | MAE vs GT median | infer P50 ms | FA4 bb | FA4 mot | FA4 fallback |"
  echo "|---|---|---:|---:|---:|---:|---:|---|---|---|"
} > "$MD"

for prec in $PRECS; do
  for row in $ROWLIST; do
    cfg=""
    cal=""
    prof=""
    if [ -n "$PROFILES" ]; then
      # A profile row: the named profile carries the switches, so only the
      # profile's name is exported (the compare script reads PROFILE).
      prof="PROFILE=${row#profile_}"
    else
      cfg="${CFG[$row]-__missing__}"
      [ "$cfg" = "__missing__" ] && { echo "unknown row: $row" >&2; continue; }
      case "$prec" in
        fp8_static|fp8_static_cutlass)
          if [[ "$cfg" == *TEXT_TRIM=1* ]]; then cal="CALIBRATION=$CAL_TRIM"; else cal="CALIBRATION=$CAL"; fi ;;
      esac
    fi
    log="$OUT/matrix_${SUITE}_${prec}_${row}.log"
    echo "[$(date +%T)] $prec / $row"
    # shellcheck disable=SC2086
    env PRECISION=$prec SUITE=$SUITE $FULL $cfg $prof $cal python benchmarks/imagewam_e2e_official_compare.py > "$log" 2>&1
    rc=$?
    IFS=, read -r frmin frmed mae p50 fa4 mot fb <<< "$(parse_log "$log")"
    echo "$prec,$row,$rc,$frmin,$frmed,$mae,$p50,$fa4,$mot,$fb,$COMMIT" >> "$CSV"
    echo "| $prec | $row | $rc | $frmin | $frmed | $mae | $p50 | $fa4 | $mot | $fb |" >> "$MD"
  done
done
echo "done: $MD"
