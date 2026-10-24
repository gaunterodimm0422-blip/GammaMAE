#!/usr/bin/env bash
# =====================================================================
# GammaMAE v2 GATE batch runner (VM side, "蚂蚁搬家" mode)
#   每个 chunk: Gate -> Hits ROOT -> compress_hits.C -> 删 Hits
#   每个 job  : 全部 chunk 完成 -> 拷到共享盘 -> 核对大小 -> 写 DONE -> 删本地
# 用法:
#   SHARE_OUT=/media/sf_vgate_shared/output/gammamae_v2 bash run_all.sh --dry-run
#   SHARE_OUT=/media/sf_vgate_shared/output/gammamae_v2 bash run_all.sh --only NaI_Cs137_d10_s0
#   SHARE_OUT=/media/sf_vgate_shared/output/gammamae_v2 nohup bash run_all.sh > ~/gmae_run.log 2>&1 &
# 可选环境变量: WORK(本地工作目录) MIN_FREE_GB GATE_BIN MATDB
# =====================================================================
set -u -o pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; ONLY=""; MAXJOBS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --max-jobs) MAXJOBS="$2"; shift ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac; shift
done

: "${SHARE_OUT:?必须显式设置 SHARE_OUT, 例如 SHARE_OUT=/media/sf_vgate_shared/output/gammamae_v2}"
WORK="${WORK:-$HOME/gmae_work}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
GATE_BIN="${GATE_BIN:-$(command -v Gate || true)}"
ROOT_BIN="$(command -v root || true)"

log(){ echo "[$(date '+%F %T')] $*"; }
die(){ log "STOP: $*"; exit 1; }
free_gb(){ df -Pk "$1" | awk 'NR==2{printf "%d", $4/1048576}'; }

# ---------------- preflight ----------------
[ -n "$GATE_BIN" ] || die "找不到 Gate 可执行文件 (可设 GATE_BIN=...)"
[ -n "$ROOT_BIN" ] || die "找不到 root 命令 (需要 ROOT 压缩 Hits)"
if [ -z "${MATDB:-}" ]; then
  for c in /home/vgate/Software/Gate/src/GateMaterials.db \
           $(find "$HOME" /opt /usr/local -name GateMaterials.db 2>/dev/null | head -5); do
    [ -f "$c" ] && { MATDB="$c"; break; }
  done
fi
[ -f "${MATDB:-}" ] || die "找不到 GateMaterials.db (可设 MATDB=...)"
if ! grep -Eq "^[[:space:]]*GAGG_Ce[[:space:]]*:" "$MATDB"; then   # 官方库无 GAGG: 复制一份并追加 Gd3Al2Ga3O12
  cp -f "$MATDB" "$PKG/GateMaterials_v2.db" && cat >> "$PKG/GateMaterials_v2.db" <<'EODB'

GAGG_Ce: d=6.63 g/cm3 ; n=4 ; state=solid
        +el: name=Gadolinium ; n=3
        +el: name=Aluminium  ; n=2
        +el: name=Gallium    ; n=3
        +el: name=Oxygen     ; n=12
EODB
  MATDB="$PKG/GateMaterials_v2.db"; log "已生成含 GAGG_Ce 的材料库副本 $MATDB"
fi
for m in NaITl GAGG_Ce Germanium; do
  grep -Eq "^[[:space:]]*${m}[[:space:]]*:" "$MATDB" || die "材料库 $MATDB 中没有材料 $m"
done
mkdir -p "$SHARE_OUT" "$WORK" || die "无法创建 $SHARE_OUT 或 $WORK"
touch "$SHARE_OUT/.write_test" && rm -f "$SHARE_OUT/.write_test" || die "共享盘 $SHARE_OUT 不可写(是否已挂载?)"
[ "${ALLOW_LOCAL_SHARE:-0}" = 1 ] || case "$SHARE_OUT" in "$HOME"*|/tmp*) die "SHARE_OUT 指向本地盘($SHARE_OUT), 请改为共享盘路径";; esac
log "Gate=$GATE_BIN  root=$ROOT_BIN"
log "MATDB=$MATDB"
log "WORK=$WORK (free $(free_gb "$WORK") GB)  SHARE_OUT=$SHARE_OUT (free $(free_gb "$SHARE_OUT") GB)"
cp -f "$PKG/jobs.tsv" "$PKG/source_lines.csv" "$PKG/background_lines.csv" "$SHARE_OUT/" 2>/dev/null

ndone=0; nrun=0
while IFS=$'\t' read -r JOB NCH CHS SEED0 PRIM ENTER ELINE; do
  [ "$JOB" = "job" ] && continue
  if [ -n "$ONLY" ] && ! [[ "$JOB" =~ $ONLY ]]; then continue; fi
  DST="$SHARE_OUT/$JOB"
  if [ -f "$DST/DONE" ]; then ndone=$((ndone+1)); continue; fi
  if [ -d "$DST" ]; then log "$JOB: 共享盘有未完成残留, 删除后重跑"; [ $DRY = 1 ] || rm -rf "$DST"; fi
  if [ "$MAXJOBS" -gt 0 ] && [ "$nrun" -ge "$MAXJOBS" ]; then break; fi
  nrun=$((nrun+1))
  JW="$WORK/$JOB"; mkdir -p "$JW"
  cp -f "$PKG/jobs/$JOB/meta.json" "$JW/"
  log "== $JOB  chunks=$NCH x ${CHS}s  primaries/s=$PRIM  entering/s=$ENTER"
  for ((c=0; c<NCH; c++)); do
    [ -f "$JW/events_c$c.ok" ] && continue
    TS=$((c*CHS)); TE=$(((c+1)*CHS)); SEED=$((SEED0+c))
    sed -e "s#@MATDB@#$MATDB#" -e "s#@OUTNAME@#hits_c$c#" -e "s#@SEED@#$SEED#" \
        -e "s#@TSLICE@#$CHS#" -e "s#@TSTART@#$TS#" -e "s#@TSTOP@#$TE#" \
        "$PKG/jobs/$JOB/template.mac" > "$JW/run_c$c.mac"
    if [ $DRY = 1 ]; then log "  [dry] $GATE_BIN run_c$c.mac  (t=$TS-$TE s, seed=$SEED)"; continue; fi
    F=$(free_gb "$WORK"); [ "$F" -ge "$MIN_FREE_GB" ] || die "本地盘剩余 ${F} GB < ${MIN_FREE_GB} GB"
    t0=$(date +%s)
    ( cd "$JW" && "$GATE_BIN" "run_c$c.mac" < /dev/null > "gate_c$c.log" 2>&1 ); rc=$?
    t1=$(date +%s)
    [ $rc -eq 0 ] || die "$JOB chunk $c: Gate 返回 $rc, 见 $JW/gate_c$c.log"
    [ -s "$JW/hits_c$c.root" ] || die "$JOB chunk $c: 没有生成 hits_c$c.root, 见 $JW/gate_c$c.log"
    HS=$(du -m "$JW/hits_c$c.root" | cut -f1)
    ( cd "$JW" && "$ROOT_BIN" -l -b -q "$PKG/tools/compress_hits.C(\"hits_c$c.root\",\"events_c$c.root\")" < /dev/null > "compress_c$c.log" 2>&1 )
    grep -q COMPRESS_OK "$JW/compress_c$c.log" || die "$JOB chunk $c: 压缩失败, 见 $JW/compress_c$c.log"
    rm -f "$JW/hits_c$c.root"; touch "$JW/events_c$c.ok"
    t2=$(date +%s)
    log "  chunk $c/$NCH  gate=$((t1-t0))s compress=$((t2-t1))s hits=${HS}MB  $(grep -o 'hits=.*' "$JW/compress_c$c.log")"
    printf "%s\t%d\t%d\t%d\t%d\n" "$JOB" "$c" "$((t1-t0))" "$((t2-t1))" "$HS" >> "$WORK/timing.tsv"
  done
  [ $DRY = 1 ] && continue
  ( cd "$JW" && "$ROOT_BIN" -l -b -q "$PKG/tools/quick_check.C(\"$JW\",$ELINE,600)" < /dev/null 2>/dev/null | grep CHECK > check.txt )
  log "  $(cat "$JW/check.txt")"
  mkdir -p "$DST" || die "无法创建 $DST"
  for f in "$JW"/events_c*.root "$JW"/meta.json "$JW"/check.txt "$JW"/run_c0.mac "$JW"/gate_c0.log; do
    cp -f "$f" "$DST/" || die "拷贝 $f 失败"
    [ "$(stat -c %s "$f")" = "$(stat -c %s "$DST/$(basename "$f")")" ] || die "$f 拷贝后大小不一致"
  done
  date '+%F %T' > "$DST/DONE"
  cp -f "$WORK/timing.tsv" "$SHARE_OUT/timing.tsv" 2>/dev/null
  rm -rf "$JW"
  ndone=$((ndone+1))
  log "  -> $DST  完成 (累计 $ndone)"
done < "$PKG/jobs.tsv"
log "结束: 已完成 $ndone 个 job; 本次运行 $nrun 个"
