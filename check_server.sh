#!/usr/bin/env bash
# check_server.sh — Quick server load check before starting heavy jobs.
# Usage: bash check_server.sh

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}!${RESET}  $*"; }
fail() { echo -e "  ${RED}✗${RESET}  $*"; }
hdr()  { echo -e "\n${BOLD}── $* ──${RESET}"; }

score=0   # 0=green, 1=yellow, 2=red

# ── Hostname & time ──────────────────────────────────────────────────────────
echo -e "${BOLD}========================================${RESET}"
echo -e "${BOLD} Server status check — $(hostname)${RESET}"
echo -e " $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${BOLD}========================================${RESET}"

# ── GPU ──────────────────────────────────────────────────────────────────────
hdr "GPU (nvidia-smi)"
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found"
else
    # Per-GPU summary
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name util mem_used mem_total temp; do
        util="${util// /}"; mem_used="${mem_used// /}"
        mem_total="${mem_total// /}"; temp="${temp// /}"
        pct=$(( mem_used * 100 / mem_total ))
        label="GPU${idx} [${name}]  util=${util}%  mem=${mem_used}/${mem_total}MiB (${pct}%)  temp=${temp}°C"
        if   (( util > 80 || pct > 80 )); then fail  "$label"
        elif (( util > 30 || pct > 40 )); then warn  "$label"
        else                                   ok    "$label"
        fi
    done

    # Running GPU processes (other users)
    echo ""
    echo "  Processes using GPU:"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name \
               --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r pid mem pname; do
        pid="${pid// /}"; mem="${mem// /}"; pname="${pname// /}"
        owner=$(ps -o user= -p "$pid" 2>/dev/null || echo "?")
        echo "    PID ${pid} | ${mem}MiB | user=${owner} | ${pname##*/}"
    done || true
fi

# ── CPU ──────────────────────────────────────────────────────────────────────
hdr "CPU"
# Load average
read -r l1 l5 l15 _ < /proc/loadavg
ncpu=$(nproc)
pct_l1=$(awk "BEGIN{printf \"%.0f\", $l1/$ncpu*100}")
label="Load avg: ${l1} / ${l5} / ${l15}  (1/5/15 min)  CPUs: ${ncpu}  → ~${pct_l1}% of capacity"
if   (( pct_l1 > 90 )); then fail  "$label"; score=2
elif (( pct_l1 > 60 )); then warn  "$label"; (( score < 1 )) && score=1
else                         ok    "$label"
fi

# Top CPU consumers
echo "  Top 5 CPU processes:"
ps -eo user,pid,pcpu,pmem,comm --sort=-%cpu 2>/dev/null | head -n 6 | \
    awk 'NR==1{printf "    %-12s %6s %6s %6s  %s\n",$1,$2,$3,$4,$5}
         NR>1{printf "    %-12s %6s %6s %6s  %s\n",$1,$2,$3,$4,$5}'

# ── RAM ──────────────────────────────────────────────────────────────────────
hdr "RAM"
mem_line=$(free -m | awk 'NR==2')
total=$(echo "$mem_line" | awk '{print $2}')
used=$(echo  "$mem_line" | awk '{print $3}')
avail=$(echo "$mem_line" | awk '{print $7}')
pct=$(( used * 100 / total ))
label="Used: ${used} MiB / ${total} MiB (${pct}%)  Available: ${avail} MiB"
if   (( pct > 85 )); then fail  "$label"; (( score < 2 )) && score=2
elif (( pct > 65 )); then warn  "$label"; (( score < 1 )) && score=1
else                     ok    "$label"
fi

# ── Disk I/O (brief) ─────────────────────────────────────────────────────────
hdr "Disk — home filesystem"
df -h "$HOME" | awk 'NR==2{
    pct=$5; sub(/%/,"",pct);
    used=$3; total=$2; avail=$4;
    printf "  Used: %s / %s  avail: %s  (%s%%)\n", used, total, avail, pct
}'

# ── Logged-in users ──────────────────────────────────────────────────────────
hdr "Logged-in users"
who | awk '{printf "  %-12s  tty=%-10s  from=%s  since=%s %s\n", $1,$2,$NF,$(NF-1),$(NF)}' || true
n_users=$(who | wc -l)
if (( n_users > 3 )); then
    warn "${n_users} users currently logged in — server may be busy"
    (( score < 1 )) && score=1
fi

# ── Verdict ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}========================================${RESET}"
case $score in
    0) echo -e "${GREEN}${BOLD} ✓  Server looks free — safe to launch jobs${RESET}" ;;
    1) echo -e "${YELLOW}${BOLD} !  Moderate load — proceed with care${RESET}" ;;
    2) echo -e "${RED}${BOLD} ✗  High load — consider waiting or using another node${RESET}" ;;
esac
echo -e "    Other aulus nodes: aulus2 aulus3 aulus4 aulus6 aulus7"
echo -e "${BOLD}========================================${RESET}"
