#!/usr/bin/env bash
# Vendored verbatim từ eval/gpu_lib.sh (repo dev) — chỉ thêm comment này (ghi provenance).
# Ham dung chung de chon GPU tren may dung chung. `source` file nay.
#
# Vi sao co file nay: ban truoc moi job tu viet `flock 9; G=$(pick N); chay...`. Sai o
# cho `pick` NGU TRONG LUC DANG GIU KHOA — job can khe 10GB om khoa hang gio, chan sach
# cac job chi can 5GB du GPU dang co 5,5GB trong. Da xay ra that: job_title (5000) bi
# job_submit_bgem3 (6000) chan.
#
# Nguyen tac: doi thi KHONG giu khoa; chi giu khoa dung luc kiem-lai-va-phong, roi tha.
# Van giu 75s sau khi phong vi co ~60s tre giua luc chon va luc bo nho hien ra trong
# nvidia-smi — khong giu thi hai job cung chon mot the roi OOM ca hai.

# v2: file cu /tmp/uit_dsc_gpu_pick.lock da HONG VINH VIEN. Ba tien trinh python ket
# D-state tren NFS ke thua fd 9 tu bash cha; bash cha bi kill -9 truoc khi kip `flock -u`
# nen khoa khong duoc tha, ma python thi kill -9 khong go duoc. Khoa flock gan voi "open
# file description", ma description do duoc CHIA SE khi fork -> con song la khoa song.
# Xem them `"$@" 9>&- &` ben duoi: do la cach chan goc, doi ten file chi la don dep.
LOCK=${LOCK:-/tmp/uit_dsc_gpu_pick2.lock}

# ALLOW_GPUS: chi duoc dung nhung the nay. Mac dinh 0-7 (nguoi dung mo toan bo the,
# 2026-09-04). Nguong VRAM van la thu chan tranh chap: chi bam the con du bo nho.
ALLOW_GPUS=${ALLOW_GPUS:-"0 1 2 3 4 5 6 7"}

_best_free(){   # in ra "<MiB trong> <chi so GPU>" cua the trong nhat TRONG DANH SACH
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | awk -F', ' -v ok="$ALLOW_GPUS" '
        BEGIN{n=split(ok,a," "); for(i=1;i<=n;i++) allow[a[i]]=1}
        ($1 in allow){split($2,u," ");split($3,t," ");print t[1]-u[1], $1}' \
    | sort -rn | head -1
}

# Cho den khi tien trinh vua phong THAT SU cap phat VRAM tren the da chon, roi moi tha
# khoa. `sleep 75` co dinh la sai: no gia dinh model nap xong trong 75s. Model chua co
# trong cache HF phai tai vai phut (server nay hay timeout voi us.aws.cdn.hf.co), suot
# thoi gian do the van trong trong mat job ke tiep -> hai job cung chon mot the roi OOM
# ca hai. Da xay ra that: vietlegal-e5 shard 2 va 3 cung nhay vao GPU 0 luc 12:59-13:00.
# Tin hieu la BO NHO TRONG TREN THE DA CHON GIAM DI, khong phai "PID nay co trong
# nvidia-smi". Ban bam theo PID da that bai that: worker vietlegal-e5 shard 2 om khoa
# tron 10 phut du python con cua no da nam tren GPU 6 tu lau, chan 7 worker khac trong
# khi GPU 1 con trong 11 GB. Do bo nho thi khong the sai — do dung thu ma job ke tiep
# se doc de quyet dinh.
#
# Tran 150s: du cho model DA prefetch ve cache nap xong; het tran thi tha khoa nhu ban
# `sleep 75` cu, tuc la xau nhat cung khong te hon truoc.
_wait_alloc(){
  local pid=$1 gpu=$2 t0=$SECONDS before now
  before=$(_free_of "$gpu")
  while kill -0 "$pid" 2>/dev/null; do
    now=$(_free_of "$gpu")
    [ -n "$now" ] && [ -n "$before" ] && [ $((before - now)) -ge 500 ] && return 0
    [ $((SECONDS - t0)) -ge "${ALLOC_TIMEOUT:-150}" ] && return 0
    sleep 5
  done
}

_free_of(){     # MiB con trong tren the $1
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | awk -F', ' -v g="$1" '$1==g{split($2,u," ");split($3,t," ");print t[1]-u[1]}'
}

# run_on_gpu <nguong MiB> <ten_ham>
# Dat bien G = chi so GPU (da export) roi goi <ten_ham> o nen. Tra ve ma thoat cua no.
# Dung TEN HAM chu khong phai chuoi lenh: chuoi lenh phai qua `sh -c`, ma khi do $G o
# shell cha khong nhin thay duoc, con noi chuoi thi vo so bay trich dan.
run_on_gpu(){
  local thresh=$1; shift
  while :; do
    # --- doi, KHONG giu khoa ---
    # Poll 5s chu khong 40s: tren may dung chung (7-8 nguoi), khe trong chi ton tai
    # vai giay truoc khi co nguoi khac vao. 40s la du de mat sach moi khe. nvidia-smi
    # ton ~50ms nen 5s van khong dang ke.
    while :; do
      local g; g=$(_best_free)
      [ "${g% *}" -ge "$thresh" ] && break
      sleep "${POLL_SEC:-5}"
    done
    # --- kiem lai duoi khoa roi phong ngay ---
    exec 9>"$LOCK"; flock 9
    local g2; g2=$(_best_free)
    if [ "${g2% *}" -ge "$thresh" ]; then
      G=${g2#* }; export G
      # 9>&- : DONG fd khoa truoc khi phong con. Khong dong thi con ke thua open file
      # description cua khoa; con treo (D-state tren NFS, kill -9 vo hieu) la khoa bi
      # giu vinh vien du bash cha da chet. Da lam hong han mot file khoa vi loi nay.
      "$@" 9>&- &
      local P=$!
      _wait_alloc "$P" "$G"    # cho bo nho THAT SU hien ra truoc khi tha khoa
      flock -u 9; exec 9>&-
      wait $P
      return $?
    fi
    flock -u 9; exec 9>&-      # nguoi khac vua chiem mat, thu lai ngay
    sleep 5
  done
}
