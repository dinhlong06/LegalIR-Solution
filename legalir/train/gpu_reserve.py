# Vendored verbatim từ eval/gpu_reserve.py (repo dev) — chỉ thay đổi duy nhất là
# comment này (ghi provenance). Dùng cho các script train trong legalir/train/.
"""Giữ chỗ VRAM tới mức đỉnh, để tiến trình khác trên máy dùng chung không chen vào
giữa chừng rồi làm ta OOM lúc phình lên đỉnh.

Cơ chế: caching allocator của PyTorch **không trả bộ nhớ đã cấp phát về driver**
(miễn là KHÔNG bật `expandable_segments:True`). Nên cấp phát một loạt khối lớn rồi
`del` chúng đi: bộ nhớ nằm lại trong cache của ta — người khác không lấy được, còn ta
thì tái sử dụng chính nó cho activation về sau. Không lãng phí, chỉ là chiếm sớm hơn.

Hai cái bẫy đã trả giá để biết:

1. **Phải gọi SAU khi model đã lên GPU.** `from_pretrained` (qua accelerate) có gọi
   `torch.cuda.empty_cache()`; giữ chỗ trước đó là bị xoá sạch. Triệu chứng: log báo
   giữ 10GB nhưng `nvidia-smi` chỉ thấy 5,8GB.

2. **Không được bật `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** cùng lúc —
   cờ đó làm PyTorch trả bộ nhớ về driver, tức tự tay nhường chỗ cho người khác.
"""
import os


# Giữ tham chiếu ở cấp module: chừng nào list này còn sống thì PyTorch KHÔNG thể trả
# phần đó về driver, kể cả khi nó đang bí bộ nhớ.
_BALLAST = []


def reserve_vram(gb: float, device: int = 0, log=print, chunk_gb: float = 0.25,
                 ballast_gb: float = 0.0) -> float:
    """Chiếm tới `gb` GB (hoặc gần hết chỗ trống nếu ít hơn). Trả về số GB giữ được.

    `ballast_gb` là phần giữ CỨNG, không bao giờ nhả. Phần còn lại chỉ nằm trong cache.

    Vì sao phải tách hai loại — đo được trên máy thật: hai tiến trình `score_pairs`
    log "giữ chỗ 3,50 GB · PyTorch reserved 4,56 GB" nhưng `nvidia-smi` chỉ thấy
    2.820 MiB. Bộ nhớ đã bị trả về driver. Nguyên nhân KHÔNG phải `empty_cache`
    (FlagEmbedding không gọi) mà là chính bộ cấp phát của PyTorch: khi một lần cấp
    phát không tìm được khối liền mạch đủ lớn, nó gọi `release_cached_blocks()` —
    tức cudaFree mọi segment đang rỗi trong cache — rồi mới thử lại. Chỗ ta "giữ" bằng
    cách `del` nằm đúng trong đám segment rỗi đó.

    Hệ quả không tránh được: bộ nhớ hoặc **ta dùng lại được** (nằm trong cache, và vì
    thế bị nhả khi bí) hoặc **khoá thật** (ballast, và vì thế ta không dùng được). Nên:
      ballast = (khe muốn giữ) − (đỉnh thật của chính mình)
    Phần đỉnh của mình cứ để trong cache, phần dôi ra thì khoá cứng cho người khác
    khỏi chen vào.
    """
    if gb <= 0:
        return 0.0
    import torch
    if not torch.cuda.is_available():
        return 0.0
    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        log("⚠️  expandable_segments đang bật — giữ chỗ sẽ vô hiệu, PyTorch trả "
            "bộ nhớ về driver. Bỏ cờ đó đi.")
    free_b, _ = torch.cuda.mem_get_info(device)
    want = min(gb, free_b / 2 ** 30 - 0.4)          # chừa 0,4GB cho CUDA context
    chunks, per = [], int(chunk_gb * (1024 ** 3) / 2)
    try:
        for _ in range(max(0, int(want / chunk_gb))):
            chunks.append(torch.empty(per, dtype=torch.float16, device=f"cuda:{device}"))
    except torch.cuda.OutOfMemoryError:
        pass
    got = len(chunks) * chunk_gb
    n_hold = min(len(chunks), int(max(ballast_gb, 0.0) / chunk_gb))
    if n_hold:
        _BALLAST.extend(chunks[:n_hold])            # không bao giờ nhả
    del chunks
    log(f"giữ chỗ VRAM: {got:.2f} GB (khoá cứng {n_hold*chunk_gb:.2f} GB, "
        f"còn lại trong cache) · PyTorch reserved "
        f"{torch.cuda.memory_reserved(device) / 2 ** 30:.2f} GB")
    return got
