"""Chấm cặp (câu hỏi, đề mục văn bản) bằng Qwen3-Reranker-0.6B GỐC (không fine-tune).

Model trả lời yes/no; điểm = P(yes) lấy từ logit của hai token đó ở vị trí cuối, đúng
template chính thức của Qwen3-Reranker. Đây là bộ đọc ở cấp VĂN BẢN (mục lục cả văn bản),
bù cho cross-encoder vốn chỉ thấy từng chunk 450 từ. Đề mục do `legalir.headings` nạp.

GIỮ NGUYÊN thứ tự cặp truyền vào (thứ tự file điểm): điểm fp16 dao động theo batch ~1e-4,
đổi thứ tự là đổi bài.
"""
import torch

INSTRUCTION = ("Given a Vietnamese legal question, determine whether the document "
               "contains information relevant to answering it")
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
          "\"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class HeadingReranker:
    def __init__(self, model_path, max_length=1024):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.prefix_ids = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.max_length = max_length

    @torch.no_grad()
    def score_batch(self, pairs):
        raw = [f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
        budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        inputs = self.tok(raw, padding=False, truncation="longest_first",
                          return_attention_mask=False, max_length=budget)
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_ids + ids + self.suffix_ids
        inputs = self.tok.pad(inputs, padding=True, return_tensors="pt",
                              max_length=self.max_length)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
        stacked = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
        return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().float().cpu().tolist()

    def score(self, pairs, batch_size=24, log_every=100):
        out = []
        for i in range(0, len(pairs), batch_size):
            out.extend(self.score_batch(pairs[i:i + batch_size]))
            if (i // batch_size) % log_every == 0:
                print(f"  {i}/{len(pairs)}", flush=True)
        return out
