import os
import json
import argparse
import time
import re
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import torch.multiprocessing as mp
from openai import OpenAI
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# OpenAI-compatible HTTP client (aligned with code/LongBench/pred.py)
# -----------------------------------------------------------------------------

# 为 chat 模板等与客户端 token 计数偏差预留的固定余量（用于输入上限与总窗口对齐）
_CONTEXT_RESERVE_TOKENS = 128


def _default_base_url() -> str:
    v = os.environ.get("OPENAI_BASE_URL")
    if v:
        return v
    port = os.environ.get("VLLM_PORT", "8000")
    return f"http://127.0.0.1:{port}/v1"


def _tokenizer_path(model_key: str, model2path: dict) -> str:
    v = model2path[model_key]
    return v["tokenizer"] if isinstance(v, dict) else v


def _api_model_id(model_key: str, model2path: dict) -> str:
    v = model2path[model_key]
    if isinstance(v, dict):
        return v.get("api_model", v["tokenizer"])
    return v


def _count_tokens_text(text: str, tokenizer: Any) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0


def _resolve_total_context_tokens(
    model_key: str, model2path: dict, tokenizer: Any
) -> Optional[int]:
    """
    推理总上下文上限（prompt + 生成），用于钳位 max_tokens，避免超过服务端窗口。

    优先级：model2path[model_key] 字典里的 max_model_len >
    tokenizer.model_max_length（仅在数值合理时采用，排除 HF 常见的超大哨兵值）。
    """
    v = model2path.get(model_key)
    if isinstance(v, dict):
        ml = v.get("max_model_len")
        if ml is not None:
            try:
                return int(ml)
            except (TypeError, ValueError):
                pass
    mml = getattr(tokenizer, "model_max_length", None)
    if mml is None:
        return None
    try:
        n = int(mml)
    except (TypeError, ValueError):
        return None
    # 排除「无限制」类哨兵（常见为 1e30 量级或 int64 极大值）
    if n <= 0 or n > 1_000_000:
        return None
    return n


def _effective_input_max_tokens(
    max_length: int, max_gen: int, total_ctx: Optional[int], reserve: int
) -> int:
    """
    输入侧 token 上限：默认等于 model2maxlen（max_length）。
    当 max_length + max_gen 超过模型总上下文 total_ctx 时，收紧为
    total_ctx - max_gen - reserve，使「配置上的最长输入 + 最长输出 + 余量」
    落在总窗口内；再配合 --overlength 对超长 prompt 跳过或裁 context。
    total_ctx 未知时不收紧。
    """
    L = int(max_length)
    G = int(max_gen)
    if total_ctx is None:
        return L
    M = int(total_ctx)
    if L + G <= M:
        return L
    cap = M - G - int(reserve)
    return max(0, min(L, cap))


def query_llm_http(
    client: OpenAI,
    api_model: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    stop: Optional[list] = None,
) -> str:
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                stream=False,
                stop=stop,
            )
            return (completion.choices[0].message.content or "").strip()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = str(e) or repr(e)
            print(f'Error Occurs: "{msg}"        Retry ...')
            resp = getattr(e, "response", None)
            body_text = None
            if resp is not None:
                try:
                    status = getattr(resp, "status_code", None)
                    text = getattr(resp, "text", None)
                    if callable(text):
                        text = text()
                    body_text = text
                    print(f"[server_response] status={status} body={text}")
                except Exception:
                    pass
            err_text = body_text or msg
            m_ctx = re.search(r"maximum context length is\s+(\d+)\s+tokens", err_text)
            if m_ctx:
                print(
                    f"[data] server max context={int(m_ctx.group(1))}; prompt still exceeds limit. Skip."
                )
                return ""
            time.sleep(1)
    print("Max tries. Failed.")
    return ""


def query_llm_http_streaming(
    client: OpenAI,
    api_model: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    stop: Optional[list] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    stream=True 统计 TTFT / E2E（客户端视角，含网络与服务端排队）。
    与 code/LongBench/pred.py 中 query_llm_streaming 行为对齐。
    """
    start = time.perf_counter()
    first_chunk_t: Optional[float] = None
    text_parts: list = []
    tries = 0

    while tries < 5:
        tries += 1
        try:
            stream = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                stream=True,
                stop=stop,
            )
            for chunk in stream:
                now = time.perf_counter()
                if first_chunk_t is None:
                    first_chunk_t = now
                delta = None
                try:
                    delta = chunk.choices[0].delta
                except Exception:
                    delta = None
                if delta is not None:
                    content = getattr(delta, "content", None)
                    if content:
                        text_parts.append(content)
                else:
                    try:
                        msg = chunk.choices[0].message
                        if msg and getattr(msg, "content", None):
                            text_parts.append(msg.content)
                    except Exception:
                        pass
            end = time.perf_counter()
            out = "".join(text_parts).strip()
            metrics = {
                "ttft_ms": (first_chunk_t - start) * 1000 if first_chunk_t else None,
                "e2e_ms": (end - start) * 1000,
            }
            return out, metrics
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = str(e) or repr(e)
            print(f'Error Occurs: "{msg}"        Retry ...')
            resp = getattr(e, "response", None)
            body_text = None
            if resp is not None:
                try:
                    status = getattr(resp, "status_code", None)
                    text = getattr(resp, "text", None)
                    if callable(text):
                        text = text()
                    body_text = text
                    print(f"[server_response] status={status} body={text}")
                except Exception:
                    pass
            err_text = body_text or msg
            m_ctx = re.search(r"maximum context length is\s+(\d+)\s+tokens", err_text)
            if m_ctx:
                print(
                    f"[data] server max context={int(m_ctx.group(1))}; prompt still exceeds limit. Skip."
                )
                return "", {"ttft_ms": None, "e2e_ms": None}
            time.sleep(1)

    return "", {"ttft_ms": None, "e2e_ms": None}


def _built_prompt_to_text(
    tokenizer: Any, built: Union[str, Any], model_name: str
) -> str:
    """Local pred uses tensors for chatglm3; HTTP path needs a single string."""
    if isinstance(built, str):
        return built
    if hasattr(built, "tolist"):
        return tokenizer.decode(built.tolist(), skip_special_tokens=False)
    return str(built)


def parse_args(args=None):
    model2path = json.load(
        open(os.path.join(os.path.dirname(__file__), "config/model2path.json"), "r")
    )
    model_choices = list(model2path.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="可选：输出目录。不传则默认写入当前目录。"
        "传入后将写入 <save_dir>/pred/（LongBench）或 <save_dir>/pred_e/（LongBench-E）。",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=model_choices,
        help="Key in config/model2path.json（用于 tokenizer / api_model / maxlen 等配置）。",
    )
    parser.add_argument(
        "--e", action="store_true", help="Evaluate on LongBench-E"
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4))),
        help="Parallel worker processes (HTTP); does not require local GPUs.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="可选：本地数据目录。若提供，将优先从该目录读取数据集文件："
        "<data_path>/{dataset}.jsonl 或 .json；LongBench-E 则为 <data_path>/{dataset}_e.jsonl/.json。"
        "若对应文件不存在，将回退到 Hugging Face 在线 load_dataset。",
    )
    parser.add_argument(
        "--overlength",
        choices=["skip", "truncate_context"],
        default="skip",
        help="当拼接后的 prompt token 数超过「有效输入上限」时的处理："
        "skip=跳过；truncate_context=仅裁剪样本里的 {context}。"
        "有效输入上限默认等于 model2maxlen；若其与本数据集 max_gen 之和超过模型总上下文，"
        "会先收紧输入上限再应用本策略。",
    )
    parser.add_argument(
        "--measure_latency",
        dest="measure_latency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启：使用 stream=True 统计 TTFT_MS/E2E_MS，并结合 output_tokens 计算 TPOT_MS。"
        "关闭：--no-measure-latency，非流式请求，时延字段为 null（仍写入 input/output_tokens）。",
    )
    return parser.parse_args(args)


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    rank,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    model2path,
    out_path,
    measure_latency: bool,
    overlength: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_path(model_name, model2path), trust_remote_code=True
    )
    api_model = _api_model_id(model_name, model2path)
    base_url = _default_base_url()
    api_key = os.environ.get("OPENAI_API_KEY", "token-abc123")
    client = OpenAI(base_url=base_url, api_key=api_key)

    total_ctx = _resolve_total_context_tokens(model_name, model2path, tokenizer)
    effective_max_length = _effective_input_max_tokens(
        max_length, max_gen, total_ctx, _CONTEXT_RESERVE_TOKENS
    )
    if effective_max_length < max_length:
        print(
            f"[pred] worker-{rank}: tighten input token cap {max_length} -> {effective_max_length} "
            f"(total_ctx={total_ctx}, max_gen={max_gen}, reserve={_CONTEXT_RESERVE_TOKENS})"
        )
    if total_ctx is not None and int(max_length) + int(max_gen) > int(total_ctx):
        if effective_max_length < 1:
            print(
                f"[pred] worker-{rank}: warn: effective input cap < 1; "
                f"most prompts will be skipped unless empty (total_ctx={total_ctx}, max_gen={max_gen})."
            )

    for json_obj in tqdm(data, position=rank, desc=f"worker-{rank}"):
        prompt = _build_prompt_with_overlength_policy(
            prompt_format=prompt_format,
            json_obj=json_obj,
            tokenizer=tokenizer,
            max_length=effective_max_length,
            overlength=overlength,
        )
        if prompt is None:
            continue

        # 客户端侧 token：与 vLLM chat 模板额外 token 不完全一致，仅作近似（同 code/LongBench/pred.py 说明）
        input_tokens = _count_tokens_text(prompt, tokenizer)

        temp = 1.0 if dataset == "samsum" else 0.0

        if measure_latency:
            pred_raw, metrics = query_llm_http_streaming(
                client,
                api_model,
                prompt,
                max_new_tokens=max_gen,
                temperature=temp,
                stop=None,
            )
        else:
            pred_raw = query_llm_http(
                client,
                api_model,
                prompt,
                max_new_tokens=max_gen,
                temperature=temp,
                stop=None,
            )
            metrics = {"ttft_ms": None, "e2e_ms": None}

        pred = post_process(pred_raw, model_name)
        output_tokens = _count_tokens_text(pred, tokenizer)

        ttft = metrics.get("ttft_ms")
        e2e = metrics.get("e2e_ms")
        tpot_ms = None
        if (
            isinstance(ttft, (int, float))
            and isinstance(e2e, (int, float))
            and output_tokens > 0
            and e2e >= ttft
        ):
            tpot_ms = (e2e - ttft) / output_tokens

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "ttft_ms": ttft,
                    "e2e_ms": e2e,
                    "tpot_ms": tpot_ms,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


def _truncate_text_head_by_tokens(text: str, tokenizer: Any, keep_tokens: int) -> str:
    keep = max(int(keep_tokens), 0)
    if keep <= 0:
        return ""
    try:
        ids = tokenizer.encode(text)
    except Exception:
        # fallback: if tokenizer.encode isn't available for some remote-code tokenizers
        ids = tokenizer(text, truncation=False, return_tensors="pt").input_ids[0].tolist()
    if len(ids) <= keep:
        return text
    try:
        return tokenizer.decode(ids[:keep], skip_special_tokens=True)
    except Exception:
        return tokenizer.decode(ids[:keep])


def _build_prompt_with_overlength_policy(
    *,
    prompt_format: str,
    json_obj: dict,
    tokenizer: Any,
    max_length: int,
    overlength: str,
) -> Optional[str]:
    """
    将样本拼成最终 prompt（字符串），并按 overlength 策略处理超长：
    - skip: 超过 max_length token 直接返回 None（跳过）
    - truncate_context: 仅裁剪 json_obj['context']，让整体 token 数不超过 max_length
      （若模板无 {context} 或无 context 字段，则无法裁剪，直接返回 None）
    """
    try:
        prompt = prompt_format.format(**json_obj)
    except KeyError as e:
        # 数据字段缺失，跳过
        print(f"[data] missing field {e} in sample keys={list(json_obj.keys())}")
        return None

    # 先算当前 prompt 的 token 长度
    try:
        ids = tokenizer.encode(prompt)
        prompt_len = len(ids)
    except Exception:
        prompt_len = len(tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0])

    if prompt_len <= int(max_length):
        return prompt

    if overlength == "skip":
        return None

    # truncate_context：只裁剪 {context}
    if "{context}" not in prompt_format or "context" not in json_obj:
        return None

    # 固定部分 = 用空 context 替换后的 prompt
    base_obj = dict(json_obj)
    base_obj["context"] = ""
    try:
        fixed_prompt = prompt_format.format(**base_obj)
    except Exception:
        return None

    try:
        fixed_len = len(tokenizer.encode(fixed_prompt))
    except Exception:
        fixed_len = len(tokenizer(fixed_prompt, truncation=False, return_tensors="pt").input_ids[0])

    ctx_budget = int(max_length) - int(fixed_len)
    if ctx_budget <= 0:
        return None

    ctx = json_obj.get("context") or ""
    ctx_trunc = _truncate_text_head_by_tokens(str(ctx), tokenizer, ctx_budget)
    new_obj = dict(json_obj)
    new_obj["context"] = ctx_trunc
    try:
        prompt2 = prompt_format.format(**new_obj)
    except Exception:
        return None

    # 保险：若仍超限则放弃（避免二次裁剪改变语义太多）
    try:
        if len(tokenizer.encode(prompt2)) > int(max_length):
            return None
    except Exception:
        if len(tokenizer(prompt2, truncation=False, return_tensors="pt").input_ids[0]) > int(max_length):
            return None
    return prompt2


def _load_local_items(path: str) -> list:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON format (expected list): {path}")


def _try_load_dataset_local(data_dir: str, dataset: str, is_e: bool) -> Optional[list]:
    """
    约定本地文件命名：
    - LongBench:     <data_dir>/{dataset}.jsonl 或 .json
    - LongBench-E:   <data_dir>/{dataset}_e.jsonl 或 .json
    返回 list[dict] 或 None（表示本地未命中）。
    """
    if not data_dir:
        return None
    if not os.path.isdir(data_dir):
        raise SystemExit(f"--data_path 需要是目录：{data_dir}")
    base = f"{dataset}_e" if is_e else dataset
    for ext in (".jsonl", ".json"):
        p = os.path.join(data_dir, base + ext)
        if os.path.exists(p):
            print(f"[data] using local dataset file: {p}")
            return _load_local_items(p)
    return None


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    cfg_dir = os.path.join(os.path.dirname(__file__), "config")
    model2path = json.load(open(os.path.join(cfg_dir, "model2path.json"), "r"))
    model2maxlen = json.load(open(os.path.join(cfg_dir, "model2maxlen.json"), "r"))

    model_name = args.model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ]
    dataset2prompt = json.load(open(os.path.join(cfg_dir, "dataset2prompt.json"), "r"))
    dataset2maxlen = json.load(open(os.path.join(cfg_dir, "dataset2maxlen.json"), "r"))

    base_out = args.save_dir or "."
    os.makedirs(base_out, exist_ok=True)
    if args.e:
        os.makedirs(os.path.join(base_out, "pred_e"), exist_ok=True)
    else:
        os.makedirs(os.path.join(base_out, "pred"), exist_ok=True)

    n_proc = max(1, int(args.n_proc))
    for dataset in datasets:
        if args.e:
            local_items = _try_load_dataset_local(args.data_path, dataset, is_e=True)
            if local_items is None:
                data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
                data_all = [data_sample for data_sample in data]
            else:
                data_all = local_items
            out_path = os.path.join(base_out, f"pred_e/{dataset}.jsonl")
        else:
            local_items = _try_load_dataset_local(args.data_path, dataset, is_e=False)
            if local_items is None:
                data = load_dataset("THUDM/LongBench", dataset, split="test")
                data_all = [data_sample for data_sample in data]
            else:
                data_all = local_items
            out_path = os.path.join(base_out, f"pred/{dataset}.jsonl")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_subsets = [data_all[i::n_proc] for i in range(n_proc)]
        processes = []
        for rank in range(n_proc):
            p = mp.Process(
                target=get_pred,
                args=(
                    rank,
                    data_subsets[rank],
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    model_name,
                    model2path,
                    out_path,
                    args.measure_latency,
                    args.overlength,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
