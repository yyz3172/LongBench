import os, csv, json
import argparse
import time
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import re
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
import torch.multiprocessing as mp

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())


def _tokenizer_path(model_key):
    v = model_map[model_key]
    return v["tokenizer"] if isinstance(v, dict) else v


def _api_model_id(model_key):
    v = model_map[model_key]
    if isinstance(v, dict):
        return v.get("api_model", v["tokenizer"])
    return v


def _encode_text_for_model(text: str, model_key: str, tokenizer):
    if model_key in model_map:
        return tokenizer.encode(text)
    return tokenizer.encode(text, disallowed_special=())


def _decode_ids_for_model(ids, model_key: str, tokenizer) -> str:
    if model_key in model_map:
        return tokenizer.decode(ids, skip_special_tokens=True)
    return tokenizer.decode(ids)


def _truncate_head_tokens(text: str, limit_tokens: int, model_key: str, tokenizer) -> str:
    """按 token 头部截断（只保留前 limit_tokens 个 token）。"""
    keep = max(int(limit_tokens), 0)
    if keep <= 0:
        return ""
    ids = _encode_text_for_model(text, model_key, tokenizer)
    if len(ids) <= keep:
        return text
    return _decode_ids_for_model(ids[:keep], model_key, tokenizer)


def build_prompt_truncating_context_head(
    *,
    template: str,
    context: str,
    question: str,
    c_a: str,
    c_b: str,
    c_c: str,
    c_d: str,
    model_key: str,
    tokenizer,
    max_prompt_len_tokens: int,
) -> str:
    """
    仅裁剪 context（$DOC$）的头部保留策略：
    - 先把模板中除 $DOC$ 以外的字段都替换掉，得到 prefix/suffix 的固定开销
    - 将剩余 token 预算全部分配给 context，并且只保留 context 的前半段
    - 若模板中无 $DOC$，则不做 context 裁剪，直接返回完整替换后的 prompt
    """
    # 先替换除 $DOC$ 外的占位符，确保 token 开销计算包含题干与选项等“固定部分”
    t = (
        template.replace("$Q$", (question or "").strip())
        .replace("$C_A$", (c_a or "").strip())
        .replace("$C_B$", (c_b or "").strip())
        .replace("$C_C$", (c_c or "").strip())
        .replace("$C_D$", (c_d or "").strip())
    )
    if "$DOC$" not in t:
        return t
    prefix, suffix = t.split("$DOC$", 1)
    fixed = prefix + suffix
    fixed_tokens = len(_encode_text_for_model(fixed, model_key, tokenizer))
    ctx_budget = int(max_prompt_len_tokens) - fixed_tokens
    if ctx_budget <= 0:
        raise ValueError(
            f"Prompt fixed part already exceeds token budget: "
            f"fixed_tokens={fixed_tokens} max_prompt_len_tokens={int(max_prompt_len_tokens)} "
            f"(model={model_key})."
        )
    ctx = _truncate_head_tokens((context or "").strip(), ctx_budget, model_key, tokenizer)
    return prefix + ctx + suffix

def _default_base_url() -> str:
    # 兼容 PD 分离：可通过环境变量覆盖端口/整条 base_url
    # - OPENAI_BASE_URL: 例如 "http://127.0.0.1:9010/v1"
    # - VLLM_PORT: 例如 "9010"（将拼成 http://127.0.0.1:<port>/v1）
    v = os.environ.get("OPENAI_BASE_URL")
    if v:
        return v
    port = os.environ.get("VLLM_PORT", "8000")
    return f"http://127.0.0.1:{port}/v1"


URL = _default_base_url()
API_KEY = "token-abc123"
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # vLLM 的 ChatCompletions 会套 chat template（role / special tokens），
    # 实际送入模型的 input_ids 会比 messages[0]["content"] 多一些 token。
    # 这里预留余量，避免“只超出几 token”导致 400。
    CHAT_TEMPLATE_BUFFER_TOKENS = 256

    tries = 0
    api_model = _api_model_id(model) if model in model_map else model
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                stream=False,
            )
            return completion.choices[0].message.content, None
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            msg = str(e) or repr(e)
            print(f'Error Occurs: "{msg}"        Retry ...')
            # 尽量打印服务端返回，方便定位 500 的具体原因（如超长上下文、模型名不匹配、并发过高等）
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
            # vLLM 常见报错：上下文不足（400），自动把 prompt 截断到服务端上限再重试
            err_text = body_text or msg
            m_ctx = re.search(r"maximum context length is\s+(\d+)\s+tokens", err_text)
            if m_ctx:
                # 只做一次裁剪（上游已按 context 预算裁剪）；若仍超限则直接失败，避免二次裁剪改变评测语义
                server_limit = int(m_ctx.group(1))
                print(
                    f"[data] server max context={server_limit}; prompt still exceeds limit after context truncation. Skip."
                )
                return "", None
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return '', None


def query_llm_streaming(
    prompt: str,
    model: str,
    tokenizer,
    client=None,
    temperature: float = 0.5,
    max_new_tokens: int = 128,
    stop=None,
) -> Tuple[str, Dict[str, Any]]:
    """
    使用 OpenAI 兼容的 stream=True 来统计 TTFT / E2E。
    注意：TTFT 是“客户端视角”的首 chunk 到达延迟，包含网络与服务端排队/预处理。
    """
    # vLLM 的 ChatCompletions 会套 chat template（role / special tokens）
    CHAT_TEMPLATE_BUFFER_TOKENS = 256

    api_model = _api_model_id(model) if model in model_map else model
    start = time.perf_counter()
    first_chunk_t: Optional[float] = None
    text_parts = []
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
            )
            for chunk in stream:
                now = time.perf_counter()
                if first_chunk_t is None:
                    first_chunk_t = now
                # 兼容不同 SDK / 服务端的 chunk 结构
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
                    # fallback：有些实现可能直接给 message/content
                    try:
                        msg = chunk.choices[0].message
                        if msg and getattr(msg, "content", None):
                            text_parts.append(msg.content)
                    except Exception:
                        pass
            end = time.perf_counter()
            out = "".join(text_parts)
            metrics = {
                "ttft_ms": (first_chunk_t - start) * 1000 if first_chunk_t else None,
                "e2e_ms": (end - start) * 1000,
            }
            return out, metrics
        except KeyboardInterrupt as e:
            raise e
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
                server_limit = int(m_ctx.group(1))
                print(
                    f"[data] server max context={server_limit}; prompt still exceeds limit after context truncation. Skip."
                )
                return "", {"ttft_ms": None, "e2e_ms": None}
            time.sleep(1)

    return "", {"ttft_ms": None, "e2e_ms": None}


def _count_tokens_text(text: str, model_key: str, tokenizer) -> int:
    if not text:
        return 0
    try:
        if model_key in model_map:
            return len(tokenizer.encode(text))
        return len(tokenizer.encode(text, disallowed_special=()))
    except Exception:
        return 0

def _longbench_v2_row(item):
    row = {
        "_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "question": item["question"],
        "choice_A": item["choice_A"],
        "choice_B": item["choice_B"],
        "choice_C": item["choice_C"],
        "choice_D": item["choice_D"],
        "answer": item["answer"],
        "context": item["context"],
    }
    if "retrieved_context" in item:
        row["retrieved_context"] = item["retrieved_context"]
    return row


def load_data_all(data_json_path):
    """从本地文件加载 LongBench-v2，避免依赖 Hugging Face 在线下载。"""
    if data_json_path.endswith(".jsonl"):
        with open(data_json_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    else:
        with open(data_json_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and "train" in raw:
            items = raw["train"]
        else:
            raise ValueError(
                "本地 JSON 应为对象列表，或形如 {\"train\": [...] } 的字典"
            )
    return [_longbench_v2_row(x) for x in items]


def _resolve_data_path(args):
    """
    统一将 --data_path 解析为“具体文件路径”：
    - 传目录：默认使用 <dir>/data.json
    - 传文件：直接使用该文件
    """
    if not args.data_path:
        raise SystemExit("请显式传入 --data_path（data.json/.jsonl 文件或包含 data.json 的目录）")

    if os.path.isdir(args.data_path):
        return os.path.join(args.data_path, "data.json")
    return args.data_path


def _download_longbench_v2_data_json(dst_file):
    """
    直接把 LongBench-v2 的 data.json 下载到 dst_file，不走 HF 缓存目录。
    支持通过环境变量 HF_ENDPOINT 设置镜像站点（例如 https://hf-mirror.com）。
    """
    base = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    url = f"{base}/datasets/THUDM/LongBench-v2/resolve/main/data.json"
    os.makedirs(os.path.dirname(dst_file) or ".", exist_ok=True)
    print(f"[data] local file not found, downloading: {url} -> {dst_file}")
    try:
        urllib.request.urlretrieve(url, dst_file)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"下载失败：HTTP {e.code}，url={url}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"下载失败：网络错误，url={url} err={e}") from e
    print(f"[data] download complete: {dst_file}")


def _load_longbench_v2_data(args):
    data_file = _resolve_data_path(args)
    if not os.path.exists(data_file):
        _download_longbench_v2_data_json(data_file)
    else:
        print(f"[data] using local dataset file: {data_file}")
    return load_data_all(data_file)


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, fout):
    model = args.model
    if "gpt" in model or "o1" in model:
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    else:
        tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path(model), trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        # 预截断策略调整：
        # - 大部分超长来自 context，因此按 token 预算只裁剪 $DOC$（context）的头部
        # - 预算 = max_model_len - max_new_tokens - chat_template_buffer - 固定字段开销
        CHAT_TEMPLATE_BUFFER_TOKENS = 256
        max_model_len = int(maxlen_map[model])
        # CoT 第一段生成更长，因此 max_new_tokens 更大；第二段/非 CoT 用较小值
        max_new_tokens = 1024 if args.cot else 128
        max_prompt_len = max(256, max_model_len - int(max_new_tokens) - CHAT_TEMPLATE_BUFFER_TOKENS)
        try:
            prompt = build_prompt_truncating_context_head(
                template=template,
                context=context,
                question=item["question"],
                c_a=item["choice_A"],
                c_b=item["choice_B"],
                c_c=item["choice_C"],
                c_d=item["choice_D"],
                model_key=model,
                tokenizer=tokenizer,
                max_prompt_len_tokens=max_prompt_len,
            )
        except ValueError as e:
            _id = item.get("_id")
            print(f"[data] skip item due to prompt budget error: _id={_id} err={e}")
            continue
        if args.measure_latency:
            if args.cot:
                output, metrics = query_llm_streaming(
                    prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024
                )
            else:
                output, metrics = query_llm_streaming(
                    prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128
                )
        else:
            if args.cot:
                output, metrics = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024)
            else:
                output, metrics = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)

        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            if args.measure_latency:
                output, metrics2 = query_llm_streaming(
                    prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128
                )
            else:
                output, metrics2 = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
            # COT 模式下有两段请求：记录两段的 TTFT/E2E
            item["ttft_ms_cot"] = metrics.get("ttft_ms") if metrics else None
            item["e2e_ms_cot"] = metrics.get("e2e_ms") if metrics else None
            metrics = metrics2
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:256] + ' ...'
        if metrics:
            item["ttft_ms"] = metrics.get("ttft_ms")
            item["e2e_ms"] = metrics.get("e2e_ms")
            out_tok = _count_tokens_text(response, model, tokenizer)
            item["output_tokens"] = out_tok
            ttft = metrics.get("ttft_ms")
            e2e = metrics.get("e2e_ms")
            if isinstance(ttft, (int, float)) and isinstance(e2e, (int, float)) and out_tok > 0 and e2e >= ttft:
                item["tpot_ms"] = (e2e - ttft) / out_tok
            else:
                item["tpot_ms"] = None
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.out_file:
        out_name = args.out_file
        if not out_name.endswith(".jsonl"):
            out_name = out_name + ".jsonl"
        out_file = os.path.join(args.save_dir, out_name)
    else:
        name = args.model.split("/")[-1]
        suffix = f"_batch_{args.n_proc}"
        if args.rag > 0:
            name = name + f"_rag_{str(args.rag)}"
        elif args.no_context:
            name = name + "_no_context"
        elif args.cot:
            name = name + "_cot"
        out_file = os.path.join(args.save_dir, name + suffix + ".jsonl")

    data_all = _load_longbench_v2_data(args)

    # cache
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="GLM-4-9B-Chat")
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="可选：指定输出文件名（位于 --save_dir 下）。不传则按原逻辑自动生成；未写 .jsonl 会自动补齐。",
    )
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument(
        "--measure_latency",
        dest="measure_latency",
        action="store_true",
        help="启用 stream=True 统计 TTFT（首 chunk 延迟）与 E2E（端到端时延），并写入 results/*.jsonl",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="数据路径：可为 data.json/.jsonl 文件，或一个目录（将使用 <dir>/data.json）。若文件不存在会自动下载到此处。",
    )
    args = parser.parse_args()
    main()