import os, csv, json
import argparse
import time
import urllib.request
import urllib.error
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

URL = "http://127.0.0.1:8000/v1"
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

    def _encode(text):
        if model in model_map:
            return tokenizer.encode(text)
        return tokenizer.encode(text, disallowed_special=())

    def _decode(ids):
        if model in model_map:
            return tokenizer.decode(ids, skip_special_tokens=True)
        return tokenizer.decode(ids)

    def _truncate(text, limit):
        input_ids = _encode(text)
        if len(input_ids) <= limit:
            return text
        # 保留头尾，适合 LongBench 的长上下文
        keep = max(limit, 0)
        half = keep // 2
        input_ids = input_ids[:half] + input_ids[-(keep - half):]
        return _decode(input_ids)

    # truncate（以本地配置为上限；预留 chat template 余量；若服务端更小，后面会自动适配）
    max_len = max(256, maxlen_map[model] - CHAT_TEMPLATE_BUFFER_TOKENS)
    prompt = _truncate(prompt, max_len)
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
            )
            return completion.choices[0].message.content
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
            # vLLM 常见报错：最大上下文长度不足（400），自动把 prompt 截断到服务端上限再重试
            # 例如：This model's maximum context length is 8192 tokens. However, your request has 120009 input tokens.
            err_text = body_text or msg
            m = re.search(r"maximum context length is\s+(\d+)\s+tokens", err_text)
            if m:
                server_limit = int(m.group(1))
                # 预留 chat template 余量，避免再次越界（无需扣 max_new_tokens：这里只影响 input tokens）
                new_limit = max(256, server_limit - CHAT_TEMPLATE_BUFFER_TOKENS)
                if new_limit < max_len:
                    max_len = new_limit
                    prompt = _truncate(prompt, max_len)
                    print(
                        f"[data] server max context={server_limit}, "
                        f"auto-truncate to {max_len} (buffer={CHAT_TEMPLATE_BUFFER_TOKENS}) and retry"
                    )
                    continue
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

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
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=1024)
        else:
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + ".jsonl")

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
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="数据路径：可为 data.json/.jsonl 文件，或一个目录（将使用 <dir>/data.json）。若文件不存在会自动下载到此处。",
    )
    args = parser.parse_args()
    main()