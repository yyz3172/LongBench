import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 数据集仓库到本地目录（方式A：snapshot_download）。"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="THUDM/LongBench-v2",
        help="Hugging Face dataset repo id，如 THUDM/LongBench-v2",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/autodl-tmp/datasets/LongBench-v2",
        help="下载到本地的目录（会创建）",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("HF_ENDPOINT", ""),
        help="可选：镜像站点，如 https://hf-mirror.com（也可用环境变量 HF_ENDPOINT）",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="可选：指定分支/commit/tag。默认使用仓库默认 revision",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="可选：私有仓库才需要。默认读取环境变量 HF_TOKEN",
    )
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(
            "缺少依赖 huggingface_hub，请先安装：pip install -U huggingface_hub\n"
            f"原始错误：{e}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.out_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
    )
    print("downloaded_to:", local_path)


if __name__ == "__main__":
    main()

