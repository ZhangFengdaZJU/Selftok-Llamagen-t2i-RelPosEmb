from pathlib import Path
import shutil

SRC_ROOT = Path("/home/jovyan/zfd/LlamaGen/output/t5_embs_out")
DEST = Path("/home/jovyan/zfd/LlamaGen/output/t5_embs_merged")
MODE = "skip"       # 可选: "skip" | "overwrite" | "rename"

def copy_all(src_root: Path, dest: Path, mode: str = "skip"):
    assert mode in {"skip", "overwrite", "rename"}
    dest.mkdir(parents=True, exist_ok=True)

    # 找到 train-*-of-00207 这些子目录
    dirs = sorted([p for p in src_root.glob("train-*-of-00207") if p.is_dir()])
    if not dirs:
        raise RuntimeError(f"没有找到目录: {src_root}/train-*-of-00207")

    copied = 0
    skipped = 0
    renamed = 0

    for d in dirs:
        # 遍历目录下所有文件（含子目录）
        for f in d.rglob("*"):
            if not f.is_file():
                continue

            target = dest / f.name

            if mode == "skip":
                # 不覆盖
                if target.exists():
                    skipped += 1
                    continue
                shutil.copy2(f, target)
                copied += 1

            elif mode == "overwrite":
                # 覆盖
                shutil.copy2(f, target)
                copied += 1

            else:  # mode == "rename"
                out = target
                if out.exists():
                    # 在重名时用“源目录名_原文件名”
                    out = dest / f"{d.name}_{f.name}"
                    # 再次碰到重名就加递增后缀
                    i = 1
                    stem = out.stem
                    suffix = out.suffix
                    while out.exists():
                        out = dest / f"{stem}_{i}{suffix}"
                        i += 1
                    renamed += 1
                shutil.copy2(f, out)
                copied += 1

    print(f"完成：复制 {copied} 个文件；跳过 {skipped}；重命名 {renamed}。")
    print(f"输出目录：{dest}")

if __name__ == "__main__":
    copy_all(SRC_ROOT, DEST, MODE)
