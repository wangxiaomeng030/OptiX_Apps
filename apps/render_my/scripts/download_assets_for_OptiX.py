#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import objaverse
import gzip
# from datasets import load_dataset  # 如需用 OPP 在线标注才需要；如遇 pandas/GLIBCXX 报错先注释
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

ASSETS_ROOT = Path(os.environ.get("ASSETS_ROOT", "/data/codes/optix_all/assets_store_new"))
OBJAVERSE_DIR = ASSETS_ROOT / "objaverse_opp_36k"
PBR_DIR = ASSETS_ROOT / "pbr"
HDRI_DIR = ASSETS_ROOT / "hdri"

# ---- 兜底下载地址（按顺序尝试；文件名是连字符）----
OBJECT_PATHS_URLS = [
    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/object-paths.json.gz",
    "https://storage.googleapis.com/objaverse/objaverse/object-paths.json.gz",
    "https://objaverse.allenai.org/object-paths.json.gz",
]

# ----------------------- common utils -----------------------

def ensure_dirs() -> None:
    OBJAVERSE_DIR.mkdir(parents=True, exist_ok=True)
    PBR_DIR.mkdir(parents=True, exist_ok=True)
    HDRI_DIR.mkdir(parents=True, exist_ok=True)


def read_list_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"List file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def place_file(src: Path, dst: Path, mode: str = "symlink") -> None:
    """
    mode: "symlink" | "hardlink" | "copy"
    - symlink: 省空间； - hardlink: 同盘共享数据； - copy: 复制一份
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        try:
            os.symlink(src, dst); return
        except OSError:
            try:
                os.link(src, dst); return
            except OSError:
                shutil.copy2(src, dst); return
    elif mode == "hardlink":
        try:
            os.link(src, dst); return
        except OSError:
            shutil.copy2(src, dst); return
    else:
        shutil.copy2(src, dst); return


def http_get(url: str, out_path: Path, max_retries: int = 3, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            mode = "wb"
            if out_path.exists():
                existing_size = out_path.stat().st_size
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"
                    mode = "ab"
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                total_header = r.headers.get("Content-Length")
                total = int(total_header) if total_header is not None else None
                desc = f"Downloading {out_path.name}"
                progress = tqdm(total=total, unit="B", unit_scale=True, desc=desc, disable=(total is None))
                with open(out_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if total is not None:
                                progress.update(len(chunk))
                progress.close()
            return
        except Exception as e:
            exc = e
            print(f"Attempt {attempt}/{max_retries} failed for {url}: {e}")
            time.sleep(2 * attempt)
    if exc:
        raise exc


def _is_gzip_json_valid(gz_path: Path) -> bool:
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            _ = json.load(f)
        return True
    except Exception:
        return False


def _download_clean_object_paths(dest_path: Path) -> None:
    # 逐个源尝试下载，直到成功
    tmp = dest_path.with_suffix(".tmp")
    for url in OBJECT_PATHS_URLS:
        try:
            print(f"[fetch] Trying object_paths from: {url}")
            http_get(url, tmp)
            if _is_gzip_json_valid(tmp):
                tmp.replace(dest_path)
                print(f"[fetch] Saved clean object-paths.json.gz to: {dest_path}")
                return
            else:
                print(f"[fetch] Downloaded but invalid gzip/json from {url}, will try next source.")
                tmp.unlink(missing_ok=True)
        except Exception as e:
            print(f"[fetch] Failed from {url}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Failed to fetch a valid object-paths.json.gz from all known sources.")


def _candidate_object_paths(cache_root: Path) -> List[Path]:
    """归纳所有可能的 object-paths/object_paths.json.gz 候选位置（含包内与常见缓存目录）"""
    names = ["object-paths.json.gz", "object_paths.json.gz"]  # 同时兼容连字符与下划线
    cands: List[Path] = []
    # 1) cache_root
    if cache_root:
        try:
            for n in names:
                cands += list(Path(cache_root).rglob(n))
        except Exception:
            pass
        for n in names:
            cands.append(Path(cache_root) / n)
    # 2) 用户目录缓存
    home = Path.home()
    for root in [home / ".objaverse", home / ".cache", home / ".cache" / "objaverse"]:
        if root.exists():
            try:
                for n in names:
                    cands += list(root.rglob(n))
            except Exception:
                pass
    # 3) 包内
    try:
        pkg_dir = Path(objaverse.__file__).parent
        for n in names:
            cands += list(pkg_dir.rglob(n))
    except Exception:
        pass
    # 去重
    seen = set(); uniq: List[Path] = []
    for p in cands:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def validate_repair_and_patch_objaverse(cache_root: Path) -> Path:
    """
    1) 扫描候选路径，删除损坏的 object-*.json.gz
    2) 如无有效文件，则下载一份干净的到 cache_root/object-paths.json.gz
    3) monkey-patch objaverse._load_object_paths 读取该文件
    """
    if cache_root is None:
        cache_root = Path.cwd()
    cache_root.mkdir(parents=True, exist_ok=True)

    candidates = _candidate_object_paths(cache_root)

    valid_file: Optional[Path] = None
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                if _is_gzip_json_valid(p):
                    if valid_file is None:
                        valid_file = p
                else:
                    print(f"[repair] Removing corrupted object_paths: {p}")
                    p.unlink(missing_ok=True)
        except Exception as e:
            print(f"[repair] Failed when checking {p}: {e}")

    target = Path(cache_root) / "object-paths.json.gz"  # 统一到连字符命名
    if valid_file is None:
        print("[repair] No valid object-paths.json.gz found, fetching a clean copy...")
        _download_clean_object_paths(target)
        valid_file = target
    else:
        if valid_file != target:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                try:
                    os.link(valid_file, target)
                except OSError:
                    shutil.copy2(valid_file, target)
            except Exception as e:
                print(f"[repair] Failed to mirror object_paths into cache root: {e}")

    # patch 读取
    def _patched_load_object_paths():
        with gzip.open(target, "rt", encoding="utf-8") as f:
            return json.load(f)
    try:
        objaverse._load_object_paths = _patched_load_object_paths  # type: ignore[attr-defined]
        print(f"[patch] objaverse._load_object_paths patched to use: {target}")
    except Exception as e:
        print(f"[patch] Failed to patch objaverse loader: {e}")

    return target


# ----------------------- Objaverse LVIS -----------------------

def gather_lvis_uids() -> List[str]:
    ann = objaverse.load_lvis_annotations()  # Dict[str, List[str]]
    uids_set = set()
    for _, uid_list in ann.items():
        uids_set.update(uid_list)
    return sorted(uids_set)


def scan_existing_objaverse(out_dir: Path) -> List[str]:
    if not out_dir.exists():
        return []
    return [p.stem for p in out_dir.glob("*.glb")]


def download_objaverse_lvis_batched(
    uids: Optional[List[str]],
    out_dir: Path,
    target_count: Optional[int] = None,
    batch_size: int = 500,
    processes: int = 1,
    cache_dir: Optional[Path] = None,
    link_mode: str = "symlink",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = out_dir

    # 把 HF/TMP 指到大盘（HF 的其它缓存也不占家目录）
    os.environ["OBJAVERSE_CACHE_DIR"] = str(cache_dir)
    os.environ["OBJAVERSE_DATA_DIR"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    tmpdir = Path(cache_dir) / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)

    # ✅ 关键：把 Objaverse 的资产根目录强制改到 cache_dir（否则会落到 ~/.objaverse）
    objaverse.BASE_PATH = str(Path(cache_dir))
    objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")
    Path(objaverse._VERSIONED_PATH).mkdir(parents=True, exist_ok=True)
    print("[objaverse paths]", objaverse.BASE_PATH, "->", objaverse._VERSIONED_PATH)

    if uids is None:
        uids = gather_lvis_uids()

    existing = set(scan_existing_objaverse(out_dir))
    uids = [u for u in uids if u not in existing]
    if target_count is not None:
        uids = uids[: max(0, target_count - len(existing))]

    print(f"Starting LVIS download: need {len(uids)} new of target {target_count or 'ALL'} (existing {len(existing)})")
    print(f"- cache_dir : {cache_dir}")
    print(f"- out_dir   : {out_dir}")
    print(f"- link_mode : {link_mode}")

    linked: Dict[str, str] = {}
    for i in range(0, len(uids), batch_size):
        batch = uids[i : i + batch_size]
        print(f"Batch {i//batch_size + 1}: {len(batch)} objects")

        # ✅ 先确保索引可用 & patch 读取
        valid_idx = validate_repair_and_patch_objaverse(Path(cache_dir))
        print(f"[index] Using object_paths: {valid_idx}")

        # ✅ 再做一次稳妥重试，兜住偶发 IO/网络
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                uid_to_local: Dict[str, str] = objaverse.load_objects(
                    uids=batch, download_processes=processes,
                )
                break
            except (EOFError, OSError, json.JSONDecodeError) as e:
                last_exc = e
                print(f"[warn] objaverse.load_objects failed (attempt {attempt}/3): {e}")
                validate_repair_and_patch_objaverse(Path(cache_dir))
                time.sleep(min(2 * attempt, 5))
        else:
            raise RuntimeError(f"objaverse.load_objects repeatedly failed after repairs: {last_exc}")

        for uid, local_path in uid_to_local.items():
            src = Path(local_path)
            dst = out_dir / f"{uid}.glb"
            place_file(src, dst, mode=link_mode)
            linked[uid] = str(dst)

        with (out_dir / "_progress.json").open("w", encoding="utf-8") as f:
            json.dump({"completed": len(existing) + len(linked)}, f, indent=2)

    with (out_dir / "download_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "existing": len(existing),
            "new_linked": len(linked),
            "total": len(existing) + len(linked),
        }, f, indent=2)
    print(f"Done LVIS. Total linked: {len(existing) + len(linked)}")


# ----------------------- Objaverse++ annotations filter（可选依赖 datasets） -----------------------

def load_opp_annotations(cache_dir: Optional[str] = None):
    from datasets import load_dataset  # 局部导入避免顶层报错
    ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train", cache_dir=cache_dir)
    return ds


def filter_opp_uids(
    ds,
    min_score: int = 2,
    styles: Optional[List[str]] = None,
    must_have_texture: Optional[bool] = None,
    allow_multi_object: Optional[bool] = None,
    allow_scene: Optional[bool] = None,
    density_in: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[str]:
    f = ds
    f = f.filter(lambda ex: int(ex.get("score", 0)) >= int(min_score))
    if styles:
        style_set = set(s.lower() for s in styles)
        f = f.filter(lambda ex: str(ex.get("style", "")).lower() in style_set)

    def _to_bool_str(x):
        return str(x).lower()

    if allow_multi_object is not None:
        flag = "true" if allow_multi_object else "false"
        f = f.filter(lambda ex: _to_bool_str(ex.get("is_multi_object", "false")) == flag)

    if allow_scene is not None:
        flag = "true" if allow_scene else "false"
        f = f.filter(lambda ex: _to_bool_str(ex.get("is_scene", "false")) == flag)

    if must_have_texture is True:
        f = f.filter(lambda ex: _to_bool_str(ex.get("is_single_color", "false")) == "false")

    if density_in:
        dens = set(density_in)
        f = f.filter(lambda ex: ex.get("density") in dens)

    uids = [row["UID"] for row in f]
    if limit is not None:
        uids = uids[: int(limit)]
    return uids


# ----------------------- Poly Haven listing -----------------------

def list_polyhaven_assets(asset_type: str) -> List[str]:
    url = f"https://api.polyhaven.com/assets?t={asset_type}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if asset_type == "textures":
        ids = [k for k, v in data.items() if isinstance(v, dict) and v.get("type") == 1]
    elif asset_type == "hdris":
        ids = [k for k, v in data.items() if isinstance(v, dict) and v.get("type") == 0]
    else:
        ids = sorted(list(data.keys()))
    return sorted(ids)


# ----------------------- Poly Haven PBR -----------------------

def pick_resolution(available: Dict[str, dict], preferred: str) -> str:
    if preferred in available:
        return preferred
    def res_key_to_int(r: str) -> int:
        try:
            return int(r.replace("k", ""))
        except Exception:
            return 0
    return sorted(available.keys(), key=res_key_to_int, reverse=True)[0]


def pick_format(file_entry: Dict[str, dict], preferred_exts: Tuple[str, ...]) -> Tuple[str, Dict[str, str]]:
    for ext in preferred_exts:
        if ext in file_entry:
            return ext, file_entry[ext]
    for ext, meta in file_entry.items():
        return ext, meta
    raise ValueError("No files available in entry")


def pbr_asset_done(tex_dir: Path) -> bool:
    return (tex_dir / "_meta.json").exists()


def download_one_pbr(name: str, res: str, out_dir: Path, prefer_png: bool = True) -> Tuple[str, bool, str]:
    try:
        info_url = f"https://api.polyhaven.com/files/{name}"
        resp = requests.get(info_url, timeout=30)
        if resp.status_code != 200:
            return name, False, f"HTTP {resp.status_code}"
        data = resp.json()
        maps_downloaded = []
        tex_dir = out_dir / name
        if pbr_asset_done(tex_dir):
            return name, True, "skipped (exists)"
        for map_key, res_dict in data.items():
            if not isinstance(res_dict, dict):
                continue
            if not res_dict:
                continue
            chosen_res = pick_resolution(res_dict, res)
            file_entry = res_dict[chosen_res]
            preferred_exts = ("png", "jpg", "exr") if prefer_png else ("jpg", "png", "exr")
            ext, meta = pick_format(file_entry, preferred_exts)
            url = meta.get("url")
            if not url:
                continue
            filename = url.split("/")[-1]
            out_path = tex_dir / filename
            http_get(url, out_path)
            maps_downloaded.append({"map": map_key, "res": chosen_res, "ext": ext, "path": str(out_path)})
        with (tex_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump({"name": name, "maps": maps_downloaded}, f, indent=2)
        return name, True, f"{len(maps_downloaded)} maps"
    except Exception as e:
        return name, False, str(e)


def download_polyhaven_pbr(
    names: List[str],
    res: str,
    out_dir: Path,
    prefer_png: bool = True,
    workers: int = 8,
    resume: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for name in names:
            if resume and pbr_asset_done(out_dir / name):
                print(f"[skip] {name} (already has _meta.json)")
                continue
            tasks.append(ex.submit(download_one_pbr, name, res, out_dir, prefer_png))
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="PBR"):
            name, ok, msg = fut.result()
            status = "ok" if ok else "fail"
            print(f"[{status}] {name}: {msg}")


# ----------------------- Poly Haven HDRI -----------------------

def hdri_asset_done(hdri_dir: Path) -> bool:
    return (hdri_dir / "_meta.json").exists()


def download_one_hdri(
    name: str,
    res: str,
    out_dir: Path,
    fmt: str = "exr",
    no_meta: bool = False,
    flatten: bool = False,
) -> Tuple[str, bool, str]:
    try:
        info_url = f"https://api.polyhaven.com/files/{name}"
        resp = requests.get(info_url, timeout=30)
        if resp.status_code != 200:
            return name, False, f"HTTP {resp.status_code}"
        data = resp.json()
        hdri_section = data.get("hdri")
        if not isinstance(hdri_section, dict) or not hdri_section:
            return name, False, "no hdri section"
        chosen_res = pick_resolution(hdri_section, res)
        file_entry = hdri_section[chosen_res]
        chosen_fmt = fmt if fmt in file_entry else ("hdr" if fmt == "exr" else "exr")
        if chosen_fmt not in file_entry:
            return name, False, f"no file for {chosen_res} in exr/hdr"
        url = file_entry[chosen_fmt]["url"]
        filename = url.split("/")[-1]
        out_path = (out_dir / filename) if flatten else (out_dir / name / filename)
        if flatten or no_meta:
            if out_path.exists():
                return name, True, "skipped (exists)"
        else:
            if hdri_asset_done(out_path.parent):
                return name, True, "skipped (exists)"
        http_get(url, out_path)
        if not no_meta:
            if flatten:
                meta_path = out_path.with_suffix(out_path.suffix + ".json")
            else:
                meta_path = out_path.parent / "_meta.json"
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump({"name": name, "file": str(out_path), "res": chosen_res, "fmt": chosen_fmt}, f, indent=2)
        return name, True, f"{chosen_res} {chosen_fmt}"
    except Exception as e:
        return name, False, str(e)


def download_polyhaven_hdri(
    names: List[str],
    res: str,
    out_dir: Path,
    fmt: str = "exr",
    workers: int = 8,
    resume: bool = True,
    no_meta: bool = False,
    flatten: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for name in names:
            if resume:
                if flatten or no_meta:
                    pass
                else:
                    if hdri_asset_done(out_dir / name):
                        print(f"[skip] {name} (already has _meta.json)")
                        continue
            tasks.append(ex.submit(download_one_hdri, name, res, out_dir, fmt, no_meta, flatten))
        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="HDRI"):
            name, ok, msg = fut.result()
            status = "ok" if ok else "fail"
            print(f"[{status}] {name}: {msg}")


# ----------------------- CLI -----------------------

def cli() -> None:
    parser = argparse.ArgumentParser(description="Asset downloader: Objaverse via LVIS/OPP, PBR materials, HDRI maps")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Objaverse
    p_obj = subparsers.add_parser("objaverse", help="Download Objaverse LVIS subset or filtered by Objaverse++")
    p_obj.add_argument("--uids-file", type=str, default=None, help="Optional file containing UIDs (one per line)")
    p_obj.add_argument("--out", type=str, default=str(OBJAVERSE_DIR), help="Output directory for files")
    p_obj.add_argument("--count", type=int, default=None, help="Target number of objects to download (after skipping existing)")
    p_obj.add_argument("--batch-size", type=int, default=500, help="Batch size for LVIS downloads")
    p_obj.add_argument("--proc", type=int, default=1, help="Download processes for objaverse.load_objects")
    p_obj.add_argument("--cache-dir", type=str, default=None, help="Cache/download dir for Objaverse/HF/TMP. Default: use --out")
    p_obj.add_argument("--link-mode", choices=["symlink","hardlink","copy"], default="symlink",
                       help="How to place files into out dir")

    # OPP（如需在线 datasets 请解开顶层导入，并确保环境满足 pandas 依赖）
    p_obj.add_argument("--opp", action="store_true", help="Use Objaverse++ annotations to filter UIDs")
    p_obj.add_argument("--opp-cache", type=str, default=None, help="Hugging Face datasets cache dir")
    p_obj.add_argument("--min-score", type=int, default=2, help="Quality score threshold 0..3")
    p_obj.add_argument("--styles", type=str, nargs="*", default=None, help="Filter by style")
    p_obj.add_argument("--must-texture", action="store_true", help="Keep only non-single-color objects")
    p_obj.add_argument("--allow-multi", action="store_true", help="Allow multi-object")
    p_obj.add_argument("--allow-scene", action="store_true", help="Allow scene")
    p_obj.add_argument("--density", type=str, nargs="*", default=None, help="Filter geometry density")
    p_obj.add_argument("--limit", type=int, default=None, help="Take only first N UIDs after filtering")

    # PBR
    p_pbr = subparsers.add_parser("pbr", help="Download PBR materials (Poly Haven)")
    p_pbr.add_argument("--names", type=str, nargs="*", default=None)
    p_pbr.add_argument("--names-file", type=str, default=None)
    p_pbr.add_argument("--auto", type=int, default=None)
    p_pbr.add_argument("--all", action="store_true")
    p_pbr.add_argument("--out", type=str, default=str(PBR_DIR))
    p_pbr.add_argument("--res", type=str, default="2k")
    p_pbr.add_argument("--no-png", action="store_true")
    p_pbr.add_argument("--workers", type=int, default=8)
    p_pbr.add_argument("--no-resume", action="store_true")

    # HDRI
    p_hdr = subparsers.add_parser("hdri", help="Download HDRI environment maps (Poly Haven)")
    p_hdr.add_argument("--names", type=str, nargs="*", default=None)
    p_hdr.add_argument("--names-file", type=str, default=None)
    p_hdr.add_argument("--auto", type=int, default=None)
    p_hdr.add_argument("--all", action="store_true")
    p_hdr.add_argument("--out", type=str, default=str(HDRI_DIR))
    p_hdr.add_argument("--res", type=str, default="2k")
    p_hdr.add_argument("--fmt", type=str, default="exr")
    p_hdr.add_argument("--workers", type=int, default=8)
    p_hdr.add_argument("--no-resume", action="store_true")
    p_hdr.add_argument("--no-meta", action="store_true")
    p_hdr.add_argument("--flatten", action="store_true")

    args = parser.parse_args()
    ensure_dirs()

    if args.cmd == "objaverse":
        out_dir = Path(args.out)
        cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir  # 默认就用 out_dir 当缓存/下载根
        link_mode = args.link_mode

        # 选择 UID 列表
        uids = read_list_file(args.uids_file)
        if uids is None and args.opp:
            print("[opp] loading Objaverse++ annotations from Hugging Face ...")
            ds = load_opp_annotations(cache_dir=args.opp_cache)
            uids = filter_opp_uids(
                ds,
                min_score=args.min_score,
                styles=args.styles,
                must_have_texture=args.must_texture,
                allow_multi_object=args.allow_multi,
                allow_scene=args.allow_scene,
                density_in=args.density,
                limit=args.limit,
            )
            print(f"[opp] selected {len(uids)} UIDs (after filters)")
            if not uids:
                raise SystemExit("No UIDs selected by Objaverse++ filters. Adjust filters and retry.")

        if uids is None:
            print("[lvis] using LVIS subset annotations")
            uids = gather_lvis_uids()
            if args.count is not None:
                uids = uids[: args.count]

        # ✅ 预检 + 兜底下载 + patch（索引）
        validate_repair_and_patch_objaverse(cache_dir)

        download_objaverse_lvis_batched(
            uids=uids,
            out_dir=out_dir,
            target_count=args.count if not args.opp else None,
            batch_size=args.batch_size,
            processes=args.proc,
            cache_dir=cache_dir,
            link_mode=link_mode,
        )

    elif args.cmd == "pbr":
        name_list = args.names or []
        file_names = read_list_file(args.names_file)
        if file_names:
            name_list.extend(file_names)
        if args.all or args.auto:
            api_list = list_polyhaven_assets("textures")
            if args.all:
                name_list = api_list
            elif args.auto:
                name_list = api_list[: args.auto]
        if not name_list:
            raise SystemExit("No PBR asset names provided. Use --names/--names-file or --auto/--all")
        download_polyhaven_pbr(
            names=name_list,
            res=args.res,
            out_dir=Path(args.out),
            prefer_png=(not args.no_png),
            workers=args.workers,
            resume=(not args.no_resume),
        )

    elif args.cmd == "hdri":
        name_list = args.names or []
        file_names = read_list_file(args.names_file)
        if file_names:
            name_list.extend(file_names)
        if args.all or args.auto:
            api_list = list_polyhaven_assets("hdris")
            if args.all:
                name_list = api_list
            elif args.auto:
                name_list = api_list[: args.auto]
        if not name_list:
            raise SystemExit("No HDRI asset names provided. Use --names/--names-file or --auto/--all")
        download_polyhaven_hdri(
            names=name_list,
            res=args.res,
            out_dir=Path(args.out),
            fmt=args.fmt,
            workers=args.workers,
            resume=(not args.no_resume),
            no_meta=args.no_meta,
            flatten=args.flatten,
        )
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    cli()

