import argparse
import os
import shutil
import tempfile
import traceback
from multiprocessing import cpu_count
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_faiss_index(index, target_path: Path) -> None:
    """Write FAISS index; fallback to ASCII temp path for Windows Unicode paths."""
    import faiss

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        faiss.write_index(index, str(target_path))
        return
    except RuntimeError:
        pass

    fd, tmp_path = tempfile.mkstemp(suffix=".index")
    os.close(fd)
    try:
        faiss.write_index(index, tmp_path)
        shutil.copy2(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def feature_dir_for(exp_dir: Path, version: str) -> Path:
    return exp_dir / ("3_feature256" if version == "v1" else "3_feature768")


def build_index(exp_name: str, version: str, outside_index_root: str = "assets/indices") -> Path:
    import faiss
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    os.chdir(ROOT)
    exp_dir = ROOT / "logs" / exp_name
    feature_dir = feature_dir_for(exp_dir, version)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    feature_files = sorted(path for path in feature_dir.iterdir() if path.suffix == ".npy")
    if not feature_files:
        raise RuntimeError(f"No feature .npy files found in: {feature_dir}")

    npys = [np.load(path) for path in feature_files]
    big_npy = np.concatenate(npys, axis=0)
    big_npy = big_npy[np.random.permutation(big_npy.shape[0])]

    n_cpu = cpu_count()
    if big_npy.shape[0] > 2e5:
        print(f"Trying k-means: {big_npy.shape[0]} frames -> 10000 centers")
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            print(traceback.format_exc())

    np.save(exp_dir / "total_fea.npy", big_npy)

    dim = 256 if version == "v1" else 768
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(1, n_ivf)
    print(f"Index source shape: {big_npy.shape}, n_ivf={n_ivf}")

    index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1

    print("Training index...")
    index.train(big_npy)
    trained_path = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
    write_faiss_index(index, trained_path)

    print("Adding vectors...")
    batch_size_add = 8192
    for start in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[start : start + batch_size_add])

    added_path = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
    write_faiss_index(index, added_path)

    outside_dir = ROOT / outside_index_root
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_path = outside_dir / f"{exp_name}_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
    try:
        if outside_path.exists() or outside_path.is_symlink():
            outside_path.unlink()
        os.symlink(added_path, outside_path)
    except OSError:
        shutil.copy2(added_path, outside_path)

    print(f"Index written: {added_path}")
    print(f"External index link/copy: {outside_path}")
    return added_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an RVC FAISS retrieval index.")
    parser.add_argument("-e", "--exp-name", required=True, help="Experiment name under logs/")
    parser.add_argument("-v", "--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--outside-index-root", default="assets/indices")
    args = parser.parse_args()
    build_index(args.exp_name, args.version, args.outside_index_root)


if __name__ == "__main__":
    main()
