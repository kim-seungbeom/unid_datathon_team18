"""Sequential by default; optional process parallelism for users with memory headroom."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extract_features import extract, parser


def run_all(
    *,
    csv_path,
    image_root,
    output_dir,
    batch_size=32,
    workers=4,
    device="cuda",
    l2_normalize=False,
    parallel=1,
):
    if parallel not in (1, 2, 3):
        raise ValueError("parallel must be 1, 2, or 3.")
    jobs = [
        dict(
            model_name=name,
            csv_path=csv_path,
            image_root=image_root,
            output_dir=Path(output_dir) / name,
            batch_size=batch_size,
            workers=workers,
            device=device,
            l2_normalize=l2_normalize,
        )
        for name in ("resnet18", "dinov2", "convnext_tiny")
    ]
    if any(job["output_dir"].exists() for job in jobs):
        raise FileExistsError("Choose a new output root; model outputs already exist.")
    if parallel == 1:
        return [extract(**job) for job in jobs]
    print(
        "Concurrent encoders share GPU memory. OOM stops the affected job; no automatic reruns.",
        flush=True,
    )
    results = []
    with ProcessPoolExecutor(
        max_workers=parallel, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(extract, **job): job["model_name"] for job in jobs}
        failures = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(futures[future])
                print(
                    f"{futures[future]} failed ({type(error).__name__}); "
                    "other outputs are preserved.",
                    flush=True,
                )
        if failures:
            raise RuntimeError("Some encoders failed; inspect their metadata.json locally.")
    return results


if __name__ == "__main__":
    p = parser("all")
    p.add_argument("--parallel", type=int, choices=(1, 2, 3), default=1)
    run_all(**vars(p.parse_args()))
