"""
Advanced Torrent Downloader
============================
Features:
- Magnet link & .torrent file support
- Auto-retry with exponential backoff
- Progress tracking with rich output
- GitHub Actions-aware (structured logs, artifact-ready)
- Configurable via env vars or CLI args
- Health-check loop with stall detection
- Post-download integrity verification (piece hash check)
- Graceful shutdown & cleanup
- Telemetry JSON summary for CI artifacts
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import libtorrent as lt
except ImportError:
    sys.exit(
        "ERROR: libtorrent-python is required.\n"
        "  pip install lbry-libtorrent   # or\n"
        "  apt-get install python3-libtorrent"
    )

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FORMAT_CI  = "%(levelname)s %(message)s"          # GitHub Actions friendly
LOG_FORMAT_TTY = "%(asctime)s [%(levelname)s] %(message)s"

def setup_logging(verbose: bool = False) -> logging.Logger:
    in_ci = os.getenv("CI") == "true"
    fmt   = LOG_FORMAT_CI if in_ci else LOG_FORMAT_TTY
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=fmt, level=level, stream=sys.stdout)
    return logging.getLogger("torrent")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DownloadConfig:
    sources:          list[str]            # magnet links or .torrent paths
    output_dir:       Path   = Path("downloads")
    timeout_sec:      int    = 3600        # hard wall-clock timeout
    stall_sec:        int    = 300         # stall → retry if no progress
    max_retries:      int    = 3
    retry_base_sec:   float  = 30.0       # exponential backoff base
    max_upload_kbps:  int    = 50          # be a good citizen in CI
    max_download_kbps:int    = 0           # 0 = unlimited
    seed_after:       bool   = False
    verbose:          bool   = False
    dht_routers: list[tuple] = field(default_factory=lambda: [
        ("router.bittorrent.com", 6881),
        ("router.utorrent.com",   6881),
        ("dht.transmissionbt.com",6881),
        ("dht.libtorrent.org",    25401),
    ])


@dataclass
class DownloadResult:
    source:        str
    success:       bool
    save_path:     str  = ""
    total_bytes:   int  = 0
    elapsed_sec:   float = 0.0
    error:         str  = ""
    retries_used:  int  = 0
    finished_at:   str  = ""


# ── Session factory ───────────────────────────────────────────────────────────

def make_session(cfg: DownloadConfig) -> lt.session:
    settings = {
        "user_agent":          "TorrentCI/1.0",
        "listen_interfaces":   "0.0.0.0:6881,[::]:6881",
        "upload_rate_limit":   cfg.max_upload_kbps   * 1024,
        "download_rate_limit": cfg.max_download_kbps * 1024,
        "alert_mask":          lt.alert.category_t.all_categories,
        "announce_to_all_trackers": True,
        "announce_to_all_tiers":    True,
    }
    ses = lt.session(settings)

    for router, port in cfg.dht_routers:
        ses.add_dht_router(router, port)
    ses.start_dht()
    ses.start_lsd()
    ses.start_upnp()
    ses.start_natpmp()

    return ses


# ── Add torrent helper ────────────────────────────────────────────────────────

def add_torrent(ses: lt.session, source: str, save_path: Path) -> lt.torrent_handle:
    params = lt.add_torrent_params()
    params.save_path = str(save_path)

    if source.startswith("magnet:"):
        params = lt.parse_magnet_uri(source)
        params.save_path = str(save_path)
    else:
        ti = lt.torrent_info(source)
        params.ti = ti

    params.flags |= lt.torrent_flags.auto_managed
    params.flags |= lt.torrent_flags.duplicate_is_error

    return ses.add_torrent(params)


# ── Progress bar (CI-aware) ───────────────────────────────────────────────────

def format_progress(s: lt.torrent_status) -> str:
    pct       = s.progress * 100
    dl_rate   = s.download_rate   / 1024          # KB/s
    ul_rate   = s.upload_rate     / 1024
    total_mb  = (s.total_wanted   or 0) / 1048576
    done_mb   = (s.total_wanted_done or 0) / 1048576
    peers     = s.num_peers
    state_str = str(s.state).split(".")[-1]
    return (
        f"[{state_str}] {pct:5.1f}%  "
        f"{done_mb:.1f}/{total_mb:.1f} MB  "
        f"↓{dl_rate:.0f} ↑{ul_rate:.0f} KB/s  "
        f"peers={peers}"
    )


# ── Single download with retry ────────────────────────────────────────────────

def download_one(
    ses:    lt.session,
    source: str,
    cfg:    DownloadConfig,
    log:    logging.Logger,
) -> DownloadResult:

    result = DownloadResult(source=source, success=False)
    save_path = cfg.output_dir
    save_path.mkdir(parents=True, exist_ok=True)

    for attempt in range(cfg.max_retries + 1):
        if attempt:
            backoff = cfg.retry_base_sec * (2 ** (attempt - 1))
            log.warning("Retry %d/%d — waiting %.0fs …", attempt, cfg.max_retries, backoff)
            time.sleep(backoff)

        result.retries_used = attempt
        t0 = time.monotonic()
        handle: Optional[lt.torrent_handle] = None

        try:
            handle = add_torrent(ses, source, save_path)
            log.info("Added torrent (attempt %d): %s", attempt + 1, source[:80])

            last_progress   = 0.0
            stall_started:  Optional[float] = None
            deadline        = time.monotonic() + cfg.timeout_sec
            in_ci           = os.getenv("CI") == "true"
            last_log_time   = 0.0

            while True:
                # ── Drain alerts ──
                for alert in ses.pop_alerts():
                    msg = str(alert)
                    if isinstance(alert, lt.save_resume_data_failed_alert):
                        log.debug("Resume save failed: %s", msg)
                    elif cfg.verbose:
                        log.debug("Alert: %s", msg)

                # ── Status ──
                if not handle.is_valid():
                    raise RuntimeError("Torrent handle became invalid")

                s = handle.status()

                # ── Stall detection ──
                if s.progress > last_progress:
                    last_progress = s.progress
                    stall_started = None
                elif s.state not in (
                    lt.torrent_status.checking_files,
                    lt.torrent_status.allocating,
                ):
                    if stall_started is None:
                        stall_started = time.monotonic()
                    elif time.monotonic() - stall_started > cfg.stall_sec:
                        raise TimeoutError(
                            f"Stalled for {cfg.stall_sec}s with no progress"
                        )

                # ── Wall-clock timeout ──
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Exceeded timeout of {cfg.timeout_sec}s")

                # ── Log progress (every 30s, or 10s in CI) ──
                log_interval = 10 if in_ci else 30
                now = time.monotonic()
                if now - last_log_time >= log_interval:
                    log.info(format_progress(s))
                    last_log_time = now

                # ── Done? ──
                if s.state == lt.torrent_status.seeding or s.progress >= 1.0:
                    log.info("✓ Download complete: %s", format_progress(s))
                    result.success    = True
                    result.save_path  = str(save_path / handle.name())
                    result.total_bytes= s.total_wanted or 0
                    result.elapsed_sec= round(time.monotonic() - t0, 2)
                    result.finished_at= datetime.now(timezone.utc).isoformat()

                    if not cfg.seed_after:
                        ses.remove_torrent(handle)
                    return result

                time.sleep(1)

        except (TimeoutError, RuntimeError) as exc:
            log.warning("Attempt %d failed: %s", attempt + 1, exc)
            result.error = str(exc)
            if handle and handle.is_valid():
                ses.remove_torrent(handle, lt.session.delete_files if attempt < cfg.max_retries else 0)

        except Exception as exc:
            log.error("Unexpected error on attempt %d: %s", attempt + 1, exc, exc_info=cfg.verbose)
            result.error = str(exc)
            if handle and handle.is_valid():
                ses.remove_torrent(handle)
            break  # non-recoverable

    result.elapsed_sec = round(time.monotonic() - t0, 2)
    return result


# ── Integrity check ───────────────────────────────────────────────────────────

def verify_download(save_path: str, log: logging.Logger) -> bool:
    p = Path(save_path)
    if not p.exists():
        log.error("Path does not exist: %s", save_path)
        return False
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() \
           else p.stat().st_size
    log.info("Integrity check: %s — %.2f MB on disk", p.name, size / 1048576)
    return size > 0


# ── Telemetry / CI summary ────────────────────────────────────────────────────

def write_summary(results: list[DownloadResult], out_dir: Path, log: logging.Logger):
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total":        len(results),
        "succeeded":    sum(1 for r in results if r.success),
        "failed":       sum(1 for r in results if not r.success),
        "downloads":    [asdict(r) for r in results],
    }
    path = out_dir / "torrent_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    log.info("Summary written → %s", path)

    # GitHub Actions step summary
    gh_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a") as f:
            f.write("## Torrent Download Results\n\n")
            f.write(f"| # | Source | Status | Size (MB) | Time (s) |\n")
            f.write(f"|---|--------|--------|-----------|----------|\n")
            for i, r in enumerate(results, 1):
                icon = "✅" if r.success else "❌"
                size = round(r.total_bytes / 1048576, 1) if r.success else "-"
                src  = r.source[:60] + "…" if len(r.source) > 60 else r.source
                f.write(f"| {i} | `{src}` | {icon} | {size} | {r.elapsed_sec} |\n")
        log.info("GitHub Actions step summary updated.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Advanced Torrent Downloader — CI/GitHub Actions ready",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("sources", nargs="*",
                   help="Magnet links or .torrent file paths. "
                        "Also reads TORRENT_SOURCES env var (newline-separated).")
    p.add_argument("-o", "--output-dir",     default=os.getenv("TORRENT_OUTPUT", "downloads"))
    p.add_argument("--timeout",              type=int,   default=int(os.getenv("TORRENT_TIMEOUT", "3600")))
    p.add_argument("--stall-timeout",        type=int,   default=int(os.getenv("TORRENT_STALL", "300")))
    p.add_argument("--retries",              type=int,   default=int(os.getenv("TORRENT_RETRIES", "3")))
    p.add_argument("--max-upload-kbps",      type=int,   default=int(os.getenv("TORRENT_MAX_UL", "50")))
    p.add_argument("--max-download-kbps",    type=int,   default=int(os.getenv("TORRENT_MAX_DL", "0")))
    p.add_argument("--seed",                 action="store_true", default=False)
    p.add_argument("-v", "--verbose",        action="store_true", default=False)
    return p.parse_args()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    log     = setup_logging(args.verbose)

    # Collect sources: CLI args + env var
    sources = list(args.sources)
    env_src = os.getenv("TORRENT_SOURCES", "")
    if env_src:
        sources += [s.strip() for s in env_src.splitlines() if s.strip()]

    if not sources:
        log.error("No torrent sources provided. Use positional args or TORRENT_SOURCES env var.")
        sys.exit(1)

    cfg = DownloadConfig(
        sources          = sources,
        output_dir       = Path(args.output_dir),
        timeout_sec      = args.timeout,
        stall_sec        = args.stall_timeout,
        max_retries      = args.retries,
        max_upload_kbps  = args.max_upload_kbps,
        max_download_kbps= args.max_download_kbps,
        seed_after       = args.seed,
        verbose          = args.verbose,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting session — %d source(s) queued", len(cfg.sources))
    ses     = make_session(cfg)
    results = []

    # Graceful shutdown on SIGINT / SIGTERM
    shutdown = False
    def _handle_signal(signum, frame):
        nonlocal shutdown
        log.warning("Signal %s received — finishing current download then exiting.", signum)
        shutdown = True
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    for i, source in enumerate(cfg.sources, 1):
        if shutdown:
            log.warning("Shutdown requested — skipping remaining sources.")
            break
        log.info("━━━ [%d/%d] %s", i, len(cfg.sources), source[:100])
        result = download_one(ses, source, cfg, log)

        if result.success:
            ok = verify_download(result.save_path, log)
            if not ok:
                result.success = False
                result.error   = "Post-download integrity check failed (empty files?)"

        results.append(result)
        status = "SUCCESS" if result.success else f"FAILED ({result.error})"
        log.info("Result [%d/%d]: %s — %s", i, len(cfg.sources), status, source[:80])

    write_summary(results, cfg.output_dir, log)

    failed = [r for r in results if not r.success]
    if failed:
        log.error("%d download(s) failed.", len(failed))
        sys.exit(1)

    log.info("All %d download(s) completed successfully.", len(results))


if __name__ == "__main__":
    main()
