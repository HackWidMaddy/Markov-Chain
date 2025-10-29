import argparse
import os
import sys
import threading
import multiprocessing as mp
from queue import Queue, Empty
from collections import Counter
from typing import Iterable, Optional, TextIO, List, Tuple


def build_letter_counter(s: str, letters_only: bool) -> Counter:
    if letters_only:
        # Consider only ASCII letters a-z
        filtered = [ch.lower() for ch in s if ch.isalpha()]
    else:
        filtered = [ch.lower() for ch in s]
    return Counter(filtered)


def contains_multiset(haystack: Counter, needle: Counter) -> bool:
    # True if haystack has at least the counts in needle
    for ch, need in needle.items():
        if haystack.get(ch, 0) < need:
            return False
    return True


def exact_anagram(haystack: Counter, needle: Counter) -> bool:
    # Exact same counts
    return haystack == needle


def worker(
    line_queue: Queue,
    target_counter: Counter,
    letters_only: bool,
    exact_length: bool,
    print_lock: threading.Lock,
    out_file: Optional[TextIO],
    stop_after: Optional[int],
    found_counter: "threading.local",
    finished_flag: threading.Event,
):
    try:
        while not finished_flag.is_set():
            try:
                line = line_queue.get(timeout=0.2)
            except Empty:
                # If producer finished and queue empty, exit
                if finished_flag.is_set() and line_queue.empty():
                    break
                continue

            pwd = line.rstrip("\n\r")
            if not pwd:
                line_queue.task_done()
                continue

            pwd_counter = build_letter_counter(pwd, letters_only)
            if exact_length:
                match = exact_anagram(pwd_counter, target_counter)
            else:
                match = contains_multiset(pwd_counter, target_counter)

            if match:
                with print_lock:
                    if out_file is not None:
                        out_file.write(pwd + "\n")
                        out_file.flush()
                    else:
                        print(pwd, flush=True)
                    if stop_after is not None:
                        found_counter.count += 1
                        if found_counter.count >= stop_after:
                            finished_flag.set()

            line_queue.task_done()
    except Exception as exc:
        with print_lock:
            print(f"[worker-error] {exc}", file=sys.stderr, flush=True)
        finished_flag.set()


def producer(file_path: str, line_queue: Queue, finished_flag: threading.Event):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if finished_flag.is_set():
                    break
                line_queue.put(line)
    finally:
        # Signal that no more lines will be added
        finished_flag.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find passwords in a wordlist that contain a target set of letters. "
            "By default, matches any password whose character multiset contains "
            "the target letters (case-insensitive). Use --exact-length for exact anagrams."
        )
    )
    parser.add_argument(
        "letters",
        help="Jumbled letters to search for (e.g., LOHE to match HELO).",
    )
    parser.add_argument(
        "--file",
        default="rockyou.txt",
        help="Path to wordlist (default: rockyou.txt).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Number of worker threads (default: CPU count).",
    )
    parser.add_argument(
        "--letters-only",
        action="store_true",
        help="Count only alphabetic letters (ignore digits/symbols) for matching.",
    )
    parser.add_argument(
        "--exact-length",
        action="store_true",
        help=(
            "Require exact-length anagram matches (password's char counts exactly match the letters)."
        ),
    )
    parser.add_argument(
        "--max-extra",
        type=int,
        help=(
            "Allow at most this many extra characters beyond the target letters. "
            "0 means exact letters only; omit for unlimited extras."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["thread", "process"],
        default="process",
        help="Execution engine: thread (original) or process (faster, default).",
    )
    parser.add_argument(
        "--output",
        help="Optional output file to write matches to (default: stdout).",
    )
    parser.add_argument(
        "--procs",
        type=int,
        default=max(1, (os.cpu_count() or 4)),
        help="Number of worker processes for process engine (default: CPU count).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after printing this many matches.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10000,
        help="Internal queue size for producer/consumer (default: 10000).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of lines per chunk for process engine (default: 4096).",
    )
    return parser.parse_args()


############################################################
# High-performance multiprocessing engine (bytes-based)
############################################################

# Globals set in process initializer for fast access
_G_TARGET_COUNTS: Optional[Tuple[int, ...]] = None
_G_TARGET_INDICES: Optional[Tuple[int, ...]] = None
_G_LETTERS_ONLY: bool = False
_G_EXACT_LENGTH: bool = False
_G_TARGET_SUM: int = 0
_G_MAX_EXTRA: int = -1  # -1 means unlimited


def _mp_init(target_counts: Tuple[int, ...], target_indices: Tuple[int, ...], letters_only: bool, exact_length: bool, target_sum: int, max_extra: int):
    global _G_TARGET_COUNTS, _G_TARGET_INDICES, _G_LETTERS_ONLY, _G_EXACT_LENGTH, _G_TARGET_SUM, _G_MAX_EXTRA
    _G_TARGET_COUNTS = target_counts
    _G_TARGET_INDICES = target_indices
    _G_LETTERS_ONLY = letters_only
    _G_EXACT_LENGTH = exact_length
    _G_TARGET_SUM = target_sum
    _G_MAX_EXTRA = max_extra


@staticmethod
def _lower_ascii_byte(b: int) -> int:
    return b + 32 if 65 <= b <= 90 else b


def _count_line_bytes(line: bytes, letters_only: bool) -> Tuple[List[int], int]:
    counts = [0] * 256
    total = 0
    for b in line:
        b = _lower_ascii_byte(b)
        if letters_only:
            if 97 <= b <= 122:  # a-z
                counts[b] += 1
                total += 1
        else:
            counts[b] += 1
            total += 1
    return counts, total


def _match_counts(counts: List[int]) -> bool:
    # Uses globals set by initializer
    target = _G_TARGET_COUNTS  # type: ignore
    indices = _G_TARGET_INDICES  # type: ignore
    if _G_EXACT_LENGTH:
        # Fast path: if total lengths differ, fail
        # Note: total is validated by caller for speed; equality check remains
        for i in indices:
            if counts[i] != target[i]:
                return False
        # Ensure no extra counts for characters not in target
        for i in range(256):
            if target[i] == 0 and counts[i] != 0:
                return False
        return True
    else:
        # Subset containment for indices with positive target
        for i in indices:
            if counts[i] < target[i]:
                return False
        return True


def _process_chunk(lines: List[bytes]) -> List[bytes]:
    out: List[bytes] = []
    letters_only = _G_LETTERS_ONLY
    exact_length = _G_EXACT_LENGTH
    target_sum = _G_TARGET_SUM
    max_extra = _G_MAX_EXTRA
    for raw in lines:
        line = raw.rstrip(b"\r\n")
        if not line:
            continue
        counts, total = _count_line_bytes(line, letters_only)
        if exact_length and total != target_sum:
            continue
        if _match_counts(counts):
            if not exact_length and max_extra >= 0:
                extra = total - target_sum
                if extra < 0 or extra > max_extra:
                    continue
            out.append(line)
    return out


def _build_target_bytes(letters: str, letters_only: bool) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    counts = [0] * 256
    total = 0
    for ch in letters:
        b = ord(ch)
        b = _lower_ascii_byte(b)
        if letters_only:
            if 97 <= b <= 122:
                counts[b] += 1
                total += 1
        else:
            counts[b] += 1
            total += 1
    indices = tuple(i for i, v in enumerate(counts) if v > 0)
    return tuple(counts), indices, total


def run_process_engine(
    file_path: str,
    letters: str,
    letters_only: bool,
    exact_length: bool,
    max_extra: Optional[int],
    procs: int,
    chunk_size: int,
    output_path: Optional[str],
    limit: Optional[int],
):
    target_counts, target_indices, target_sum = _build_target_bytes(letters, letters_only)

    # Open output (binary)
    out_bin: Optional[TextIO] = None
    if output_path:
        out_bin = open(output_path, "wb")

    total_found = 0

    # Normalize max_extra for child processes (-1 for unlimited)
    max_extra_norm = -1 if max_extra is None else int(max_extra)

    with mp.get_context("spawn").Pool(
        processes=max(1, procs),
        initializer=_mp_init,
        initargs=(target_counts, target_indices, letters_only, exact_length, target_sum, max_extra_norm),
    ) as pool:
        def chunk_iter() -> Iterable[List[bytes]]:
            with open(file_path, "rb") as f:
                batch: List[bytes] = []
                for line in f:
                    batch.append(line)
                    if len(batch) >= chunk_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

        try:
            for matches in pool.imap_unordered(_process_chunk, chunk_iter(), chunksize=1):
                if not matches:
                    continue
                for m in matches:
                    if out_bin is not None:
                        out_bin.write(m + b"\n")
                        out_bin.flush()
                    else:
                        sys.stdout.buffer.write(m + b"\n")
                        sys.stdout.buffer.flush()
                    total_found += 1
                    if limit is not None and total_found >= limit:
                        pool.terminate()
                        raise StopIteration
        except StopIteration:
            pass
        except KeyboardInterrupt:
            # Graceful Ctrl+C: terminate workers and stop output
            try:
                pool.terminate()
            except Exception:
                pass
        finally:
            if out_bin is not None:
                out_bin.close()


def run_thread_engine(args, target_counter: Counter):
    line_queue: Queue = Queue(maxsize=args.queue_size)
    print_lock = threading.Lock()
    finished_flag = threading.Event()

    out_file: Optional[TextIO] = None
    try:
        if args.output:
            out_file = open(args.output, "w", encoding="utf-8")

        # Launch producer
        prod_thread = threading.Thread(
            target=producer, args=(args.file, line_queue, finished_flag), daemon=True
        )
        prod_thread.start()

        # Shared counter for limit
        found_counter = threading.local()
        found_counter.count = 0

        # Launch workers
        workers = []
        for _ in range(max(1, args.threads)):
            t = threading.Thread(
                target=worker,
                args=(
                    line_queue,
                    target_counter,
                    args.letters_only,
                    args.exact_length,
                    print_lock,
                    out_file,
                    args.limit,
                    found_counter,
                    finished_flag,
                ),
                daemon=True,
            )
            t.start()
            workers.append(t)

        # Wait for queue to drain or early stop
        try:
            while any(t.is_alive() for t in workers):
                if finished_flag.is_set() and line_queue.empty():
                    break
                # Yield GIL periodically
                for _ in range(50):
                    if finished_flag.is_set() and line_queue.empty():
                        break
                    finished_flag.wait(timeout=0.01)
        except KeyboardInterrupt:
            finished_flag.set()

        # Ensure all work done
        line_queue.join()

    finally:
        if out_file is not None:
            out_file.close()


def main():
    # Interactive zero-arg mode for convenience
    if len(sys.argv) == 1:
        interactive_run()
        return

    args = parse_args()

    if not os.path.exists(args.file):
        print(f"Wordlist not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Derive max-extra/exactness behavior
    effective_exact = args.exact_length
    effective_max_extra: Optional[int] = args.max_extra
    if effective_max_extra is not None:
        # Override exact flag if max-extra specified
        effective_exact = (effective_max_extra == 0)

    if args.engine == "process":
        # Run fast multiprocessing engine
        try:
            run_process_engine(
                file_path=args.file,
                letters=args.letters,
                letters_only=args.letters_only,
                exact_length=effective_exact,
                max_extra=effective_max_extra,
                procs=args.procs,
                chunk_size=args.chunk_size,
                output_path=args.output,
                limit=args.limit,
            )
        except KeyboardInterrupt:
            # Ensure clean exit on Ctrl+C in main
            pass
    else:
        # Fallback to threading engine
        target_counter = build_letter_counter(args.letters, letters_only=args.letters_only)
        if not target_counter:
            print("No valid letters provided after filtering.", file=sys.stderr)
            sys.exit(1)
        # Simple enforcement for max-extra in thread engine: wrap stdout to filter by extra length
        if effective_max_extra is None and not effective_exact:
            try:
                run_thread_engine(args, target_counter)
            except KeyboardInterrupt:
                pass
        else:
            # Re-run using process engine for correctness and performance in this mode
            try:
                run_process_engine(
                    file_path=args.file,
                    letters=args.letters,
                    letters_only=args.letters_only,
                    exact_length=effective_exact,
                    max_extra=effective_max_extra,
                    procs=max(1, os.cpu_count() or 4),
                    chunk_size=args.chunk_size,
                    output_path=args.output,
                    limit=args.limit,
                )
            except KeyboardInterrupt:
                pass


def _prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    ans = input(f"{prompt} ({suffix}): ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes", "1", "true", "t")


def _prompt_int(prompt: str, default: int, min_value: int = 1) -> int:
    ans = input(f"{prompt} [{default}]: ").strip()
    if not ans:
        return default
    try:
        val = int(ans)
        return max(min_value, val)
    except Exception:
        return default


def interactive_run():
    print("Fast letter-set password finder (process engine)\n")
    default_file = "rockyou.txt"
    file_path = default_file if os.path.exists(default_file) else input("Wordlist path [rockyou.txt]: ").strip() or default_file
    if not os.path.exists(file_path):
        print(f"Wordlist not found: {file_path}", file=sys.stderr)
        return

    letters = input("Enter jumbled letters (e.g., LOHE): ").strip()
    if not letters:
        print("No letters provided.", file=sys.stderr)
        return

    letters_only = _prompt_bool("Letters-only (ignore digits/symbols)?", True)
    exact_length = _prompt_bool("Exact-length anagram only?", False)
    procs = _prompt_int("Processes", max(1, os.cpu_count() or 4))
    chunk_size = _prompt_int("Chunk size (lines per task)", 8192)
    set_limit = _prompt_bool("Limit number of matches?", False)
    limit = None
    if set_limit:
        limit = _prompt_int("Stop after N matches", 100)
    out_path = input("Output file (leave empty for stdout): ").strip() or None

    try:
        run_process_engine(
            file_path=file_path,
            letters=letters,
            letters_only=letters_only,
            exact_length=exact_length,
            procs=procs,
            chunk_size=chunk_size,
            output_path=out_path,
            limit=limit,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


