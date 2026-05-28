#!/usr/bin/env python3
"""Import existing TensorBoard logs into a new Weights & Biases run."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch
import wandb
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    STORE_EVERYTHING_SIZE_GUIDANCE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import TensorBoard logs into a new W&B run."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project name"
    )
    parser.add_argument(
        "--entity",
        required=True,
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for the new W&B run (default: folder name from --model-dir)"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model output directory containing 'tensorboard' and 'checkpoints' subdirectories"
    )
    parser.add_argument(
        "--dryrun",
        type=int,
        nargs="?",
        const=10,
        default=None,
        help="Dry run mode: print first K steps (default: 10) without logging to W&B"
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update an existing W&B run instead of creating a new one. Only logs new keys."
    )
    return parser.parse_args()


def discover_paths(model_dir: str) -> tuple:
    """Discover tensorboard and checkpoint paths from a model directory.

    Args:
        model_dir: Model output directory containing 'tensorboard' and 'checkpoints' subdirs

    Returns:
        tuple of (tensorboard_dir, checkpoint_file_path or None if not found)
    """
    model_path = Path(model_dir)
    assert model_path.exists(), f"Model directory does not exist: {model_dir}"

    # Find tensorboard directory
    tensorboard_dir = model_path / "tensorboard"
    assert tensorboard_dir.exists(), f"TensorBoard directory not found: {tensorboard_dir}"

    # Find checkpoints directory
    checkpoints_dir = model_path / "checkpoints"
    if not checkpoints_dir.exists():
        print("Warning: Checkpoints directory not found, skipping config loading")
        return tensorboard_dir, None

    # Determine which directory contains the iter_* folders
    iter_parent_dir = checkpoints_dir
    iter_dirs_in_checkpoints = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    if not iter_dirs_in_checkpoints:
        # Check for tp_1 subdirectory
        tp1_dir = checkpoints_dir / "tp_1"
        if tp1_dir.exists():
            print(f"No iter_* dirs in checkpoints, checking {tp1_dir}...")
            iter_parent_dir = tp1_dir
            iter_dirs_in_checkpoints = [d for d in tp1_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]

        if not iter_dirs_in_checkpoints:
            print("Warning: No checkpoint iteration directories found, skipping config loading")
            return tensorboard_dir, None

    # Read latest checkpoint iteration
    latest_iter_file = iter_parent_dir / "latest_checkpointed_iteration.txt"
    if not latest_iter_file.exists():
        # Try parent checkpoints dir
        latest_iter_file = checkpoints_dir / "latest_checkpointed_iteration.txt"

    if not latest_iter_file.exists():
        print("Warning: latest_checkpointed_iteration.txt not found, skipping config loading")
        return tensorboard_dir, None

    latest_iter = latest_iter_file.read_text().strip()
    print(f"Latest checkpoint iteration: {latest_iter}")

    # Find the checkpoint iteration directory
    iter_dir = iter_parent_dir / f"iter_{int(latest_iter):07d}"
    if not iter_dir.exists():
        # Fall back to finding checkpoint dirs by scanning and sorting in reverse order
        print(f"Warning: Checkpoint dir {iter_dir} not found, searching for previous checkpoint...")
        iter_dirs = sorted(
            [d for d in iter_parent_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            reverse=True
        )
        if not iter_dirs:
            print("Warning: No checkpoint iteration directories found, skipping config loading")
            return tensorboard_dir, None
        iter_dir = iter_dirs[0]
        print(f"Using fallback checkpoint: {iter_dir.name}")

    # Find the first mp_rank directory (sorted alphabetically, rank 0 should be first)
    mp_rank_dirs = sorted([d for d in iter_dir.iterdir() if d.is_dir() and d.name.startswith("mp_rank")])
    if mp_rank_dirs:
        # Use first mp_rank directory
        rank_dir = mp_rank_dirs[0]
        checkpoint_file = rank_dir / "model_optim_rng.pt"
        if not checkpoint_file.exists():
            print(f"Warning: model_optim_rng.pt not found in {rank_dir}, skipping config loading")
            return tensorboard_dir, None
        print(f"Using checkpoint: {checkpoint_file}")
    else:
        # Non-distributed checkpoint format - look for model_optim_rng.pt directly
        checkpoint_file = iter_dir / "model_optim_rng.pt"
        if not checkpoint_file.exists():
            checkpoint_file = iter_dir / "model_rng.pt"
        if not checkpoint_file.exists():
            print(f"Warning: No checkpoint file found in {iter_dir}, skipping config loading")
            return tensorboard_dir, None
        print(f"Using checkpoint: {checkpoint_file}")

    return tensorboard_dir, checkpoint_file


def load_config_from_checkpoint(checkpoint_file: Path) -> dict:
    """Load training config from a Megatron checkpoint file.

    Args:
        checkpoint_file: Path to checkpoint file (e.g., model_optim_rng.pt)

    Returns:
        dict of training arguments, or empty dict if not found
    """
    print(f"Loading config from checkpoint: {checkpoint_file}")

    try:
        # Load with map_location to CPU to avoid GPU memory issues
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

        if "args" not in state_dict:
            print("Warning: Checkpoint does not contain 'args'")
            return {}

        checkpoint_args = state_dict["args"]

        # Convert namespace to dict, filtering out non-serializable items
        config = {}
        for key, value in vars(checkpoint_args).items():
            # Skip items that can't be serialized to JSON
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                config[key] = value
            elif hasattr(value, "__name__"):  # Functions/classes
                config[key] = str(value)
            else:
                try:
                    # Try to convert to string as fallback
                    config[key] = str(value)
                except:
                    pass

        print(f"Loaded {len(config)} config parameters from checkpoint")
        return config

    except ModuleNotFoundError as e:
        if "megatron" in str(e):
            raise RuntimeError(
                f"Failed to load checkpoint config: {e}\n"
                "Please update your PYTHONPATH to include the megatron-lm directory:\n"
                "  export PYTHONPATH=/path/to/megatron-lm:$PYTHONPATH"
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint config: {e}") from e


def find_existing_run(api: wandb.Api, entity: str, project: str, run_name: str):
    """Find an existing run with the given name in the project.

    Returns:
        wandb.Run object if found, None otherwise
    """
    try:
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        return runs[0] if len(runs) > 0 else None
    except wandb.errors.CommError:
        # Project doesn't exist yet
        return None


def get_existing_keys(run) -> set:
    """Get the set of metric keys that already exist in a W&B run.

    Args:
        run: wandb.Run object from the API

    Returns:
        set of existing metric key names
    """
    try:
        # Get history keys from the run
        history = run.history(samples=1)
        if history.empty:
            return set()
        return set(history.columns)
    except Exception as e:
        print(f"Warning: Could not fetch existing keys: {e}")
        return set()


def load_tensorboard_events(logdir: str) -> EventAccumulator:
    """Load TensorBoard events from the specified directory.

    Uses STORE_EVERYTHING_SIZE_GUIDANCE to load all events without truncation.
    """
    event_acc = EventAccumulator(logdir, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)
    event_acc.Reload()
    return event_acc


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration string."""
    duration = timedelta(seconds=seconds)
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    else:
        return f"{minutes}m {secs}s"


def extract_runtime_info(event_acc: EventAccumulator, std_threshold: float = 3.0) -> dict:
    """Extract runtime information from TensorBoard events.

    Calculates both wall clock time and actual training time (excluding gaps from job restarts).
    Uses statistical outlier detection to identify job restarts: a gap is considered a restart
    if it's more than std_threshold standard deviations above the mean gap.

    Args:
        event_acc: TensorBoard event accumulator
        std_threshold: Number of standard deviations above mean to detect job restarts (default: 3.0)

    Returns:
        dict with runtime info including wall_time, actual_training_time, and job segments
    """
    import statistics

    tags = event_acc.Tags().get("scalars", [])
    if not tags:
        return {}

    # Use first available tag to get timestamps
    events = event_acc.Scalars(tags[0])
    if not events or len(events) < 2:
        return {}

    start_time = events[0].wall_time
    end_time = events[-1].wall_time
    wall_clock_seconds = end_time - start_time

    # Calculate all time gaps between consecutive events
    gaps = []
    for i in range(1, len(events)):
        gap = events[i].wall_time - events[i-1].wall_time
        gaps.append(gap)

    # Calculate mean and std of gaps to detect outliers (job restarts)
    if len(gaps) >= 2:
        mean_gap = statistics.mean(gaps)
        std_gap = statistics.stdev(gaps)
        # Gap threshold: mean + std_threshold * std (but at least 60 seconds to avoid false positives)
        gap_threshold = max(mean_gap + std_threshold * std_gap, 60.0)
    else:
        # Not enough data points, use a fixed threshold
        gap_threshold = 600.0  # 10 minutes
        mean_gap = gaps[0] if gaps else 0
        std_gap = 0

    # Identify gaps that exceed the threshold
    restart_gaps = [(i, gaps[i]) for i in range(len(gaps)) if gaps[i] > gap_threshold]

    # Build job segments
    actual_training_seconds = 0.0
    job_segments = []
    segment_start_idx = 0

    for gap_idx, gap_duration in restart_gaps:
        # End the current segment at the event before the gap
        segment_end_idx = gap_idx
        segment_start = events[segment_start_idx].wall_time
        segment_end = events[segment_end_idx].wall_time
        segment_duration = segment_end - segment_start

        job_segments.append({
            "start": datetime.fromtimestamp(segment_start).isoformat(),
            "end": datetime.fromtimestamp(segment_end).isoformat(),
            "duration_seconds": segment_duration,
            "iterations": segment_end_idx - segment_start_idx + 1,
            "gap_after_seconds": gap_duration,
        })
        actual_training_seconds += segment_duration

        # Start new segment after the gap
        segment_start_idx = gap_idx + 1

    # Add the final segment
    segment_start = events[segment_start_idx].wall_time
    segment_end = events[-1].wall_time
    segment_duration = segment_end - segment_start
    job_segments.append({
        "start": datetime.fromtimestamp(segment_start).isoformat(),
        "end": datetime.fromtimestamp(segment_end).isoformat(),
        "duration_seconds": segment_duration,
        "iterations": len(events) - segment_start_idx,
    })
    actual_training_seconds += segment_duration

    return {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "wall_clock_seconds": wall_clock_seconds,
        "wall_clock_str": format_duration(wall_clock_seconds),
        "actual_training_seconds": actual_training_seconds,
        "actual_training_str": format_duration(actual_training_seconds),
        "num_job_segments": len(job_segments),
        "job_segments": job_segments,
        "gap_stats": {
            "mean_seconds": mean_gap,
            "std_seconds": std_gap,
            "threshold_seconds": gap_threshold,
            "num_detected_restarts": len(restart_gaps),
        },
    }


def collect_scalar_data(event_acc: EventAccumulator, std_threshold: float = 3.0) -> dict:
    """Collect all scalar metrics from TensorBoard into a dict.

    Correlates iteration-based and sample-based scalars by event index,
    so all metrics are logged together at each iteration step.

    Logs ALL data - if some tags have more events than others, those extra
    events are still logged (with missing tags omitted for those iterations).

    Returns:
        dict mapping iteration -> {tag: value, samples: sample_count, ...}
    """
    import statistics

    tags = event_acc.Tags().get("scalars", [])
    if not tags:
        print("Warning: No scalar tags found in TensorBoard logs.")
        return {}

    print(f"Found {len(tags)} scalar tags: {tags}")

    # Separate iteration-based and sample-based tags
    iter_tags = [t for t in tags if "vs samples" not in t]
    sample_tags = [t for t in tags if "vs samples" in t]

    print(f"  Iteration-based tags: {len(iter_tags)}")
    print(f"  Sample-based tags: {len(sample_tags)}")

    if not iter_tags:
        print("Warning: No iteration-based tags found.")
        return {}

    # Pre-load all events for each tag
    iter_events = {tag: event_acc.Scalars(tag) for tag in iter_tags}
    sample_events = {tag: event_acc.Scalars(tag) for tag in sample_tags}

    # Find max event count across all tags to ensure we log everything
    all_event_counts = [len(events) for events in iter_events.values()]
    all_event_counts += [len(events) for events in sample_events.values()]
    max_events = max(all_event_counts)

    # Report any count mismatches
    for tag, events in {**iter_events, **sample_events}.items():
        if len(events) != max_events:
            print(f"  Note: {tag} has {len(events)} events (max is {max_events})")

    # Use the first iteration-based tag to get iteration numbers and wall_time
    reference_tag = iter_tags[0]
    reference_events = iter_events[reference_tag]

    # Calculate gap threshold for detecting job restarts (same logic as extract_runtime_info)
    gaps = []
    for i in range(1, len(reference_events)):
        gap = reference_events[i].wall_time - reference_events[i-1].wall_time
        gaps.append(gap)

    if len(gaps) >= 2:
        mean_gap = statistics.mean(gaps)
        std_gap = statistics.stdev(gaps)
        gap_threshold = max(mean_gap + std_threshold * std_gap, 60.0)
    else:
        gap_threshold = 600.0

    # Build iteration -> data mapping by iterating through all event indices
    step_data = {}  # iteration -> {tag: value, ...}

    # Get start time for relative time calculation
    start_wall_time = reference_events[0].wall_time if reference_events else 0

    # Track cumulative actual training time (excluding gaps)
    cumulative_train_time = 0.0
    total_gap_time = 0.0
    prev_samples_count = 0

    for i in range(max_events):
        is_job_restart = False
        time_since_prev = 0.0

        # Get iteration number and wall_time from iteration-based events if available
        if i < len(reference_events):
            iteration = reference_events[i].step
            wall_time = reference_events[i].wall_time

            # Calculate time since previous event
            if i > 0:
                time_since_prev = wall_time - reference_events[i-1].wall_time
                if time_since_prev > gap_threshold:
                    # This is a gap (job restart), don't add to training time
                    total_gap_time += time_since_prev
                    is_job_restart = True
                else:
                    cumulative_train_time += time_since_prev
        else:
            # Extrapolate iteration for extra events (assume sequential)
            iteration = reference_events[-1].step + (i - len(reference_events) + 1)
            wall_time = reference_events[-1].wall_time if reference_events else 0

        step_data[iteration] = {"iteration": iteration}

        # Add relative times (in hours)
        step_data[iteration]["perf/est_process_time_hrs"] = (wall_time - start_wall_time) / 3600.0
        step_data[iteration]["perf/est_train_time_hrs"] = cumulative_train_time / 3600.0

        # Add iteration-based tag values (if available at this index)
        for tag, events in iter_events.items():
            if i < len(events):
                step_data[iteration][tag] = events[i].value

        # Add sample-based tag values (if available at this index)
        samples_count = None
        for tag, events in sample_events.items():
            if i < len(events):
                step_data[iteration][tag] = events[i].value
                # Store the sample count as "samples vs steps"
                if "samples vs steps" not in step_data[iteration]:
                    step_data[iteration]["samples vs steps"] = events[i].step
                    samples_count = events[i].step

        # Calculate throughput metrics
        if samples_count is not None and cumulative_train_time > 0:
            # Total throughput: cumulative samples / cumulative training time
            step_data[iteration]["perf/est_total_samples_per_sec"] = samples_count / cumulative_train_time

            # Per-iteration throughput: only when not starting a new job segment
            if not is_job_restart and time_since_prev > 0 and i > 0:
                samples_this_iter = samples_count - prev_samples_count
                step_data[iteration]["perf/est_iter_samples_per_sec"] = samples_this_iter / time_since_prev

        # Track previous samples for next iteration
        if samples_count is not None:
            prev_samples_count = samples_count

    return step_data


def log_time_summary(step_data: dict) -> None:
    """Log summary time values as human-readable strings to W&B run summary."""
    if not step_data:
        return

    # Get the last iteration's time values
    last_iteration = max(step_data.keys())
    last_data = step_data[last_iteration]

    train_hrs = last_data.get("perf/est_train_time_hrs", 0)
    process_hrs = last_data.get("perf/est_process_time_hrs", 0)

    # Convert hours to seconds for formatting, then log as strings
    if train_hrs:
        wandb.run.summary["perf/est_train_time_str"] = format_duration(train_hrs * 3600)
    if process_hrs:
        wandb.run.summary["perf/est_process_time_str"] = format_duration(process_hrs * 3600)


def import_scalars_to_wandb(
    step_data: dict,
    update_mode: bool = False,
    existing_keys: set = None
) -> int:
    """Import scalar metrics to W&B.

    Args:
        step_data: dict mapping iteration -> {tag: value, ...}
        update_mode: If True, don't use step= and only log new keys
        existing_keys: Set of keys that already exist (only used in update_mode)

    Returns the number of data points logged.
    """
    if not step_data:
        return 0

    existing_keys = existing_keys or set()
    # Keys that should always be included for context
    always_include = {"iteration", "samples vs steps"}

    total_points = 0
    for step in sorted(step_data.keys()):
        if update_mode:
            # Filter to only new keys (plus always-include keys)
            log_data = {
                k: v for k, v in step_data[step].items()
                if k not in existing_keys or k in always_include
            }
            if log_data:
                # Don't use step= when updating to avoid having wandb skip data from prev steps
                wandb.log(log_data)
                total_points += len(log_data)
        else:
            wandb.log(step_data[step], step=step)
            total_points += len(step_data[step])

    return total_points


def preview_scalars(event_acc: EventAccumulator, step_data: dict, num_steps: int) -> None:
    """Print a preview of scalar data, showing per-tag statistics and samples."""
    if not step_data:
        print("No data to preview.")
        return

    tags = event_acc.Tags().get("scalars", [])

    # Show per-tag breakdown
    print(f"\n=== DRY RUN: Per-tag breakdown ===\n")
    for tag in tags:
        events = event_acc.Scalars(tag)
        if events:
            first_step = events[0].step
            last_step = events[-1].step
            first_val = events[0].value
            last_val = events[-1].value
            print(f"{tag}:")
            print(f"  Events: {len(events)}, Steps: {first_step} -> {last_step}")
            print(f"  First: {first_val:.6g}, Last: {last_val:.6g}")
            print()

    # Show first N iterations with all correlated data
    sorted_iterations = sorted(step_data.keys())
    total_iterations = len(sorted_iterations)
    print(f"=== First {num_steps} iterations (with correlated scalars) ===\n")

    for iteration in sorted_iterations[:num_steps]:
        data = step_data[iteration]
        samples = data.get("samples vs steps", "N/A")
        train_hrs = data.get("perf/est_train_time_hrs", 0)
        process_hrs = data.get("perf/est_process_time_hrs", 0)
        # Convert hours back to seconds for human-readable display
        train_time_str = format_duration(train_hrs * 3600) if train_hrs else "0s"
        process_time_str = format_duration(process_hrs * 3600) if process_hrs else "0s"
        print(f"Iteration {iteration} (samples: {samples}, train: {train_time_str}, process: {process_time_str}):")
        for key, value in sorted(data.items()):
            if key != "iteration":
                print(f"  {key}: {value}")
        print()

    if total_iterations > num_steps:
        print(f"... and {total_iterations - num_steps} more iterations")

    # Print summary
    total_points = sum(len(d) for d in step_data.values())
    # Count unique tags per iteration (excluding "iteration" key)
    if sorted_iterations:
        sample_data = step_data[sorted_iterations[0]]
        num_metrics = len([k for k in sample_data.keys() if k != "iteration"])
        print(f"\nSummary: {total_iterations} iterations, {num_metrics} metrics per iteration, {total_points} total data points")

    # Print runtime info
    runtime_info = extract_runtime_info(event_acc)
    if runtime_info:
        print(f"\n=== Runtime Information ===")
        print(f"  Start: {runtime_info['start_time']}")
        print(f"  End: {runtime_info['end_time']}")
        print(f"  Wall clock time: {runtime_info['wall_clock_str']}")
        print(f"  Actual training time: {runtime_info['actual_training_str']}")

        # Show gap detection stats
        gap_stats = runtime_info.get('gap_stats', {})
        if gap_stats:
            print(f"  Gap detection: mean={gap_stats['mean_seconds']:.1f}s, std={gap_stats['std_seconds']:.1f}s, threshold={gap_stats['threshold_seconds']:.1f}s")

        if runtime_info['num_job_segments'] > 1:
            print(f"  Job segments: {runtime_info['num_job_segments']}")
            for i, seg in enumerate(runtime_info['job_segments']):
                gap_info = f", gap after: {format_duration(seg['gap_after_seconds'])}" if 'gap_after_seconds' in seg else ""
                print(f"    Segment {i+1}: {format_duration(seg['duration_seconds'])} ({seg['iterations']} iters){gap_info}")


def main():
    args = parse_args()

    # Derive run name from model directory if not provided
    if args.run_name is None:
        args.run_name = Path(args.model_dir).name
        print(f"Using run name from model directory: {args.run_name}")

    # Discover tensorboard and checkpoint paths from model directory
    print(f"Discovering paths in model directory: {args.model_dir}")
    tensorboard_dir, checkpoint_file = discover_paths(args.model_dir)

    # Check for event files
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"Error: No TensorBoard event files found in '{tensorboard_dir}'")
        sys.exit(1)

    print(f"Found {len(event_files)} event file(s) in '{tensorboard_dir}'")

    # Load TensorBoard events
    print(f"Loading TensorBoard events from '{tensorboard_dir}'...")
    event_acc = load_tensorboard_events(str(tensorboard_dir))

    # Collect scalar data
    step_data = collect_scalar_data(event_acc)

    # Load config from checkpoint (if available)
    if checkpoint_file is not None:
        wandb_config = load_config_from_checkpoint(checkpoint_file)
    else:
        wandb_config = {}

    # Extract and add runtime info to config
    runtime_info = extract_runtime_info(event_acc)
    if runtime_info:
        wandb_config["training_start_time"] = runtime_info["start_time"]
        wandb_config["training_end_time"] = runtime_info["end_time"]
        wandb_config["training_wall_clock_seconds"] = runtime_info["wall_clock_seconds"]
        wandb_config["training_wall_clock"] = runtime_info["wall_clock_str"]
        wandb_config["training_actual_seconds"] = runtime_info["actual_training_seconds"]
        wandb_config["training_actual_time"] = runtime_info["actual_training_str"]
        wandb_config["training_num_job_segments"] = runtime_info["num_job_segments"]
        print(f"Training time: {runtime_info['actual_training_str']} (wall clock: {runtime_info['wall_clock_str']}, {runtime_info['num_job_segments']} job segment(s))")

    # Dry run mode: preview data and exit
    if args.dryrun is not None:
        preview_scalars(event_acc, step_data, args.dryrun)
        return

    # Check if run name already exists
    api = wandb.Api()
    existing_run = find_existing_run(api, args.entity, args.project, args.run_name)

    if existing_run:
        if not args.update_existing:
            print(
                f"Error: A run with the name '{args.run_name}' already exists "
                f"in project '{args.project}' under entity '{args.entity}'.\n"
                f"Use --update-existing to add new metrics to the existing run."
            )
            sys.exit(1)

        # Update existing run
        existing_keys = get_existing_keys(existing_run)
        new_keys = set(step_data[next(iter(step_data))].keys()) - existing_keys

        if not new_keys:
            print("No new keys to log. All metrics already exist in the run.")
            return

        print(f"Updating existing W&B run '{args.run_name}'...")
        print(f"  Existing keys: {sorted(existing_keys)}")
        print(f"  New keys to log: {sorted(new_keys)}")

        # Resume the existing run (use shared mode in case it's currently running)
        wandb.init(
            entity=args.entity,
            project=args.project,
            id=existing_run.id,
            resume="must",
            settings=wandb.Settings(mode="shared") if args.update_existing else None,
        )

        # Import only new scalars
        num_points = import_scalars_to_wandb(step_data, update_mode=True, existing_keys=existing_keys)

        # Log summary time strings
        log_time_summary(step_data)

        wandb.finish()
        print(f"Successfully added {num_points} data points to W&B run '{args.run_name}'.")

    else:
        if args.update_existing:
            print(
                f"Error: No run with the name '{args.run_name}' exists "
                f"in project '{args.project}' under entity '{args.entity}'.\n"
                f"Remove --update-existing to create a new run."
            )
            sys.exit(1)

        # Create new run
        print(f"Creating W&B run '{args.run_name}' in project '{args.entity}/{args.project}'...")
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.run_name,
            config=wandb_config,
        )

        # Import scalars
        num_points = import_scalars_to_wandb(step_data)

        # Log summary time strings
        log_time_summary(step_data)

        # Finish run
        wandb.finish()

        print(f"Successfully imported {num_points} data points into W&B run '{args.run_name}'.")


if __name__ == "__main__":
    """Example usage:

    # New runs: just provide model dir (run name derived from folder name)
    PYTHONPATH=. python examples/multimodal/tools/update_wandb_from_tensorboard.py \
        --project megatron-vlm-v3 \
        --entity adlr \
        --model-dir /lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/sft_moe_test_speed_collin

    # Duplicating runs: provide a new run name (b/c different user or sandboxing an existing run)
    PYTHONPATH=. python examples/multimodal/tools/update_wandb_from_tensorboard.py \
        --project megatron-vlm-v3 \
        --entity adlr \
        --run-name sft_moe_rl_llm_eval_mode_radio_v4_v1365_0126_dup \
        --model-dir /lustre/fsw/portfolios/llmservice/users/amalasanjayd/workspace/output/sft_moe_rl_llm_eval_mode_radio_v4_v1365_0126/

    # Update an existing run with new metrics (e.g., perf/est_train_time_hrs, perf/est_process_time_hrs):
    PYTHONPATH=. python examples/multimodal/tools/update_wandb_from_tensorboard.py \
        --project megatron-vlm-v3 \
        --entity adlr \
        --model-dir /lustre/fsw/portfolios/llmservice/users/cmccarthy/workspace/output/sft_moe_rl_llm_2e_bs_x2_radio_so400m_rc3_1230 \
        --update-existing

    # Use --dryrun to preview without logging:
    PYTHONPATH=. python examples/multimodal/tools/update_wandb_from_tensorboard.py \
        --project megatron-vlm-v3 \
        --entity adlr \
        --model-dir /lustre/fsw/portfolios/llmservice/users/trintamaki/workspace/output/sft_moe_1204 \
        --dryrun
    """
    main()
