"""
Training Time Tracker for RVNP-SBI

Tracks training time for each model component and computes total time.
Key insight: Simulator flow is trained ONCE and reused, so we count it only once.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class StageTime:
    """Record of time for a single training stage."""
    stage_name: str
    time_seconds: float
    description: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingTimeTracker:
    """
    Tracks training time for all components of RVNP model.

    Components counted:
    - Simulator flow (trained once, reused)
    - Embedding (if applicable)
    - Posterior initialization
    - Correction model coarse training
    - Joint refinement
    - Final posterior tuning
    """

    stages: List[StageTime] = field(default_factory=list)
    start_times: Dict[str, float] = field(default_factory=dict)

    def start_stage(self, stage_name: str, description: str = ""):
        """Start timing a training stage."""
        self.start_times[stage_name] = time.time()
        return time.time()

    def end_stage(self, stage_name: str, description: str = ""):
        """End timing a training stage and record the time."""
        if stage_name not in self.start_times:
            raise ValueError(f"Stage '{stage_name}' was never started")

        elapsed = time.time() - self.start_times[stage_name]
        stage_time = StageTime(
            stage_name=stage_name,
            time_seconds=elapsed,
            description=description
        )
        self.stages.append(stage_time)
        del self.start_times[stage_name]

        return elapsed

    def add_stage(self, stage_name: str, time_seconds: float, description: str = ""):
        """Directly add a stage time (for already-computed times)."""
        stage_time = StageTime(
            stage_name=stage_name,
            time_seconds=time_seconds,
            description=description
        )
        self.stages.append(stage_time)

    def get_total_time(self) -> float:
        """
        Get total training time (sum of all stage times).
        Note: If simulator is in stages, it's counted once as it's trained once and reused.
        """
        return sum(stage.time_seconds for stage in self.stages)

    def get_stage_time(self, stage_name: str) -> Optional[float]:
        """Get time for a specific stage."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage.time_seconds
        return None

    def get_breakdown(self) -> Dict:
        """Get detailed breakdown of training times."""
        breakdown = {
            'total_training_time_seconds': self.get_total_time(),
            'total_training_time_minutes': self.get_total_time() / 60,
            'total_training_time_hours': self.get_total_time() / 3600,
            'stages': []
        }

        for stage in self.stages:
            breakdown['stages'].append({
                'stage_name': stage.stage_name,
                'time_seconds': stage.time_seconds,
                'time_minutes': stage.time_seconds / 60,
                'time_hours': stage.time_seconds / 3600,
                'description': stage.description,
                'percentage': (stage.time_seconds / self.get_total_time() * 100) if self.get_total_time() > 0 else 0
            })

        return breakdown

    def save_to_json(self, filepath: str):
        """Save timing breakdown to JSON file."""
        breakdown = self.get_breakdown()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(breakdown, f, indent=2)

        print(f"Training time breakdown saved to: {filepath}")

    def print_summary(self, title: str = "TRAINING TIME SUMMARY"):
        """Pretty-print training time summary."""
        breakdown = self.get_breakdown()
        total_time = breakdown['total_training_time_seconds']

        print()
        print("=" * 80)
        print(title)
        print("=" * 80)

        if not self.stages:
            print("No training stages recorded.")
            print("=" * 80)
            return

        # Print each stage
        for stage_info in breakdown['stages']:
            stage_name = stage_info['stage_name']
            time_sec = stage_info['time_seconds']
            time_min = stage_info['time_minutes']
            percentage = stage_info['percentage']
            desc = stage_info['description']

            # Format stage name (pad to 25 chars)
            name_display = f"{stage_name}:"
            if desc:
                name_display += f" ({desc})"

            print(f"{name_display:50s} {time_sec:8.1f}s ({time_min:6.1f} min) [{percentage:5.1f}%]")

        # Separator
        print("─" * 80)

        # Total
        total_hours = breakdown['total_training_time_hours']
        total_mins = breakdown['total_training_time_minutes']
        print(f"{'Total Training Time:':<50s} {total_time:8.1f}s ({total_mins:6.1f} min / {total_hours:5.2f} hours)")

        print("=" * 80)
        print()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


# Global tracker instance (can be used across modules)
_global_tracker = None


def get_global_tracker() -> TrainingTimeTracker:
    """Get or create the global time tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = TrainingTimeTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset the global time tracker."""
    global _global_tracker
    _global_tracker = TrainingTimeTracker()


if __name__ == '__main__':
    # Example usage
    print("Example usage of TrainingTimeTracker")
    print()

    tracker = TrainingTimeTracker()

    # Simulate training stages
    import time as time_module

    tracker.start_stage("simulator_flow", "Training simulator p(x|θ)")
    time_module.sleep(0.1)
    tracker.end_stage("simulator_flow")

    tracker.start_stage("embedding", "Training embedding network")
    time_module.sleep(0.05)
    tracker.end_stage("embedding")

    tracker.start_stage("posterior_init", "Initializing posterior")
    time_module.sleep(0.08)
    tracker.end_stage("posterior_init")

    tracker.start_stage("correction_coarse", "Coarse correction training")
    time_module.sleep(0.06)
    tracker.end_stage("correction_coarse")

    tracker.start_stage("joint_refinement", "Joint refinement")
    time_module.sleep(0.12)
    tracker.end_stage("joint_refinement")

    # Print summary
    tracker.print_summary()

    # Save to file
    tracker.save_to_json("example_timing.json")

    # Show breakdown
    print("Breakdown dictionary:")
    import pprint
    pprint.pprint(tracker.get_breakdown())
