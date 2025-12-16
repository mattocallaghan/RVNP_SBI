"""Quick validation tests for refactoring."""

def test_experiment_config():
    """Test ExperimentConfig logic."""
    from training_utils import ExperimentConfig

    print("Testing ExperimentConfig...")

    # Test n=1 case: 50 experiments × 1 point
    config_1 = ExperimentConfig.from_num_tests(1)
    assert config_1.num_experiments == 50
    assert config_1.points_per_experiment == 1
    assert config_1.total_points_needed == 50
    print("  ✓ n=1: 50 experiments × 1 point = 50 total")

    # Test n=10 case: 10 experiments × 10 points
    config_10 = ExperimentConfig.from_num_tests(10)
    assert config_10.num_experiments == 10
    assert config_10.points_per_experiment == 10
    assert config_10.total_points_needed == 100
    print("  ✓ n=10: 10 experiments × 10 points = 100 total")

    # Test n=50 case: 2 experiments × 50 points
    config_50 = ExperimentConfig.from_num_tests(50)
    assert config_50.num_experiments == 2
    assert config_50.points_per_experiment == 50
    assert config_50.total_points_needed == 100
    print("  ✓ n=50: 2 experiments × 50 points = 100 total")

    # Test n=100 case: 1 experiment × 100 points
    config_100 = ExperimentConfig.from_num_tests(100)
    assert config_100.num_experiments == 1
    assert config_100.points_per_experiment == 100
    assert config_100.total_points_needed == 100
    print("  ✓ n=100: 1 experiment × 100 points = 100 total")

    # Test n=1000 case: 1 experiment × 1000 points
    config_1000 = ExperimentConfig.from_num_tests(1000)
    assert config_1000.num_experiments == 1
    assert config_1000.points_per_experiment == 1000
    assert config_1000.total_points_needed == 1000
    print("  ✓ n=1000: 1 experiment × 1000 points = 1000 total")

    # Test invalid case
    try:
        ExperimentConfig.from_num_tests(7)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported num_tests value: 7" in str(e)
        print("  ✓ Invalid n=7: Correctly raises ValueError")

    print("✅ ExperimentConfig tests passed!\n")


def test_logger():
    """Test TrainingLogger functionality."""
    import os
    import time
    from training_utils import TrainingLogger

    print("Testing TrainingLogger...")

    # Create logger
    logger = TrainingLogger("test_experiment", log_dir="logs_test")

    # Test timer context manager
    with logger.timer("Test operation") as timer_info:
        time.sleep(0.1)

    assert timer_info['elapsed'] >= 0.1
    assert "Test operation" in logger.stage_times
    print("  ✓ Timer context manager works")

    # Test epoch logging
    logger.log_epoch(epoch=1, train_loss=0.5, val_loss=0.6)
    print("  ✓ Epoch logging works")

    # Test stage logging
    logger.log_stage("Test Stage", param1=10, param2=20)
    print("  ✓ Stage logging works")

    # Check log file was created
    log_file = "logs_test/test_experiment.log"
    assert os.path.exists(log_file)
    print(f"  ✓ Log file created: {log_file}")

    # Clean up
    import shutil
    shutil.rmtree("logs_test")
    print("  ✓ Cleanup successful")

    print("✅ TrainingLogger tests passed!\n")


def test_validation():
    """Test config validation."""
    from training_utils import validate_config
    from types import SimpleNamespace

    print("Testing config validation...")

    # Create valid config
    valid_config = SimpleNamespace(
        seed=42,
        data=SimpleNamespace(
            dataset='CS',
            num_simulations=100000,
            num_tests=10,
            inference_simulations=100,
            num_iid=10,
            data_path='Data'
        ),
        model=SimpleNamespace(
            name='nlpe_rqs_posterior',
            flow_dimension=3,
            cond_dim=4,
            correction_type='simple'
        ),
        training=SimpleNamespace(
            batch_size=1024,
            n_iters=100
        ),
        optim=SimpleNamespace(
            lr=0.001
        )
    )

    # Test valid config
    assert validate_config(valid_config) == True
    print("  ✓ Valid config passes validation")

    # Test missing field
    invalid_config = SimpleNamespace(
        data=SimpleNamespace(
            dataset='CS',
            num_simulations=100000
            # Missing data_path
        ),
        model=SimpleNamespace(
            name='nlpe_rqs_posterior',
            flow_dimension=3,
            cond_dim=4
        ),
        training=SimpleNamespace(
            batch_size=1024,
            n_iters=100
        ),
        optim=SimpleNamespace(
            lr=0.001
        )
    )

    try:
        validate_config(invalid_config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing required field: data.data_path" in str(e)
        print("  ✓ Missing field correctly detected")

    # Test invalid correction type
    invalid_correction = SimpleNamespace(
        data=SimpleNamespace(
            dataset='CS',
            num_simulations=100000,
            data_path='Data',
            num_tests=10
        ),
        model=SimpleNamespace(
            name='nlpe_rqs_posterior',
            flow_dimension=3,
            cond_dim=4,
            correction_type='invalid_type'
        ),
        training=SimpleNamespace(
            batch_size=1024,
            n_iters=100
        ),
        optim=SimpleNamespace(
            lr=0.001
        )
    )

    try:
        validate_config(invalid_correction)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "correction_type must be one of" in str(e)
        print("  ✓ Invalid correction type correctly detected")

    print("✅ Config validation tests passed!\n")


def test_format_experiment_name():
    """Test experiment name formatting."""
    from training_utils import format_experiment_name
    from types import SimpleNamespace

    print("Testing experiment name formatting...")

    config = SimpleNamespace(
        model=SimpleNamespace(
            name='nlpe_rqs_posterior',
            correction_type='simple'
        ),
        data=SimpleNamespace(
            dataset='CS',
            num_simulations=100000,
            num_tests=10
        )
    )

    # Without experiment index
    name_no_idx = format_experiment_name(config)
    assert name_no_idx == 'nlpe_rqs_posterior_CS_n100000_tests10_simple'
    print(f"  ✓ Name without index: {name_no_idx}")

    # With experiment index
    name_with_idx = format_experiment_name(config, experiment_idx=5)
    assert name_with_idx == 'nlpe_rqs_posterior_CS_n100000_tests10_simple_exp5'
    print(f"  ✓ Name with index: {name_with_idx}")

    print("✅ Experiment name formatting tests passed!\n")


if __name__ == "__main__":
    print("=" * 70)
    print("RVNP-SBI Refactoring Validation Tests")
    print("=" * 70)
    print()

    test_experiment_config()
    test_logger()
    test_validation()
    test_format_experiment_name()

    print("=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("Refactoring validation successful. The code is ready to use.")
