"""Test runner harness for FireDetection_Classification.
Executes all unit and E2E test suites (Tiers 1-4) and prints an aggregated report.
"""

import os
import sys
import time
import pytest


def run_all_suites():
    """Runs pytest across unit and e2e test directories and collects results."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(project_root, "tests")

    suites = [
        ("Unit Tests", os.path.join(test_dir, "unit")),
        ("Tier 1: Feature Coverage", os.path.join(test_dir, "e2e", "test_tier1_features.py")),
        ("Tier 2: Boundary & Corner Cases", os.path.join(test_dir, "e2e", "test_tier2_boundaries.py")),
        ("Tier 3: Pairwise Interactions", os.path.join(test_dir, "e2e", "test_tier3_combinations.py")),
        ("Tier 4: Application Workloads", os.path.join(test_dir, "e2e", "test_tier4_workloads.py")),
    ]

    print("=" * 70)
    print(" FIRE & SMOKE CLASSIFICATION: E2E TEST RUNNER")
    print(f" Working Directory: {project_root}")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    suite_results = []

    overall_start = time.perf_counter()

    class CustomPlugin:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.skipped = 0

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    self.passed += 1
                elif report.failed:
                    self.failed += 1
                elif report.skipped:
                    self.skipped += 1
            elif report.when == "setup" and report.skipped:
                self.skipped += 1

    for name, path in suites:
        if not os.path.exists(path):
            print(f"[-] {name}: Path not found ({path})")
            continue

        plugin = CustomPlugin()
        t0 = time.perf_counter()
        # Run pytest silently for this suite
        exit_code = pytest.main(["-q", "--no-header", path], plugins=[plugin])
        elapsed = time.perf_counter() - t0

        total_passed += plugin.passed
        total_failed += plugin.failed
        total_skipped += plugin.skipped

        status = "PASSED" if plugin.failed == 0 else "FAILED"
        suite_results.append({
            "name": name,
            "status": status,
            "passed": plugin.passed,
            "failed": plugin.failed,
            "skipped": plugin.skipped,
            "duration": elapsed
        })

    overall_duration = time.perf_counter() - overall_start

    print("\n" + "-" * 70)
    print(f"{'SUITE':<35} | {'STATUS':<8} | {'PASS':<5} | {'FAIL':<5} | {'SKIP':<5} | {'TIME (s)'}")
    print("-" * 70)
    for res in suite_results:
        print(f"{res['name']:<35} | {res['status']:<8} | {res['passed']:<5} | {res['failed']:<5} | {res['skipped']:<5} | {res['duration']:.2f}s")
    print("-" * 70)
    print(f"{'TOTAL AGGREGATE':<35} | {'':<8} | {total_passed:<5} | {total_failed:<5} | {total_skipped:<5} | {overall_duration:.2f}s")
    print("=" * 70)

    if total_failed > 0:
        print(f"\n[!] FAILURE: {total_failed} test(s) failed!")
        sys.exit(1)
    else:
        print(f"\n[*] SUCCESS: All executed tests passed cleanly ({total_passed} passed, {total_skipped} skipped pending module completion).")
        sys.exit(0)


if __name__ == "__main__":
    run_all_suites()
