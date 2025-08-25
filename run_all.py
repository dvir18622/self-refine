import argparse
import json
import logging
import os
from datetime import datetime
import sys
import subprocess
import yaml
import tempfile
import pandas as pd  # Add this import for handling JSONL files

PIE_RUN_OUTPUT_FILE="pie_run"
RESULT_SUMMARY_FILE="result_summary.txt"

FEEDBACK_TYPES = ["self-refine-feedback", "none"]
# FEEDBACK_TYPES = ["none", "classic", "self-refine-feedback"]

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create an empty directory based on model and timestamp.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use")
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples.")
    args = parser.parse_args()

    # Create directory
    # Format timestamp: YYYY-MM-DD_HH-MM
    timestamp = datetime.now().strftime(f"%Y-%m-%d_%H-%M")
    dir_path = os.path.join("run_all_results", args.model, timestamp)
    os.makedirs(dir_path, exist_ok=True)

    logging.info(f"Created directory: {dir_path}")

    # Save args to a file in dir_path
    args_file_path = os.path.join(dir_path, "args_dump.txt")
    with open(args_file_path, "w") as args_file:
        for arg, value in vars(args).items():
            args_file.write(f"{arg}: {value}\n")
    
    logging.info(f"Arguments saved to {args_file_path}")


    # Run feedback types
    for feedback_type in FEEDBACK_TYPES:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        logging.info(f"run for feedback \"{feedback_type}\"")
        run_feedback_type(feedback_type, args, dir_path)
        logging.info(f"Completed run for feedback \"{feedback_type}\"")


    logging.info("All runs completed.")


def run_feedback_type(feedback_type: str, args, dir_path: str):
    
    # Create directories
    feedback_dir = os.path.join(dir_path, feedback_type)
    os.makedirs(feedback_dir, exist_ok=True)

    # Run the refinement process
    logging.info(f"Starting pie/run.py")
    command = [
        "python", "-u", "src/pie/run.py",
        "--slow_programs_file", "data/tasks/pie/codenet-python-test-1k.jsonl",
        "--max_attempts", "1",
        "--outfile", os.path.join(feedback_dir, PIE_RUN_OUTPUT_FILE),
        "--feedback_type", feedback_type,
        "--num_examples", str(args.num_examples),  # Convert to string
        "--model", args.model,
    ]
    subprocess.run(command, check=True)

    pie_run_out_file = os.path.join(feedback_dir, PIE_RUN_OUTPUT_FILE) + ".jsonl"
    logging.info(f"Completed running pie/run.py, output saved to {pie_run_out_file}")

    # Run the prep_for_pie_eval.py script
    logging.info(f"Starting prep_for_pie_eval.py")
    prep_command = [
        "python", "-u", "src/pie/prep_for_pie_eval.py",
        args.model,
        pie_run_out_file,
        feedback_dir
    ]
    subprocess.run(prep_command, check=True)

    flat_output_file = os.path.join(feedback_dir, "output.attempt_codes")
    logging.info(f"Completed running prep_for_pie_eval.py, output saved to {flat_output_file}")

    # Update the perf_run_config.yaml file
    config_path = "perf_run_config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config["model_generated_outputs_path"] = flat_output_file
    perf_report_file = os.path.join(feedback_dir, "perf_report.jsonl")
    config["output_report_file_path"] = perf_report_file

    # Dump the updated config to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_config_file:
        temp_config_path = temp_config_file.name
        yaml.safe_dump(config, temp_config_file)

    logging.info(f"Temporary config file created at {temp_config_path}")

    # Run the evaluation script
    logging.info(f"Starting run_eval.py")
    eval_command = [
        "python", "-u", "pie-perf/src/codenet_eval/run_eval.py",
        "--eval_config", temp_config_path
    ]
    subprocess.run(
        eval_command,
        check=True,
        env={**os.environ, "PYTHONPATH": os.path.abspath("pie-perf")}
    )

    logging.info(f"Completed running run_eval.py, output saved to {perf_report_file}")
    
    # Summarize the results
    logging.info(f"Summarizing results")
    result_summary = summary_results(perf_report_file)
    result_path = os.path.join(feedback_dir, RESULT_SUMMARY_FILE)
    with open(result_path, "w") as f:
        json.dump(result_summary, f, indent=4)

    logging.info(f"Result summary saved to {result_path}")
    logging.info(f"Result Summary: {result_summary}")


def summary_results(perf_report_file: str) -> dict:
    """
    Summarizes the results from the evaluation report.

    Args:
        perf_report_file (str): Path to the performance report JSONL file.

    Returns:
        dict: A summary of the results.
    """
    
    # Load the performance report
    run_metrics = pd.read_json(perf_report_file, lines=True, orient="records")

    summary={"total_programs": len(run_metrics)}

    # Filter rows where the accuracy is not almost 1 (same as in the pie-perf repo, run_eval.py, print summary function)
    run_metrics_accurate = run_metrics[
        (run_metrics[f"final_attempt_code_acc"] > 0.99) & (run_metrics["input_acc"] > 0.99)
    ]
    if run_metrics.empty:
        logging.error(f"No valid run metrics found.")
        raise ValueError("No valid run metrics found.")
    
    summary["total_programs_accurate"] = len(run_metrics_accurate)
    summary["not_accurate_programs"] = len(run_metrics) - len(run_metrics_accurate)
    summary["accuracy_rate"] = len(run_metrics_accurate) / len(run_metrics) * 100 if len(run_metrics) > 0 else 0

    # All the programs where there was an improvment.
    # Same as in the pie-perf repo, run_eval.py, print summary function
    # Why "reference_time_mean" and not "input_time_mean"?????? CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    run_metrics_improved = run_metrics_accurate[
        run_metrics_accurate[f"final_attempt_code_time_mean"] < run_metrics_accurate["reference_time_mean"]
    ]

    summary["improved_programs"] = len(run_metrics_improved)
    summary["not_improved_programs"] = len(run_metrics_accurate) - len(run_metrics_improved)
    summary["improvement_rate"] = len(run_metrics_improved) / len(run_metrics) * 100 if len(run_metrics) > 0 else 0
    
    return summary

if __name__ == "__main__":
    main()