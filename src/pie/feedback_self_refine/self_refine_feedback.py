import pandas as pd
from tqdm import tqdm


from pie.feedback_self_refine.feedback import PieSRFFeedback
from pie.feedback_self_refine.queries import PERFECT_FEEDBACK_WORDS
from pie.feedback_self_refine.task_init import PieSRFInit
from pie.feedback_self_refine.task_iterate import PieSRFIterate
from src.pie.task_init import PieInit
from src.pie.task_iterate import PieIterate
from src.pie.feedback import PieFeedback

from src.utils import retry_parse_fail_prone_cmd

import pandas as pd
from prompt_lib.backends import openai_api

from src.utils import Prompt

class PieSRF():
    def __init__(self, engine: str, temperature: float, max_tokens: int = 300) -> None:
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature

    @retry_parse_fail_prone_cmd
    def get_self_refined_feedback(self, slow_code: str, max_attempts: int, temperature: float):

        # initialize all the required components

        # generation of the first fast version
        task_init = PieSRFInit(engine=self.engine, temperature=self.temperature)


        task_feedback = PieSRFFeedback(engine=self.engine, temperature=temperature)

        # iteratively improving the code
        task_iterate = PieSRFIterate(engine=self.engine, temperature=temperature)

        # Initialize the task

        n_attempts = 0

        log = []
        feedback = None
        feedback_on_feedback = None
        prev_feedback=None

        while n_attempts < max_attempts:

            if n_attempts == 0:
                feedback = task_init(slow_code=slow_code)
            else:
                prev_feedback=feedback
                feedback = task_iterate(slow_code=slow_code, feedback=feedback, feedback_on_feedback=feedback_on_feedback)

            # feedback = task_feedback(slow_code=slow_code)
            feedback_on_feedback = task_feedback(slow_code=slow_code, feedback=feedback)

            log.append({"feedback": feedback, "feedback_on_feedback": feedback_on_feedback, "prev_feedback": prev_feedback, "attempt": n_attempts})
            # show_example(**log[-1])
            

            # TODO: what should be the breaking sentence?
            if PERFECT_FEEDBACK_WORDS in feedback_on_feedback.lower():
                break

            # feedback = feedback

            n_attempts += 1

        return feedback, log


def show_example(**kwargs):
    # shows {"fast_code": fast_code, "feedback": feedback, "slow_code": slow_code, "attempt": n_attempts}
    print(f"SLOW CODE:\n{kwargs['slow_code']}\n")
    print(f"\n\nFEEDBACK:\n{kwargs['feedback']}\n")
    print(f"\n\nFAST CODE:\n{kwargs['fast_code']}\n")
    print("-" * 100)
    
def run_over_slow_programs(slow_programs_file: str, max_attempts: int, outfile: str, feedback_type: str, temperature: float, backup_file: str = None):

    slow_programs_df = pd.read_json(slow_programs_file, lines=True, orient="records")
    slow_programs_df["run_logs"] = None

    if backup_file:
        backup_df = pd.read_json(backup_file, lines=True, orient="records")
        processed_inputs = set(backup_df["submission_id_v0"].tolist())
        results = backup_df.to_dict("records")
    else:
        processed_inputs = set()
        results = []

    for i, row in tqdm(slow_programs_df.iterrows(), total=len(slow_programs_df)):
        if i==1:
            break
        if row["submission_id_v0"] in processed_inputs:
            continue

        row_copy = row.to_dict()
        try:
            run_logs = iterative_pie(slow_code=row["input"], max_attempts=max_attempts, feedback_type=feedback_type, temperature=temperature)
            print(run_logs)
            row_copy["run_logs"] = run_logs
            results.append(row_copy)
            if i % 20 == 0:
                pd.DataFrame(results).to_json(outfile + f".{i}.jsonl", orient="records", lines=True)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            # raise e
            # pass
    pd.DataFrame(results).to_json(outfile, orient="records", lines=True)
    return run_logs



def test():
    slow_code = (
        "def sum(n):\\n    res = 0\\n    for i in range(n):\\n        res += i\\n    return res"
    )
    logs = run_over_slow_programs(
        slow_programs=[slow_code], max_attempts=3, outfile="/tmp/test.jsonl"
    )
    for (slow_code, log) in logs.items():
        for attempt in log:
            print(f"Slow code:\n {attempt['slow_code']}")
            print(f"Feedback: {attempt['feedback']}")
            print(f"Fast code:\n {attempt['fast_code']}")
            print()

if __name__ == "__main__":
    import sys

    if sys.argv[1] == "test":
        test()
    else:
        import argparse
        import os
        args = argparse.ArgumentParser()
        args.add_argument("--slow_programs_file", type=str, required=True)
        args.add_argument("--max_attempts", type=int, default=3)
        args.add_argument("--outfile", type=str, required=True)
        args.add_argument("--feedback_type", type=str)
        args.add_argument("--temperature", type=float, default=0.0)
        args.add_argument("--backup_file", type=str)
        
        args = args.parse_args()
        args.outfile = f"{args.outfile}.fb_{args.feedback_type}.temp_{args.temperature}.engine_{ENGINE}.jsonl"
        if os.path.exists(args.outfile):
            
            v = 0
            while os.path.exists(args.outfile + f".v{v}"):
                v += 1
            args.outfile = args.outfile + f".v{v}"
            print(f"Output file {args.outfile} already exists. Adding a suffix to it (v{v})")
        run_over_slow_programs(slow_programs_file=args.slow_programs_file, max_attempts=args.max_attempts, outfile=args.outfile, feedback_type=args.feedback_type, temperature=args.temperature, backup_file=args.backup_file)