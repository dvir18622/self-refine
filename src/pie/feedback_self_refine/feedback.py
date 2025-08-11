import pandas as pd
from prompt_lib.backends import openai_api

from pie.feedback_self_refine.self_refine_feedback import FEEDBACK_INIT_Q, FEEDBACK_ON_FEEDBACK_Q
from src.utils import Prompt


class PieSRFFeedback(Prompt):
    def __init__(self, engine: str, temperature: float, max_tokens: int = 300) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix=FEEDBACK_ON_FEEDBACK_Q+"\n",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n### END ###n\n",
        )
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.setup_prompt_from_examples_file("./src/pie/feedback_self_refine/data/feedback.txt")

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        with open(examples_path, "r") as f:
            self.prompt = f.read()
    
    def __call__(self, slow_code: str, feedback: str):
        generation_query = self.make_query(slow_code=slow_code, wrong_feedback=feedback)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("task SBF feedback")
        print(generation_query)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        output = openai_api.OpenaiAPIWrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=self.max_tokens,
            stop_token="### END",
            temperature=self.temperature,
        )
        
        generated_feedback = openai_api.OpenaiAPIWrapper.get_first_response(output)
        if "### END" in generated_feedback:
            generated_feedback = generated_feedback.split("### END")[0]
        
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("task SBF feedback")
        print(generated_feedback)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        return generated_feedback.strip()
    
    def make_query(self, wrong_feedback: str, slow_code: str):
        example_template = """{slow_code}

{FEEDBACK_INIT_Q}

{wrong_feedback}

{FEEDBACK_ON_FEEDBACK_Q}

"""   
        query = example_template.format(slow_code=slow_code, FEEDBACK_INIT_Q=FEEDBACK_INIT_Q, wrong_feedback=wrong_feedback, FEEDBACK_ON_FEEDBACK_Q=FEEDBACK_ON_FEEDBACK_Q)
        # wrong_feedback = f"""{self.question_prefix}{slow_code}{wrong_feedback}{self.intra_example_sep}{self.answer_prefix}"""
        return f"{self.prompt}{query}"
    

def test():
    task_fb = PieSRFFeedback(
        prompt_examples="data/prompt/pie/feedback.txt",
        engine="gpt-3.5-turbo",
        temperature=0.0
    )

    print(task_fb.prompt)
    slow_code = "def sum(n):\\n    res = 0\\n    for i in range(n):\\n        res += i\\n    return res"
    print(task_fb(slow_code))
    

if __name__ == '__main__':
    test()