import os
from datetime import datetime

from agent.agent import Agent
from .task_billiards import BilliardsTask
from .parser import Parser
from .feedback import FeedbackManager
from .prompter import Prompter

# ============ LLMAgent ============
class LLMAgent(Agent):
    # LLM-based agent
    def __init__(self):
        super().__init__()

        self.env = BilliardsTask()
        self.parser = Parser()
        self.feedback_manager = FeedbackManager()

        self.MAX_TOKENS = 512
        self.NUM_REPLANS = 3
        self.USE_FEEDBACK = True
        self.LLM_SOURCE = "deepseek-v3.2" # "qwen-plus" # "Moonshot-Kimi-K2-Instruct"
        self.TEMPERATURE = 0.0
        self.prompter = Prompter(env=self.env, 
                                 parser=self.parser, 
                                 feedback_manager=self.feedback_manager, 
                                 max_tokens=self.MAX_TOKENS, 
                                 num_replans=self.NUM_REPLANS, 
                                 use_feedback=self.USE_FEEDBACK, 
                                 llm_source=self.LLM_SOURCE, 
                                 temperature=self.TEMPERATURE
                                 )
        self.step = 0
        self.run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
       
    def decision(self, balls=None, my_targets=None, table=None):
        """
        Use LLM to decide best shot parameters.

        Args:
            balls: ball state dict {ball_id: Ball}
            my_targets: list of target ball ids ['1','2',...]
            table: table object

        Returns:
            dict: action {'V0','phi','theta','a','b'}; on failure returns random action
        """
        if balls is None:
            print(f"[LLMAgent] decision did not receive balls info, using random action.")
            return self._random_action()
        try:
            save_base = os.path.join('data', self.run_timestamp)
            save_path = os.path.join(save_base, f"step_{self.step}")
            os.makedirs(save_path, exist_ok=True)
            ready, action, feedbacks = self.prompter.prompt_one_round(balls=balls, my_targets=my_targets, table=table, save_path=save_path)
            self.step += 1

            for k in ['V0', 'phi', 'theta', 'a', 'b']:
                action[k] = float(action[k])
            return action

        except Exception as e:
            print(f"[LLMAgent] fatal error during decision, using random action. Reason: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()