import os
import json
import openai
from .task_billiards import BilliardsTask
from .parser import Parser
from .feedback import FeedbackManager

# Load API key
key_path = os.path.join(os.path.dirname(__file__), "aliyun_key.json")
assert os.path.exists(key_path), "Please put your API key in LLMAgent/aliyun_key.json"
OPENAI_KEY = str(json.load(open(key_path)))
print("Using API Key:", OPENAI_KEY[:4]+"****"+OPENAI_KEY[-4:])
client = openai.OpenAI(api_key=OPENAI_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

class Prompter:
    @staticmethod
    def _usage_summary(usage_obj):
        # Extract completion_tokens, prompt_tokens, total_tokens from usage
        c = getattr(usage_obj, 'completion_tokens', None)
        p = getattr(usage_obj, 'prompt_tokens', None)
        t = getattr(usage_obj, 'total_tokens', None)
        
        return {
            'completion_tokens': int(c) if c is not None else None,
            'prompt_tokens': int(p) if p is not None else None,
            'total_tokens': int(t) if t is not None else None,
        }
        
    def __init__(
        self, 
        env: BilliardsTask,
        parser: Parser,
        feedback_manager: FeedbackManager, 
        max_tokens: int = 512,
        num_replans: int = 3, 
        use_feedback: bool = True,
        llm_source: str = "deepseek-v3.2", 
        temperature: float = 0.0
    ):
        self.env = env
        self.parser = parser
        self.feedback_manager = feedback_manager
        self.max_tokens = max_tokens
        self.num_replans = num_replans
        self.use_feedback = use_feedback
        self.temperature = temperature
        self.llm_source = llm_source
        assert llm_source in ["deepseek-v3.2", "deepseek-r1", "qwen-plus", "Moonshot-Kimi-K2-Instruct", "glm-4.5-air"], "Unsupported LLM source: {}".format(llm_source)

    def compose_system_prompt(
        self,
        balls: dict,
        my_targets: list, 
        table: dict,
        current_chat: list = [],
        feedback_history: list = []
    ) -> str:
        # Compose system prompt string

        # Get rules description
        rules_prompt = self.env.get_rules_prompt()
        # Get action output description
        action_desp = self.env.get_action_prompt()
        # Get calculation guide
        guide_prompt = self.env.get_guide_prompt()
        # Get agent role description
        agent_prompt = self.env.get_agent_prompt(balls, my_targets, table)
        # Combine into system prompt
        system_prompt = f"{rules_prompt}\n{action_desp}\n{guide_prompt}\n{agent_prompt}\n" 
        # Append environment feedback
        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += "\n".join(feedback_history)
        # Append current round dialog
        if len(current_chat) > 0:
            system_prompt += "[Current Chat]\n" + "\n".join(current_chat) + "\n"

        return system_prompt 

    def query_once(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        max_query: int = 3
    ) -> tuple:
        # Send one request to the LLM

        response = None # LLM 的原始响应
        usage = None # LLM 的使用情况
        # print('======= system prompt ======= \n ', system_prompt)
        # print('======= user prompt ======= \n ', user_prompt)

        # Try up to max_query attempts
        for n in range(max_query):
            print('querying {}th time'.format(n))
            try:
                resp = client.chat.completions.create(
                    model=self.llm_source,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                usage = resp.usage if hasattr(resp, 'usage') else None
                usage = self._usage_summary(usage)
                response = resp.choices[0].message.content.strip()
                print('======= response =======')
                print(response) if response else print('No response')
                print('======= usage =======')
                print(usage) if usage else print('No usage')
                break
            except Exception as e:
                print("API error, try again", e)
            continue

        return response, usage

    def prompt_one_round(
        self, 
        balls: dict = None,
        my_targets: list = None,
        table: dict = None,
        save_path: str = "data/"
    ) -> tuple: 
        # Run one planning round
        plan_feedbacks = [] # feedback list
        last_success_action = None # last successfully parsed action

        # Each round allows up to num_replans replans
        for i in range(self.num_replans):
            # Run one decision attempt
            final_response = self.prompt_one_dialog_round(
                balls,
                my_targets,
                table,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
            )
            
            # Parse final response
            parse_succ, parsed_str, action = self.parser.parse(balls, final_response) 
            if parse_succ and isinstance(action, dict):
                last_success_action = action
            curr_feedback = "None"
            # Parse failed: request reformat
            if not parse_succ:  
                curr_feedback = f"""
[Parse Feedback]
This previous response failed to parse!: '{final_response}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!
"""
                ready_to_execute = False  
            # Parse succeeded: get environment feedback via simulation
            else:
                ready_to_execute = True
                # Simulate shot and get feedback
                ready_to_execute, env_feedback = self.feedback_manager.give_feedback(action, balls, my_targets, table)
                if not ready_to_execute:
                    curr_feedback = env_feedback
            
            # Update feedback list
            plan_feedbacks.append(curr_feedback)
            
            # Save feedback and (possibly) parsed action
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (final_response if not parse_succ else action),
                },
            ]
            fname = f'{save_path}/replan{i}_feedback.json'
            json.dump(tosave, open(fname, 'w'), indent=4) 

            # If action is executable, stop replanning
            if ready_to_execute: 
                break  
            else:
                print(curr_feedback)

        return ready_to_execute, last_success_action, plan_feedbacks

    def prompt_one_dialog_round(
        self, 
        balls: dict,
        my_targets: list,
        table: dict,
        feedback_history: list = [], 
        replan_idx: int = 0,
        save_path: str = 'data/',
    ) -> str:
        # Run one decision round

        # Compose system prompt
        system_prompt = self.compose_system_prompt(
            balls, 
            my_targets,
            table,
            current_chat=[],
            feedback_history=feedback_history,   
            ) 
        
        # Compose user prompt
        user_prompt = f"As an expert billiards master, your response is: "
        user_prompt += "(Do NOT include chain-of-thought or internal reasoning. Output ONLY the final action following [Action Output Instruction]!)"
        
        # Send request to LLM
        response, usage = self.query_once(
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            max_query=3,
            )
        
        # Save the current response data
        tosave = [ 
            {
                "sender": "SystemPrompt",
                "message": system_prompt,
            },
            {
                "sender": "UserPrompt",
                "message": user_prompt,
            },
            {
                "sender": "LLMResponse",
                "message": response,
            },
            json.dumps(usage),
        ]
        fname = f'{save_path}/replan{replan_idx}.json'
        json.dump(tosave, open(fname, 'w'), indent=4)  

        pruned_response = response.strip() if response else ""
 
        # print(pruned_response)  
        return pruned_response