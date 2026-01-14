import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Parser:
    def __init__(self):
        pass

    def parse(self, balls, response_text) -> tuple[bool, str, dict]:
        # Parse LLM response text and extract action parameters
        action = {}
        try:
            # Preprocess: remove possible Markdown code block markers
            clean_text = response_text.replace("```", "").strip()
            
            # Locate [RESPONSE] tag
            if "[RESPONSE]" in clean_text:
                # Only parse content after the tag
                target_text = clean_text.split("[RESPONSE]", 1)[1]
            else:
                # If tag missing, return failure
                return False, "Your response does not contain [RESPONSE] tag.", None
            
            lines = target_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line: 
                    continue
                
                # find Key: Value pattern
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) != 2: 
                        continue
                    
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # handle numeric parameters
                    if key in ['V0', 'phi', 'theta', 'a', 'b']:
                        try:
                            # extract first numeric value (ignore trailing comments)
                            # extract float using regex
                            num_match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", value_str)
                            if num_match:
                                action[key] = float(num_match.group())
                        except ValueError:
                            pass
                    # handle target field
                    elif key.lower() in ['target', 'targetball', 'target_ball']:
                        # allow numeric or quoted string (e.g., 3 or '3' or 8)
                        # only set Target when a numeric ID can be parsed
                        val = value_str.strip().strip("'\"")
                        num_match = re.search(r"\d+", val)
                        if num_match:
                            action['Target'] = str(num_match.group())
                    # handle reasoning text
                    elif key.lower() == 'reasoning':
                        action['reasoning'] = value_str

            # Validation
            if not action:
                return False, "Action is None or empty", None
            # key presence check
            required_keys = ['V0', 'phi', 'theta', 'a', 'b']
            for key in required_keys:
                if key not in action:
                    return False, f"Missing key: {key}", None
            # target existence check
            if 'Target' in action:
                if action['Target'] not in balls:
                    return False, f"Target {action['Target']} doesn't exist on the table.", None
            # range checks
            if not (0.5 <= action['V0'] <= 8.0):
                return False, f"V0={action['V0']} out of range [0.5, 8.0]", None
            if not (0 <= action['phi'] <= 360):
                return False, f"phi={action['phi']} out of range [0, 360]", None
            if not (0 <= action['theta'] <= 90):
                return False, f"theta={action['theta']} out of range [0, 90]", None
            if not (-1.0 <= action['a'] <= 1.0):
                return False, f"a={action['a']} out of range [-1.0, 1.0]", None
            if not (-1.0 <= action['b'] <= 1.0):
                return False, f"b={action['b']} out of range [-1.0, 1.0]", None

            # Passed all checks
            return True, "", action

        except Exception as e:
            return False, f"Exception during parsing: {str(e)}", None

if __name__ == "__main__":
    parser = Parser()

    print("="*20 + " Test Case 1: Standard Format " + "="*20)
    test_response_1 = """
    [RESPONSE]
    Reasoning: Standard shot.
    V0: 2.5
    phi: 45.0
    theta: 0.0
    a: 0.0
    b: -0.2
    """
    success, message, action = parser.parse({}, test_response_1)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")

    print("="*20 + " Test Case 2: With Chain of Thought " + "="*20)
    test_response_2 = """
    Thinking Process:
    1. Analyze table...
    2. Calculate angle...
    
    [RESPONSE]
    Reasoning: Shot after thinking.
    V0: 3.0
    phi: 90.0
    theta: 0.0
    a: 0.5
    b: 0.0
    """
    success, message, action = parser.parse({}, test_response_2)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")

    print("="*20 + " Test Case 3: No [RESPONSE] Tag (Legacy) " + "="*20)
    test_response_3 = """
    Reasoning: Legacy format.
    Target: 1
    V0: 4.0
    phi: 180.0
    theta: 0.0
    a: -0.5
    b: 0.5
    """
    success, message, action = parser.parse({}, test_response_3)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")

    print("="*20 + " Test Case 4: Invalid Values " + "="*20)
    test_response_4 = """
    [RESPONSE]
    Reasoning: Invalid shot.
    V0: 10.0
    phi: 400.0
    theta: 0.0
    a: 2.0
    b: 0.0
    """
    success, message, action = parser.parse({}, test_response_4)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")

    print("="*20 + " Test Case 5: With Target Present " + "="*20)
    test_response_5 = """
    [RESPONSE]
    Reasoning: Aim at a specific target ball.
    V0: 2.0
    phi: 30.0
    theta: 0.0
    a: 0.0
    b: 0.0
    Target: 3
    """
    balls = {'3': {}, 'cue': {}}
    success, message, action = parser.parse(balls, test_response_5)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")

    print("="*20 + " Test Case 6: Target Not On Table " + "="*20)
    test_response_6 = """
    [RESPONSE]
    Reasoning: Target is not on table.
    V0: 2.0
    phi: 30.0
    theta: 0.0
    a: 0.0
    b: 0.0
    Target: 99
    """
    balls2 = {'1': {}, 'cue': {}}
    success, message, action = parser.parse(balls2, test_response_6)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")

    print("="*20 + " Test Case 7: Safety / No Target " + "="*20)
    test_response_7 = """
    [RESPONSE]
    Reasoning: Play a safety, do not target a specific ball.
    Target: None
    V0: 1.0
    phi: 45.0
    theta: 0.0
    a: 0.0
    b: 0.0
    """
    success, message, action = parser.parse({'1': {}, 'cue': {}}, test_response_7)
    print(f"Success: {success}, Message: {message}\nAction: {action}\n")