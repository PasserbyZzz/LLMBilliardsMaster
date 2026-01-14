import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.utils import check_path_to_pocket

BALL_COMPOSITION = """
[Ball Composition]
1. CUE BALL
   - ID: 'cue'
2. SOLID BALLS
   - IDs: '1', '2', '3', '4', '5', '6', '7'
3. 8-BALL
   - ID: '8'   
4. STRIPE BALLS
   - IDs: '9', '10', '11', '12', '13', '14', '15'
"""

GAME_OBJECTIVE = """
[Game Objective]
1. First, pocket **all** balls of your assigned group (SOLID or STRIPE)
2. Then, pocket the 8-BALL to win the game
"""

SHOOTING_RULES = """
[SHOOTING RULES]
1. CONTINUE TURN: Pocket your own ball
2. SWITCH TURN: Fail to pocket your ball or commit TURN-LOSS FOULS
"""

FOUL_RULES = """
[FOUL RULES]
IMMEDIATE GAME-LOSS FOULS:
1. Both CUE BALL and 8-BALL are pocketed in same shot
2. Pocket 8-BALL before clearing your assigned group
TURN-LOSS FOULS:
1. CUE BALL pocketed alone or contacts no ball
2. CUE BALL hits opponent's ball or 8-BALL when group not cleared first
3. No balls pocketed and no cushion contact
"""

WIN_LOSS_DETERMINATION = """
[WIN/LOSS DETERMINATION]
1. WINNING CONDITIONS: Legally pocket 8-BALL after clearing your assigned group
2. LOSING CONDITIONS: Committ IMMEDIATE GAME-LOSS FOULS
"""

BILLIARDS_RULES = "GAME RULES: Standard 8-Ball pool rules with 16 balls." + BALL_COMPOSITION + GAME_OBJECTIVE + SHOOTING_RULES + FOUL_RULES + WIN_LOSS_DETERMINATION

BILLIARDS_ACTION_SPACE = """
[Action Output Instruction]
You must output your decision in a structured text format.
Each parameter should be on a separate line in the format "Key: Value".

Required Format:
[RESPONSE]
Reasoning: <a brief explanation of your strategy>
Target: <ball_id>   # Optional but recommended: the specific object ball you intend to pocket (e.g. 3 or 8)
V0: <value>
phi: <value>
theta: <value>
a: <value>
b: <value>

[Parameters Definition]
1. V0 (m/s): Cue stick impact velocity. Range [0.5, 8.0].
2. phi (degrees): Shooting angle on the table plane (XY plane). Range [0, 360).
   - Coordinate System: Origin (0,0) is at the Bottom-Left corner. X-axis extends to the right (Width), Y-axis extends upwards (Length).
   - 0 degrees: Shoots towards the Right Rail (+X direction).
   - 90 degrees: Shoots towards the Top Rail (+Y direction).
   - 180 degrees: Shoots towards the Left Rail (-X direction).
   - 270 degrees: Shoots towards the Bottom Rail (-Y direction).
3. theta (degrees): Cue elevation angle. Range [0, 90].
   - Definition: The angle between the cue stick and the table surface.
   - 0: Parallel to table (Flat shot, standard).
   - 90: Vertical strike (Masse shot).
   - Usually keep 0 unless necessary.
4. a (English): Horizontal offset. Range [-1, 1].
    - Definition: Horizontal contact offset expressed as a proportion of the cue ball radius.
    - Reference: From the shooter's perspective (looking down the cue stick).
    - -1: Hit the rightmost edge of the cue ball (offset = -1 * radius), producing maximum right English.
    - 0: Hit the center (No side spin).
    - 1: Hit the leftmost edge of the cue ball (offset = +1 * radius), producing maximum left English.
5. b (Vertical Spin): Vertical offset. Range [-1, 1].
    - Definition: Vertical contact offset expressed as a proportion of the cue ball radius.
    - Reference: From the shooter's perspective.
    - -1: Hit the bottom of the cue ball (Draw/Backspin) (offset = -1 * radius).
    - 0: Hit the center (No vertical spin).
    - 1: Hit the top of the cue ball (Follow/Topspin) (offset = +1 * radius).

Example Output:
[RESPONSE]
Reasoning: I will cut the 3-ball into the corner pocket with slight draw.
Target: 3
V0: 2.5
phi: 45.0
theta: 0.0
a: 0.0
b: -0.2
"""

CALCULATION_GUIDE = """
[Scientific Calculation Guide]
To ensure precision, you MUST use the following geometric formulas to calculate 'phi' and 'V0' rather than guessing:
1. Calculate Aim Point (Ghost Ball Position):
   - Let T = Target Ball Position (x, y)
   - Let P = Selected Pocket Position (x, y)
   - Let R = Ball Radius (0.0286 m)
   - Vector TP = P - T
   - Direction D = TP / length(TP)
   - Aim Point A = T - (D * 2 * R)  (The point where the cue ball must be when it contacts the target ball)
2. Calculate Shooting Angle (phi):
   - Let C = Cue Ball Position (x, y)
   - Vector CA = A - C
   - phi = degrees(atan2(CA.y, CA.x))
   - Ensure phi is in [0, 360) range.
3. Calculate Velocity (V0):
   - Let Distance d = length(T - C) (Distance between Cue Ball and Target Ball in meters)
   - If d < 0.3: V0 = 1.0
   - If 0.3 <= d < 0.6: V0 = 2.0
   - If 0.6 <= d < 1.0: V0 = 3.0
   - If 1.0 <= d < 1.5: V0 = 4.0
   - If d >= 1.5: V0 = 5.0
"""

class BilliardsTask:
    BALL_RADIUS = 0.0286
    
    def __init__(self):
        super().__init__()

    def get_rules_prompt(self) -> str:
        return BILLIARDS_RULES
    
    def get_action_prompt(self) -> str:
        return BILLIARDS_ACTION_SPACE
    
    def get_guide_prompt(self) -> str:
        return CALCULATION_GUIDE
    
    def get_path_analysis(self, balls, my_targets, table):
        # Generate path analysis report
        report = "[Path Analysis]\n"
        
        # Extract positions from ball objects
        balls_dict = {}
        cue_ball_obj = balls.get('cue')
        if cue_ball_obj and cue_ball_obj.state.s != 4:  # s==4 means pocketed
            pos = cue_ball_obj.state.rvw[0]
            balls_dict['cue'] = {'x': float(pos[0]), 'y': float(pos[1])}
        
        for ball_id, ball_obj in balls.items():
            if ball_id == 'cue':
                continue
            if ball_obj.state.s != 4:  # Not pocketed
                pos = ball_obj.state.rvw[0]
                balls_dict[ball_id] = {'x': float(pos[0]), 'y': float(pos[1])}
        
        if 'cue' not in balls_dict:
            return report + "Cue ball not found or pocketed.\n"
        
        cue_pos = np.array([balls_dict['cue']['x'], balls_dict['cue']['y'], 0])
        
        for target_id in my_targets:
            if target_id not in balls_dict:
                continue  # target already pocketed
            
            target_ball_data = balls_dict[target_id]
            target_pos = np.array([target_ball_data['x'], target_ball_data['y'], 0])
            
            pocket_results = []
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = np.array([pocket.center[0], pocket.center[1], 0])
                
                cue_blocked, target_blocked, blocking_cue, blocking_target = check_path_to_pocket(
                    cue_pos, target_pos, pocket_pos, balls, ball_radius=self.BALL_RADIUS
                )
                
                is_clear = not (cue_blocked or target_blocked)
                
                pocket_name = {
                    'lb': 'Left-Bottom',
                    'lc': 'Left-Center',
                    'lt': 'Left-Top',
                    'rb': 'Right-Bottom',
                    'rc': 'Right-Center',
                    'rt': 'Right-Top'
                }.get(pocket_id, pocket_id)
                
                if is_clear:
                    pocket_results.append(f"  Pocket '{pocket_name}': CLEAR")
                else:
                    blocking = blocking_cue + blocking_target
                    blocking_str = ", ".join(blocking)
                    pocket_results.append(f"  Pocket '{pocket_name}': BLOCKED by [{blocking_str}]")
            
            any_clear = any("CLEAR" in r for r in pocket_results)
            status = "AVAILABLE" if any_clear else "BLOCKED"
            
            report += f"- Target {target_id}: {status}\n"
            for result in pocket_results:
                report += result + "\n"
        
        return report
    
    def get_obs(self, balls, my_targets, table) -> dict:
        # 构建观测字典
        balls_data = {}
        
        # 统计剩余的球
        remaining_solids = []
        remaining_stripes = []
        
        # 实心球 ID 范围 '1'-'7'
        solid_ids = [str(i) for i in range(1, 8)]
        # 条纹球 ID 范围 '9'-'15'
        stripe_ids = [str(i) for i in range(9, 16)]

        # 获取观测：场上剩余球的坐标
        for bid, ball in balls.items():
            if ball.state.s != 4:
                pos = ball.state.rvw[0]
                balls_data[bid] = {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                }
                
                if bid in solid_ids:
                    remaining_solids.append(bid)
                elif bid in stripe_ids:
                    remaining_stripes.append(bid)

        return {
            "balls": balls_data,
            "my_targets": my_targets,
            "table": table,
            "remaining_solids": remaining_solids,
            "remaining_stripes": remaining_stripes
        }
    
    def get_agent_prompt(self, balls, my_targets, table) -> str:
        # 4. Path Analysis
        path_analysis = self.get_path_analysis(balls, my_targets, table)
        
        obs = self.get_obs(balls, my_targets, table)
        balls = obs['balls']
        table = obs['table']
        my_targets = obs['my_targets']
        
        # 1. Table Info
        pocket_names = {
            'lb': 'Left-Bottom',
            'lc': 'Left-Center',
            'lt': 'Left-Top',
            'rb': 'Right-Bottom',
            'rc': 'Right-Center',
            'rt': 'Right-Top'
        }
        pockets_str = ", ".join([f"Pocket '{pocket_names.get(pid, pid)}' at ({p.center[0]:.3f}, {p.center[1]:.3f})" for pid, p in table.pockets.items()])
        table_desc = (
            f"You are an expert billiards master playing on a table of size {table.w:.3f}m (Width) by {table.l:.3f}m (Length). "
            f"The coordinate system is defined with the origin (0,0) at the Bottom-Left corner of the playing surface. "
            f"The X-axis extends to the right (Width), and the Y-axis extends upwards (Length). "
            f"The pockets are located at: {pockets_str}."
        )

        # 2. Groups Info
        my_group_type = "8-BALL"
        opp_group_type = "Unknown"
        
        # Determine Group Types
        first_target = next((t for t in my_targets if t != '8'), None)
        if first_target:
            if 1 <= int(first_target) <= 7:
                my_group_type = "SOLID"
                opp_group_type = "STRIPE"
            else:
                my_group_type = "STRIPE"
                opp_group_type = "SOLID"
        
        # Calculate My Remaining Targets
        my_remaining = [bid for bid in my_targets if bid in balls]
        # If no group balls remaining, target is 8-ball
        if not my_remaining and '8' in balls:
            my_remaining = ['8']
            
        # Calculate Opponent Remaining Targets
        opp_remaining = []
        if opp_group_type == "SOLID":
            opp_remaining = obs['remaining_solids']
        elif opp_group_type == "STRIPE":
            opp_remaining = obs['remaining_stripes']
        # If opponent has no group balls, they are also on 8-ball
        if not opp_remaining and '8' in balls:
            opp_remaining = ['8']

        objective_desc = f"Your objective is to pocket the {my_group_type} balls (Remaining Targets: {my_remaining}), while your opponent targets the {opp_group_type} balls (Remaining Targets: {opp_remaining})."

        # 3. Balls Info
        cue_ball = balls.get('cue')
        cue_str = f"({cue_ball['x']:.3f}, {cue_ball['y']:.3f})" if cue_ball else "Pocketed"
        balls_desc = f"Currently, the Cue Ball is at {cue_str}."

        def format_balls(bids):
            # 按数字大小排序 ID
            sorted_bids = sorted(bids, key=lambda x: int(x))
            items = []
            for bid in sorted_bids:
                if bid in balls:
                    items.append(f"ID {bid} at ({balls[bid]['x']:.3f}, {balls[bid]['y']:.3f})")
            return ", ".join(items)

        if obs['remaining_solids']:
            solids_str = format_balls(obs['remaining_solids'])
            balls_desc += f" The remaining SOLID balls are located at: {solids_str}."
        
        if obs['remaining_stripes']:
            stripes_str = format_balls(obs['remaining_stripes'])
            balls_desc += f" The remaining STRIPE balls are located at: {stripes_str}."
        
        eight_ball = balls.get('8')
        if eight_ball:
            balls_desc += f" The 8-Ball (ID 8) is at ({eight_ball['x']:.3f}, {eight_ball['y']:.3f})."
        else:
            balls_desc += " The 8-Ball (ID 8) is Pocketed."

        agent_prompt = f"{table_desc}\n{objective_desc}\n{balls_desc}\n\n"
        
        # 4. Path Analysis
        agent_prompt += path_analysis + "\n"
        
        agent_prompt += (
            "Please first visualize the 2D plane of the table and balls in your mind.\n"
            "Strategic Tips:\n"
            "1. Path Check: Refer to the [Path Analysis] above. Only target balls with CLEAR pockets to avoid failures.\n"
            "2. Calculation: For your selected target and pocket, strictly follow the [Scientific Calculation Guide] to calculate 'phi' and 'V0'.\n"
            "3. Bank Shots: If all pockets are BLOCKED, consider using rails (cushions) to rebound the cue ball.\n"
            "4. Spin Control: Use 'a' (side spin) and 'b' (vertical spin) to control the cue ball's post-impact path (position play) or to swerve around obstacles.\n"
            "Improve your plan if given [Simulation Feedback], which is the simulation result of YOUR previous action.\n"
            "Finally, output your final plan which strictly follows [Action Output Instruction]!"
        )

        return agent_prompt
        
if __name__ == "__main__":
    from env.poolenv import PoolEnv
    
    task = BilliardsTask()
    env = PoolEnv()
    env.reset(target_ball='solid')
    # task.reset(target_ball='stripe')
    
    print("="*30 + " Rules Prompt " + "="*30)
    print(task.get_rules_prompt())
    
    print("="*30 + " Action Prompt " + "="*30)
    print(task.get_action_prompt())
    
    print("="*30 + " Agent Prompt " + "="*30)
    balls, my_targets, table = env.get_observation()
    prompt = task.get_agent_prompt(balls, my_targets, table)
    print(prompt)