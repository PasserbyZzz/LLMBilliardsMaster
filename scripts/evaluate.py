import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import set_random_seed
from env.poolenv import PoolEnv
from agent.BasicAgent.basic_agent import BasicAgent
from agent.LLMAgent.llm_agent import LLMAgent
from agent.AlgorithmicAgent.AlgorithmicAgents import Enhanced_Bayes_Agent, MCTSAgent, EnsembleVotingAgent
# Set random seed
set_random_seed(enable=False, seed=42)

env = PoolEnv()
results = {'AGENT_A_WIN': 0, 'AGENT_B_WIN': 0, 'SAME': 0}
n_games = 40  # number of games

agent_a, agent_b = BasicAgent(), Enhanced_Bayes_Agent()

players = [agent_a, agent_b]  # turn order
target_ball_choice = ['solid', 'solid', 'stripe', 'stripe']  # ball type rotation

for i in range(n_games): 
    print()
    print(f"======= Game {i} start =======")
    env.reset(target_ball=target_ball_choice[i % 4])
    player_class_a = players[i % 2].__class__.__name__
    ball_type_a = target_ball_choice[i % 4]
    player_class_b = players[(i + 1) % 2].__class__.__name__
    ball_type_b = 'solid' if ball_type_a == 'stripe' else 'stripe'
    print(f"Player A: {player_class_a}, target type: {ball_type_a}")
    print(f"Player B: {player_class_b}, target type: {ball_type_b}")
    while True:
        player = env.get_curr_player()
        print(f"[Hit {env.hit_count}] Player: {player}")
        obs = env.get_observation(player)
        if player == 'A':
            action = players[i % 2].decision(*obs)
        else:
            action = players[(i + 1) % 2].decision(*obs)
        step_info = env.take_shot(action)
        
        done, info = env.get_done()
        if not done:
            if step_info.get('ENEMY_INTO_POCKET'):
                print(f"Opponent pocketed: {step_info['ENEMY_INTO_POCKET']}")
        if done:
            # Count results (map Player A/B back to Agent A/B)
            if info['winner'] == 'SAME':
                results['SAME'] += 1
            elif info['winner'] == 'A':
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][i % 2]] += 1
            else:
                results[['AGENT_A_WIN', 'AGENT_B_WIN'][(i+1) % 2]] += 1
            break

# Scoring: win=1, loss=0, draw=0.5
results['AGENT_A_SCORE'] = results['AGENT_A_WIN'] * 1 + results['SAME'] * 0.5
results['AGENT_B_SCORE'] = results['AGENT_B_WIN'] * 1 + results['SAME'] * 0.5

print("\nFinal results:", results)