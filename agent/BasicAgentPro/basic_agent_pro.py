import math
import pooltool as pt
import numpy as np
import copy
import random
import signal

from agent.agent import Agent

# ============ Simulation timeout protection ============
class SimulationTimeoutError(Exception):
    """Physics simulation timeout"""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise SimulationTimeoutError("Physics simulation timed out")


def simulate_with_timeout(shot, timeout=3):
    """Run physics simulation with a timeout guard.

    Args:
        shot: pt.System object
        timeout: seconds (default 3)

    Returns:
        bool: True if simulation succeeded, False on timeout or failure

    Note:
        Uses signal.SIGALRM and therefore works only on Unix-like systems.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] Physics simulation timed out (>{timeout}s), skipping simulation")
        return False
    except Exception:
        signal.alarm(0)
        raise
    finally:
        signal.signal(signal.SIGALRM, old_handler)

# ============ Reward and BasicAgentPro ============
def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """Analyze shot result and compute reward score.

    Returns:
        float: reward score (higher is better)
    """
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]

    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # first contact analysis
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # Filter out 'cue' and non-ball objects (e.g., 'cue stick'), keep only valid ball IDs
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    # foul first hit analysis: fully aligned with player_targets
    if first_contact_ball_id is None:
        # No ball hit (but if only cue and 8 remain and cleared, no foul)
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # The first ball hit must be one of the player_targets
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    # cushion analysis
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # scoring
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class BasicAgentPro(Agent):
    """Advanced MCTS-based agent."""
    def __init__(self,
                 n_simulations=50,       # number of simulations per decision
                 c_puct=1.414):          # exploration constant for UCB
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575
        
        # Noise levels for simulation (approximate human / env error).
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

        print("BasicAgentPro (MCTS) initialized — let the simulated chaos begin!")

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table):
        """Generate candidate actions using simple, efficient heuristics."""
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # collect IDs of target balls
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # fallback to 8 if no targets remain (edge-case safeguard)
        if not target_ids:
            target_ids = ['8']

        # iterate over each target ball
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # iterate over each pocket
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 1. compute ideal pocketing angle (ghost-ball)
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. estimate base force from distance (simple linear heuristic)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. add a few action variants to the candidate pool
                # variant 1: accurate shot
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # variant 2: slightly stronger shot
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # variant 3: small angle tweak (±0.5°) to handle noise
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })

        # If heuristics produce nothing (rare), add fallback random actions.
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # shuffle candidate actions
        random.shuffle(actions)
        return actions[:30]

    def simulate_action(self, balls, table, action):
        """
        [change] Run physics simulation with injected noise.
        This helps the agent learn that some "perfect" shots fail in reality.
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # inject Gaussian noise
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        
        # preprocess
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0: my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # generate candidate actions
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # MCTS 循环
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                # compute UCB values using normalized Q
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation (with noise)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            # normalize to [0,1] using expected min/max
            normalized_reward = (raw_reward - (-500)) / 650.0
            # clip to valid range
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward  # accumulate normalized reward

        # Final decision
        # choose action with highest average reward (robust child)
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        # log best average score
        print(f"[BasicAgent_pro] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return best_action