
import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent
import signal
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


# ============ Timeout-safe simulation mechanism ============
class SimulationTimeoutError(Exception):
    """Physics simulation timeout exception"""
    pass

def _timeout_handler(signum, frame):
    """Timeout signal handler"""
    raise SimulationTimeoutError("Physics simulation timed out")

def simulate_with_timeout(shot, timeout=3):
    """Physics simulation with a timeout safeguard
    
    Args:
        shot: pt.System instance
        timeout: Timeout in seconds (default: 3)
    
    Returns:
        bool: True if the simulation succeeds; False if it times out or fails
    
    Notes:
        Uses signal.SIGALRM to enforce a timeout (Unix/Linux only).
        Automatically recovers after a timeout to avoid the program hanging.
    """
    # Install timeout signal handler
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # Set timeout
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # Cancel timeout
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] Physics simulation timed out (>{timeout}s); skipping this simulation")
        return False
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # Restore previous handler

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    Analyze the shot outcome and compute a reward score (aligned with eight-ball rules)
    
    Args:
        shot: System instance after physics simulation
        last_state: Ball states before the shot, {ball_id: Ball}
        player_targets: Current player's target ball IDs, ['1','2',...] or ['8']
    
    Returns:
        float: Reward score
            +50 per own ball pocketed, +100 for a legal 8-ball, +10 for a legal no-pot shot
            -100 for cue ball pocketed, -500 for an illegal 8-ball / cue+8, -30 for first-hit/rail foul
    
    Core rules:
        - Before clearing your set: player_targets = ['1'-'7'] or ['9'-'15']; the 8-ball belongs to neither player
        - After clearing your set: player_targets = ['8']; the 8-ball becomes the only target
    """
    
    # 1. Basic analysis
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # Determine ownership based on player_targets (the 8-ball counts as "own" only after clearing)
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. Analyze first contact (define a valid set of ball IDs)
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # Filter out 'cue' and non-ball objects (e.g., 'cue stick'); keep only valid ball IDs
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # First-hit foul: strictly aligned with player_targets
    if first_contact_ball_id is None:
        # Did not hit any ball (except when only cue+8 remain and the table is cleared)
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # The first ball contacted must be in player_targets
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. Analyze rail contact
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
        
    # Compute reward score
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

class Agent():
    """Agent base class"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """Decision method (must be implemented by subclasses)
        
        Returns: dict with keys 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """Generate a random shot action
        
        Returns: dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] degrees
            theta: [0, 90] degrees
            a, b: [-0.5, 0.5] as a fraction of ball radius
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # Initial speed: 0.5–8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # Horizontal angle (0°–360°)
            'theta': round(random.uniform(0, 90), 2),   # Vertical angle
            'a': round(random.uniform(-0.5, 0.5), 3),   # Tip lateral offset (fraction of ball radius)
            'b': round(random.uniform(-0.5, 0.5), 3)    # Tip vertical offset (fraction of ball radius)
        }
        return action


class Enhanced_Bayes_Agent(Agent): 
    """NewAgent (geometry-guided + Bayesian optimization + enhanced safety strategy)"""
    
    def __init__(self):
        super().__init__()
        
        # Import geometry utilities
        from utils.utils import (
            select_best_target,
            calculate_aim_point_for_pocket,
            calculate_angle_to_aim_point,
            calculate_distance,
            calculate_recommended_velocity,
            check_eight_ball_in_path,
            check_other_balls_in_path,
            is_point_near_line_segment,
            check_cue_ball_pocket_risk,
            predict_first_contact_ball,
            check_eight_ball_scratch_risk
        )
        self.select_best_target = select_best_target
        self.calculate_aim_point_for_pocket = calculate_aim_point_for_pocket
        self.calculate_angle_to_aim_point = calculate_angle_to_aim_point
        self.calculate_distance = calculate_distance
        self.calculate_recommended_velocity = calculate_recommended_velocity
        self.check_eight_ball_in_path = check_eight_ball_in_path
        self.check_other_balls_in_path = check_other_balls_in_path
        self.is_point_near_line_segment = is_point_near_line_segment
        self.check_cue_ball_pocket_risk = check_cue_ball_pocket_risk
        self.predict_first_contact_ball = predict_first_contact_ball
        self.check_eight_ball_scratch_risk = check_eight_ball_scratch_risk
        
        # Bayesian optimization parameters (tuned)
        self.INITIAL_SEARCH = 30  # Increase initial sampling
        self.OPT_SEARCH = 15    # Increase optimization iterations
        self.ALPHA = 5e-4         # Reduce noise
        
        # Geometry search parameters (tightened search bounds)
        self.phi_range = 40       # Angle search range ±40° (reduced from 75°)
        self.V0_range = 2.0       # Speed search range ±2.0 m/s (reduced from 3.0)
        self.theta_max = 45       # Max elevation angle (reduced from 60°)
        self.offset_range = 0.30  # Tip offset range (reduced from 0.40)
        
        # 8-ball avoidance parameters (stronger penalties)
        self.eight_ball_penalty = 400    # Penalty for illegally hitting the 8-ball (increased)
        self.wrong_ball_penalty = 250    # Penalty for hitting an opponent ball first (increased)
        self.cue_pocket_penalty = 350    # Penalty for cue ball pocketed (added)
        self.scratch_eight_penalty = 600 # Penalty for cue ball + 8-ball pocketed together (added, fatal)
        
        # Conservative parameters when shooting the 8-ball
        self.eight_ball_V0_max = 5.0     # Max speed when shooting the 8-ball
        self.eight_ball_theta_max = 25   # Max elevation when shooting the 8-ball
        self.eight_ball_offset_range = 0.20  # Tip offset range when shooting the 8-ball
        
        print("Enhanced_BayesAgent (Geometry-Guided Bayesian Optimization v2) initialized.")
    
    def print_config(self):
        """Print all important hyperparameters"""
        print("\n" + "="*50)
        print("[Enhanced_BayesAgent Hyperparameter Configuration]")
        print("  Bayesian optimization parameters:")
        print(f"    - INITIAL_SEARCH (initial random samples): {self.INITIAL_SEARCH}")
        print(f"    - OPT_SEARCH (BO iterations): {self.OPT_SEARCH}")
        print(f"    - ALPHA (GP noise): {self.ALPHA}")
        print("  Geometry heuristic parameters:")
        print(f"    - phi_range (angle search range): ±{self.phi_range}°")
        print(f"    - V0_range (speed search range): ±{self.V0_range} m/s")
        print(f"    - theta_max (max elevation): {self.theta_max}°")
        print(f"    - offset_range (tip offset): ±{self.offset_range}")
        print("  Penalty parameters:")
        print(f"    - eight_ball_penalty (illegal 8-ball hit): {self.eight_ball_penalty}")
        print(f"    - wrong_ball_penalty (wrong first ball): {self.wrong_ball_penalty}")
        print(f"    - cue_pocket_penalty (cue ball pocketed): {self.cue_pocket_penalty}")
        print(f"    - scratch_eight_penalty (cue + 8 pocketed): {self.scratch_eight_penalty}")
        print("  8-ball special parameters:")
        print(f"    - eight_ball_V0_max: {self.eight_ball_V0_max} m/s")
        print(f"    - eight_ball_theta_max: {self.eight_ball_theta_max}°")
        print(f"    - eight_ball_offset_range: ±{self.eight_ball_offset_range}")
        print("="*50)
    
    def _analyze_first_contact(self, shot, valid_ball_ids):
        """Analyze the first collision after the shot
        
        Returns:
            first_contact_ball_id: ID of the first ball contacted
        """
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    return other_ids[0]
        return None
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Geometry-guided decision (enhanced v2: cue-ball safety + safe 8-ball play)
        
        Steps:
        1. Use geometry to choose the easiest target ball and pocket (considering the 8-ball)
        2. Compute the ideal aiming angle
        3. Geometry pre-checks: first-contact validation and cue-ball scratch risk
        4. Run Bayesian optimization around the aiming angle (with stronger penalties)
        5. Use a conservative strategy when shooting the 8-ball
        
        Args:
            balls: Ball state dictionary
            my_targets: List of target ball IDs
            table: Table object
        
        Returns:
            dict: Shot action
        """
        if balls is None or my_targets is None or table is None:
            print("[NewAgent] Missing required inputs; using a random action.")
            return self._random_action()
        
        try:
            # Save state snapshot
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Check whether all target balls are already pocketed
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_eight = False
            if len(remaining_own) == 0:
                my_targets = ["8"]
                is_shooting_eight = True
                print("[NewAgent] All target balls cleared; switching to the 8-ball. ⚠️ Entering cautious mode!")
            
            # Step 1: Geometry-based target selection (avoid the 8-ball)
            cue_pos = balls['cue'].state.rvw[0]
            best_target_id, best_pocket_id, difficulty = self.select_best_target(
                cue_pos, my_targets, balls, table, avoid_eight=True
            )
            
            if best_target_id is None:
                print("[NewAgent] No valid target found; using a random action.")
                return self._random_action()
            
            target_pos = balls[best_target_id].state.rvw[0]
            pocket_pos = table.pockets[best_pocket_id].center
            
            # Check whether the 8-ball lies near the path (to adjust strategy)
            eight_in_path, eight_dist = self.check_eight_ball_in_path(cue_pos, target_pos, balls)
            
            # Check cue-ball scratch risk
            cue_pocket_risk, risky_pocket = self.check_cue_ball_pocket_risk(cue_pos, target_pos, table)
            
            # Geometric prediction of first contact ball
            predicted_first, _ = self.predict_first_contact_ball(
                cue_pos, 
                self.calculate_angle_to_aim_point(cue_pos, self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)),
                balls
            )
            
            warning_msg = ""
            if eight_in_path:
                warning_msg += " (Warning: 8-ball near the path!)"
            if cue_pocket_risk > 0.5:
                warning_msg += f" (Warning: high cue-ball scratch risk={cue_pocket_risk:.2f})"
            if predicted_first and predicted_first != best_target_id:
                warning_msg += f" (Warning: predicted first contact={predicted_first})"
            
            print(f"[NewAgent] Selected target: {best_target_id} → pocket: {best_pocket_id}, difficulty: {difficulty:.2f}{warning_msg}")
            
            # Step 2: Compute geometric aiming parameters
            aim_point = self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)
            
            if aim_point is None:
                print("[NewAgent] Failed to compute aim point; using a random action.")
                return self._random_action()
            
            # Compute ideal angle
            ideal_phi = self.calculate_angle_to_aim_point(cue_pos, aim_point)
            
            # Compute recommended speed
            distance = self.calculate_distance(cue_pos, target_pos)
            ideal_V0 = self.calculate_recommended_velocity(distance)
            
            # When shooting the 8-ball, perform an extra scratch-risk check
            scratch_risk = 0.0
            if is_shooting_eight:
                scratch_risk, risk_type = self.check_eight_ball_scratch_risk(
                    cue_pos, target_pos, pocket_pos, balls
                )
                if scratch_risk > 0.3:
                    print(f"[NewAgent] ⚠️ 8-ball scratch risk: {scratch_risk:.2f} ({risk_type}); reducing power")
                    ideal_V0 = min(ideal_V0, self.eight_ball_V0_max * (1 - scratch_risk * 0.5))
            
            print(f"[NewAgent] Geometric solution: phi={ideal_phi:.2f}°, V0={ideal_V0:.2f} m/s, distance={distance:.3f}m")
            
            # Step 3: Build the search space (adjust for path risk and 8-ball shots)
            if is_shooting_eight:
                # Use more conservative parameters when shooting the 8-ball
                phi_range = self.phi_range * 0.4
                V0_range = self.V0_range * 0.5
                V0_max = self.eight_ball_V0_max
                theta_max = self.eight_ball_theta_max
                offset_range = self.eight_ball_offset_range
                print(f"[NewAgent] 8-ball mode: tightened bounds phi±{phi_range:.1f}°, V0±{V0_range:.1f}, θ≤{theta_max}°")
            elif eight_in_path:
                phi_range = self.phi_range * 0.5
                V0_range = self.V0_range * 0.7
                V0_max = 8.0
                theta_max = self.theta_max
                offset_range = self.offset_range
            else:
                phi_range = self.phi_range
                V0_range = self.V0_range
                V0_max = 8.0
                theta_max = self.theta_max
                offset_range = self.offset_range
            
            # If cue-ball scratch risk is high, tighten further
            if cue_pocket_risk > 0.5:
                V0_max = min(V0_max, 3.5)
                theta_max = min(theta_max, 25)
                offset_range *= 0.7
            
            # Add a safety bounds function to avoid collapsing to a single point
            def _safe_bounds(center, span, min_v, max_v, eps=1e-3):
                low = max(min_v, center - span)
                high = min(max_v, center + span)
                if high - low < eps:
                    mid = (low + high) / 2
                    low = max(min_v, mid - eps / 2)
                    high = min(max_v, mid + eps / 2)
                    if high - low < eps:  # Expand slightly in extreme cases
                        high = min(max_v, low + eps)
                return low, high

            V0_low, V0_high = _safe_bounds(ideal_V0, V0_range, 0.8, V0_max)  # Raise min power to avoid numerical issues
            phi_low, phi_high = _safe_bounds(ideal_phi, phi_range, -720.0, 1080.0)  # Broaden range; will mod internally

            pbounds = {
                'V0': (V0_low, V0_high),
                'phi': (phi_low, phi_high),
                'theta': (0, max(theta_max, 1e-2)),
                'a': (-offset_range, offset_range),
                'b': (-offset_range, offset_range)
            }
            
            # Step 4: Define enhanced reward (first contact, cue scratch, cue+8 pocketed)
            valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
            
            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    # Normalize angle bounds
                    phi_normalized = phi % 360
                    shot.cue.set_state(V0=V0, phi=phi_normalized, theta=theta, a=a, b=b)
                    
                    # Use timeout protection (avoid the physics engine hanging)
                    if not simulate_with_timeout(shot, timeout=15):
                        return 0  # On timeout, return a neutral score
                except Exception as e:
                    return -500
                
                # Base score
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                # Additional checks: inspect simulation outcome
                first_contact = self._analyze_first_contact(shot, valid_ball_ids)
                
                # Check whether both cue ball and 8-ball are pocketed (fatal foul)
                cue_pocketed = shot.balls['cue'].state.s == 4
                eight_pocketed = '8' in shot.balls and shot.balls['8'].state.s == 4 and last_state_snapshot['8'].state.s != 4
                
                if cue_pocketed and eight_pocketed:
                    # Cue ball + 8-ball pocketed together: most severe foul
                    score -= self.scratch_eight_penalty
                elif cue_pocketed:
                    # Cue ball pocketed only
                    score -= self.cue_pocket_penalty
                
                # First-contact penalty
                if first_contact is not None:
                    if first_contact not in my_targets:
                        if first_contact == '8' and my_targets != ['8']:
                            score -= self.eight_ball_penalty  # Illegal 8-ball hit
                        else:
                            score -= self.wrong_ball_penalty  # Hit opponent ball first
                
                # Extra protection when shooting the 8-ball: penalize excessive speed even without a pot
                if is_shooting_eight and V0 > self.eight_ball_V0_max:
                    score -= (V0 - self.eight_ball_V0_max) * 20
                
                return score
            
            # Step 5: Bayesian optimization
            print("[NewAgent] Optimizing near the geometric solution...")
            
            gpr = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=self.ALPHA,
                n_restarts_optimizer=10,
                random_state=np.random.randint(1e6)
            )
            
            optimizer = BayesianOptimization(
                f=reward_fn_wrapper,
                pbounds=pbounds,
                random_state=np.random.randint(1e6),
                verbose=0
            )
            optimizer._gp = gpr
            
            # Add the geometric solution as an initial probe point
            optimizer.probe(
                params={
                    'V0': np.clip(ideal_V0, V0_low, V0_high),
                    'phi': np.clip(ideal_phi, phi_low, phi_high),
                    'theta': 5.0,  # Small elevation
                    'a': 0.0,
                    'b': 0.0
                },
                lazy=True
            )
            
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']
            
            if best_score < 10:
                print(f"[NewAgent] Optimized score is still low ({best_score:.2f}); using the geometric solution.")
                # Fall back to the geometric solution (use a more conservative speed for the 8-ball)
                fallback_V0 = min(ideal_V0, self.eight_ball_V0_max) if is_shooting_eight else ideal_V0
                action = {
                    'V0': fallback_V0,
                    'phi': ideal_phi,
                    'theta': 5.0,
                    'a': 0.0,
                    'b': 0.0
                }
            else:
                action = {
                    'V0': float(best_params['V0']),
                    'phi': float(best_params['phi']) % 360,
                    'theta': float(best_params['theta']),
                    'a': float(best_params['a']),
                    'b': float(best_params['b'])
                }
            
            print(f"[NewAgent] Final decision (score: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[NewAgent] Error during decision: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class GeometricAgent(Agent):
    """Pure geometry decision agent (no Bayesian optimization)
    
    Characteristics:
    - Fully geometry-based; no physics-simulation optimization
    - Very fast, suitable for real-time decision-making
    - Uses heuristic rules to improve safety and accuracy
    """
    
    def __init__(self):
        super().__init__()
        
        # Import geometry utilities
        from utils.utils import (
            select_best_target,
            calculate_aim_point_for_pocket,
            calculate_angle_to_aim_point,
            calculate_distance,
            calculate_recommended_velocity,
            check_eight_ball_in_path,
            check_cue_ball_pocket_risk,
            predict_first_contact_ball,
            check_eight_ball_scratch_risk
        )
        self.select_best_target = select_best_target
        self.calculate_aim_point_for_pocket = calculate_aim_point_for_pocket
        self.calculate_angle_to_aim_point = calculate_angle_to_aim_point
        self.calculate_distance = calculate_distance
        self.calculate_recommended_velocity = calculate_recommended_velocity
        self.check_eight_ball_in_path = check_eight_ball_in_path
        self.check_cue_ball_pocket_risk = check_cue_ball_pocket_risk
        self.predict_first_contact_ball = predict_first_contact_ball
        self.check_eight_ball_scratch_risk = check_eight_ball_scratch_risk
        
        # Geometry parameter adjustment factors
        self.angle_adjustment_factor = 0.98  # Angle fine-tuning factor (counter systematic bias)
        self.velocity_safety_factor = 0.9    # Speed safety factor (avoid excessive power)
        
        # Conservative parameters when shooting the 8-ball
        self.eight_ball_velocity_limit = 4.5  # Max speed for the 8-ball
        self.eight_ball_theta_limit = 20      # Max elevation for the 8-ball
        
        # Default parameters
        self.normal_theta_max = 35            # Max elevation for normal shots
        self.risky_velocity_limit = 3.0       # Speed limit for high-risk situations
        
        print("GeometricAgent (Pure Geometry, No Bayesian Optimization) initialized.")
    
    def print_config(self):
        """Print all important hyperparameters"""
        print("\n" + "="*50)
        print("[GeometricAgent Hyperparameter Configuration]")
        print("  Geometry adjustment parameters:")
        print(f"    - angle_adjustment_factor (angle fine-tuning): {self.angle_adjustment_factor}")
        print(f"    - velocity_safety_factor (speed safety): {self.velocity_safety_factor}")
        print("  8-ball special parameters:")
        print(f"    - eight_ball_velocity_limit: {self.eight_ball_velocity_limit} m/s")
        print(f"    - eight_ball_theta_limit: {self.eight_ball_theta_limit}°")
        print("  General shot parameters:")
        print(f"    - normal_theta_max (max elevation for normal shots): {self.normal_theta_max}°")
        print(f"    - risky_velocity_limit (speed limit in high-risk cases): {self.risky_velocity_limit} m/s")
        print("  Notes: pure geometry, no Bayesian optimization, very fast")
        print("="*50)
    
    def _calculate_cut_angle_adjustment(self, cue_pos, target_pos, pocket_pos):
        """Compute cut-shot angle adjustments
        
        Based on cue–object–pocket geometry, compute a suitable cut angle.
        Returns recommended theta and tip-offset adjustments.
        """
        # Compute the angle between the incident and outgoing vectors
        vec_in = (target_pos - cue_pos)[:2]
        vec_out = (pocket_pos - target_pos)[:2]
        
        vec_in_norm = vec_in / (np.linalg.norm(vec_in) + 1e-9)
        vec_out_norm = vec_out / (np.linalg.norm(vec_out) + 1e-9)
        
        dot_product = np.clip(np.dot(vec_in_norm, vec_out_norm), -1.0, 1.0)
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Choose theta and offsets based on the cut angle
        if angle_deg < 15:  # Straight-in shot
            theta = 3.0
            a_offset = 0.0
            b_offset = 0.0
        elif angle_deg < 45:  # Small cut shot
            theta = 8.0
            # Slight sidespin helps with cue-ball control
            a_offset = 0.05 if angle_deg < 30 else 0.10
            b_offset = 0.0
        elif angle_deg < 90:  # Medium cut shot
            theta = 15.0
            a_offset = 0.15
            b_offset = 0.05
        else:  # Large cut shot (difficult)
            theta = 25.0
            a_offset = 0.20
            b_offset = 0.10
        
        return theta, a_offset, b_offset, angle_deg
    
    def _adjust_for_distance(self, distance, base_velocity):
        """Fine-tune speed based on distance
        
        Account for friction losses: the farther the distance, the more speed is needed.
        """
        if distance < 0.3:
            # Short distance: gentle stroke
            return base_velocity * 0.85
        elif distance < 0.6:
            return base_velocity * 0.95
        elif distance < 1.0:
            return base_velocity
        elif distance < 1.5:
            # Medium distance: slightly increase power
            return base_velocity * 1.1
        else:
            # Long distance: significantly increase power, but do not exceed a safe upper bound
            return min(base_velocity * 1.25, 6.5)
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Pure geometry decision (no Bayesian optimization)
        
        Steps:
        1. Choose the best target ball and pocket
        2. Compute geometric aim point and angle
        3. Compute speed and shot parameters from distance, angle, and risk
        4. Apply safety checks and adjustments
        5. Return the geometric solution directly (no optimization)
        """
        if balls is None or my_targets is None or table is None:
            print("[GeometricAgent] Missing required inputs; using a random action.")
            return self._random_action()
        
        try:
            # Check whether all target balls are already pocketed
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_eight = False
            if len(remaining_own) == 0:
                my_targets = ["8"]
                is_shooting_eight = True
                print("[GeometricAgent] All target balls cleared; switching to the 8-ball. ⚠️ Entering conservative mode!")
            
            # Step 1: Choose target ball and pocket
            cue_pos = balls['cue'].state.rvw[0]
            best_target_id, best_pocket_id, difficulty = self.select_best_target(
                cue_pos, my_targets, balls, table, avoid_eight=True
            )
            
            if best_target_id is None:
                print("[GeometricAgent] No valid target found; using a random action.")
                return self._random_action()
            
            target_pos = balls[best_target_id].state.rvw[0]
            pocket_pos = table.pockets[best_pocket_id].center
            
            # Step 2: Risk assessment
            eight_in_path, eight_dist = self.check_eight_ball_in_path(cue_pos, target_pos, balls)
            cue_pocket_risk, risky_pocket = self.check_cue_ball_pocket_risk(cue_pos, target_pos, table)
            
            # Step 3: Compute geometric aiming parameters
            aim_point = self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)
            
            if aim_point is None:
                print("[GeometricAgent] Failed to compute aim point; using a random action.")
                return self._random_action()
            
            # Compute ideal angle
            ideal_phi = self.calculate_angle_to_aim_point(cue_pos, aim_point)
            
            # Apply angle adjustment factor
            phi = ideal_phi * self.angle_adjustment_factor
            
            # Step 4: Compute speed
            distance = self.calculate_distance(cue_pos, target_pos)
            base_velocity = self.calculate_recommended_velocity(distance)
            
            # Fine-tune by distance
            velocity = self._adjust_for_distance(distance, base_velocity)
            
            # Apply speed safety factor
            velocity = velocity * self.velocity_safety_factor
            
            # Step 5: Compute cut-shot parameters
            theta, a_offset, b_offset, cut_angle = self._calculate_cut_angle_adjustment(
                cue_pos, target_pos, pocket_pos
            )
            
            # Step 6: Adjust for special conditions
            warning_msg = ""
            
            # Special handling when shooting the 8-ball
            if is_shooting_eight:
                scratch_risk, risk_type = self.check_eight_ball_scratch_risk(
                    cue_pos, target_pos, pocket_pos, balls
                )
                
                # Highly conservative 8-ball strategy
                velocity = min(velocity, self.eight_ball_velocity_limit)
                theta = min(theta, self.eight_ball_theta_limit)
                
                # If scratch risk is high, reduce speed further
                if scratch_risk > 0.3:
                    velocity = velocity * (1.0 - scratch_risk * 0.5)
                    warning_msg += f" (scratch risk={scratch_risk:.2f})"
                
                # Use a more centered hit for the 8-ball
                a_offset = a_offset * 0.5
                b_offset = b_offset * 0.5
                
                print(f"[GeometricAgent] 8-ball mode: V0={velocity:.2f}, θ={theta:.2f}°{warning_msg}")
            
            # Handling when the 8-ball is on/near the path
            elif eight_in_path:
                # Try to avoid the 8-ball by adjusting the angle
                # Simple strategy: slightly deviate from the ideal angle
                phi_adjustment = 2.0 if eight_dist < 0.1 else 1.0
                phi = (phi + phi_adjustment) % 360
                
                # Reduce speed to mitigate accidental collisions
                velocity = min(velocity, 4.0)
                warning_msg += " (avoiding the 8-ball)"
            
            # Handling for high cue-ball scratch risk
            if cue_pocket_risk > 0.5:
                # Reduce speed significantly
                velocity = min(velocity, self.risky_velocity_limit)
                # Reduce elevation to avoid follow-through
                theta = min(theta, 15.0)
                # Apply backspin (negative b) to help the cue ball stop
                b_offset = -0.15
                warning_msg += f" (cue-ball risk={cue_pocket_risk:.2f})"
            
            # Step 7: Validate predicted first-contact ball
            predicted_first, _ = self.predict_first_contact_ball(cue_pos, phi, balls)
            
            if predicted_first and predicted_first != best_target_id:
                # Predicted first contact is not the target ball; attempt small angle adjustments
                if predicted_first in my_targets:
                    # First contact is another of our balls: acceptable but not ideal
                    warning_msg += f" (predicted first contact={predicted_first})"
                else:
                    # First contact is an opponent ball or the 8-ball: must adjust
                    # Try small angle adjustments
                    for angle_offset in [1.5, -1.5, 3.0, -3.0, 5.0, -5.0]:
                        test_phi = (phi + angle_offset) % 360
                        test_first, _ = self.predict_first_contact_ball(cue_pos, test_phi, balls)
                        if test_first == best_target_id:
                            phi = test_phi
                            warning_msg += f" (angle adjustment {angle_offset:+.1f}°)"
                            break
                    else:
                        # Could not fix; keep the original angle but reduce speed
                        velocity = min(velocity, 2.5)
                        warning_msg += f" (⚠️ first-contact risk={predicted_first})"
            
            # Final bounds check
            velocity = np.clip(velocity, 0.5, 8.0)
            phi = phi % 360
            theta = np.clip(theta, 0, 90)
            a_offset = np.clip(a_offset, -0.5, 0.5)
            b_offset = np.clip(b_offset, -0.5, 0.5)
            
            # Build action
            action = {
                'V0': float(velocity),
                'phi': float(phi),
                'theta': float(theta),
                'a': float(a_offset),
                'b': float(b_offset)
            }
            
            print(f"[GeometricAgent] Target: {best_target_id}→{best_pocket_id}, "
                  f"distance={distance:.3f}m, cut angle={cut_angle:.1f}°{warning_msg}")
            print(f"[GeometricAgent] Decision: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[GeometricAgent] Error during decision: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class MCTSAgent(Agent):
    """Agent based on Monte Carlo Tree Search (MCTS)
    
    Core idea:
    - Use geometry to generate candidate actions (target ball + pocket combinations)
    - Run multiple Monte Carlo simulations for each candidate action
    - Use the UCB formula to balance exploration and exploitation
    - Choose the action with the highest average reward
    
    Characteristics:
    - Uses geometric priors to narrow the search space
    - Evaluates action quality via physics simulation
    - Emphasizes sampling diversity more than pure Bayesian optimization
    """
    
    def __init__(self):
        super().__init__()
        
        # Import geometry utilities
        from utils.utils import (
            select_best_target,
            calculate_aim_point_for_pocket,
            calculate_angle_to_aim_point,
            calculate_distance,
            calculate_recommended_velocity,
            check_eight_ball_in_path,
            check_cue_ball_pocket_risk,
            predict_first_contact_ball,
            check_eight_ball_scratch_risk,
            calculate_shot_difficulty
        )
        self.select_best_target = select_best_target
        self.calculate_aim_point_for_pocket = calculate_aim_point_for_pocket
        self.calculate_angle_to_aim_point = calculate_angle_to_aim_point
        self.calculate_distance = calculate_distance
        self.calculate_recommended_velocity = calculate_recommended_velocity
        self.check_eight_ball_in_path = check_eight_ball_in_path
        self.check_cue_ball_pocket_risk = check_cue_ball_pocket_risk
        self.predict_first_contact_ball = predict_first_contact_ball
        self.check_eight_ball_scratch_risk = check_eight_ball_scratch_risk
        self.calculate_shot_difficulty = calculate_shot_difficulty
        
        # Core MCTS parameters
        self.num_simulations = 30          # Total simulations
        self.exploration_weight = 1.41     # UCB exploration weight (sqrt(2))
        self.num_candidates_per_target = 5 # Candidate actions per target
        
        # Action-parameter perturbation ranges
        self.phi_noise_range = 5.0         # Angle noise range ±5°
        self.V0_noise_range = 1.0          # Speed noise range ±1.0 m/s
        self.theta_noise_range = 10.0      # Elevation noise range ±10°
        self.offset_noise_range = 0.15     # Offset noise range ±0.15
        
        # Conservative parameters for the 8-ball
        self.eight_ball_V0_max = 4.5
        self.eight_ball_theta_max = 20
        
        print("MCTSAgent (Monte Carlo Tree Search) initialized.")
    
    def print_config(self):
        """Print all important hyperparameters"""
        print("\n" + "="*50)
        print("[MCTSAgent Hyperparameter Configuration]")
        print("  Core MCTS parameters:")
        print(f"    - num_simulations (simulations): {self.num_simulations}")
        print(f"    - exploration_weight (UCB exploration weight): {self.exploration_weight}")
        print(f"    - num_candidates_per_target (candidates per target): {self.num_candidates_per_target}")
        print("  Action perturbation ranges:")
        print(f"    - phi_noise_range: ±{self.phi_noise_range}°")
        print(f"    - V0_noise_range: ±{self.V0_noise_range} m/s")
        print(f"    - theta_noise_range: ±{self.theta_noise_range}°")
        print(f"    - offset_noise_range: ±{self.offset_noise_range}")
        print("  8-ball special parameters:")
        print(f"    - eight_ball_V0_max: {self.eight_ball_V0_max} m/s")
        print(f"    - eight_ball_theta_max: {self.eight_ball_theta_max}°")
        print("  Notes: geometric priors + MCTS search + physics-simulation evaluation")
        print("="*50)
    
    def _generate_candidate_actions(self, balls, my_targets, table, is_shooting_eight=False):
        """
        Generate candidate actions using geometry
        
        Approach:
        1. Iterate over all unpocketed target balls
        2. For each target ball, try each pocket
        3. Compute geometric aiming parameters
        4. Apply random perturbations to base parameters to generate variants
        
        Returns:
            list of dict: Candidate action list
        """
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:  # Already pocketed
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # Compute difficulty
                difficulty = self.calculate_shot_difficulty(
                    cue_pos, target_pos, pocket_pos, balls,
                    target_id=target_id, my_targets=my_targets
                )
                
                # Check whether the 8-ball is on/near the path
                eight_in_path, _ = self.check_eight_ball_in_path(cue_pos, target_pos, balls)
                if eight_in_path and not is_shooting_eight:
                    difficulty *= 10.0
                
                # Compute geometric aiming parameters
                aim_point = self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)
                if aim_point is None:
                    continue
                
                base_phi = self.calculate_angle_to_aim_point(cue_pos, aim_point)
                distance = self.calculate_distance(cue_pos, target_pos)
                base_V0 = self.calculate_recommended_velocity(distance)
                
                if is_shooting_eight:
                    base_V0 = min(base_V0, self.eight_ball_V0_max)
                
                # Generate multiple perturbed variants
                for _ in range(self.num_candidates_per_target):
                    phi = base_phi + np.random.uniform(-self.phi_noise_range, self.phi_noise_range)
                    V0 = base_V0 + np.random.uniform(-self.V0_noise_range, self.V0_noise_range)
                    theta = np.random.uniform(0, self.theta_noise_range if not is_shooting_eight else self.eight_ball_theta_max)
                    a = np.random.uniform(-self.offset_noise_range, self.offset_noise_range)
                    b = np.random.uniform(-self.offset_noise_range, self.offset_noise_range)
                    
                    V0 = np.clip(V0, 0.5, 8.0 if not is_shooting_eight else self.eight_ball_V0_max)
                    phi = phi % 360
                    theta = np.clip(theta, 0, 90)
                    a = np.clip(a, -0.5, 0.5)
                    b = np.clip(b, -0.5, 0.5)
                    
                    candidates.append({
                        'action': {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b},
                        'target_id': target_id,
                        'pocket_id': pocket_id,
                        'difficulty': difficulty
                    })
        
        candidates.sort(key=lambda x: x['difficulty'])
        max_candidates = self.num_simulations * 2
        return candidates[:max_candidates]
    
    def _simulate_action(self, action, balls, table, my_targets, last_state_snapshot):
        """Simulate an action and return the reward"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            shot.cue.set_state(
                V0=action['V0'],
                phi=action['phi'] % 360,
                theta=action['theta'],
                a=action['a'],
                b=action['b']
            )
            if not simulate_with_timeout(shot, timeout=3):
                return 0
        except Exception as e:
            return -500
        
        score = analyze_shot_for_reward(
            shot=shot,
            last_state=last_state_snapshot,
            player_targets=my_targets
        )
        return score
    
    def _ucb_score(self, avg_reward, visit_count, total_visits):
        """Compute UCB score: avg_reward + c * sqrt(ln(N) / n)"""
        if visit_count == 0:
            return float('inf')
        exploitation = avg_reward
        exploration = self.exploration_weight * math.sqrt(math.log(total_visits + 1) / visit_count)
        return exploitation + exploration
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Main MCTS decision flow"""
        if balls is None or my_targets is None or table is None:
            print("[MCTSAgent] Missing required inputs; using a random action.")
            return self._random_action()
        
        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_eight = False
            if len(remaining_own) == 0:
                my_targets = ["8"]
                is_shooting_eight = True
                print("[MCTSAgent] All target balls cleared; switching to the 8-ball. ⚠️ Entering cautious mode!")
            
            # Step 1: Generate candidate actions
            candidates = self._generate_candidate_actions(balls, my_targets, table, is_shooting_eight)
            
            if len(candidates) == 0:
                print("[MCTSAgent] No candidate actions; using a random action.")
                return self._random_action()
            
            print(f"[MCTSAgent] Generated {len(candidates)} candidate actions; starting MCTS search...")
            
            # Initialize statistics
            visit_counts = [0] * len(candidates)
            total_rewards = [0.0] * len(candidates)
            
            # Step 2: MCTS iterations
            for sim_idx in range(self.num_simulations):
                total_visits = sum(visit_counts)
                ucb_scores = [
                    self._ucb_score(
                        total_rewards[i] / max(1, visit_counts[i]),
                        visit_counts[i],
                        total_visits
                    )
                    for i in range(len(candidates))
                ]
                
                selected_idx = np.argmax(ucb_scores)
                selected_candidate = candidates[selected_idx]
                
                reward = self._simulate_action(
                    selected_candidate['action'],
                    balls, table, my_targets,
                    last_state_snapshot
                )
                
                visit_counts[selected_idx] += 1
                total_rewards[selected_idx] += reward
            
            # Step 3: Select the best action
            avg_rewards = [
                total_rewards[i] / max(1, visit_counts[i])
                for i in range(len(candidates))
            ]
            
            best_idx = np.argmax(avg_rewards)
            best_candidate = candidates[best_idx]
            best_action = best_candidate['action']
            best_avg_reward = avg_rewards[best_idx]
            best_visits = visit_counts[best_idx]
            
            if best_avg_reward < 10:
                print(f"[MCTSAgent] MCTS best reward is low ({best_avg_reward:.2f}); falling back to the easiest target.")
                best_candidate = candidates[0]
                best_action = best_candidate['action']
            
            print(f"[MCTSAgent] Target: {best_candidate['target_id']}→{best_candidate['pocket_id']}, "
                  f"avg reward={best_avg_reward:.2f}, visits={best_visits}")
            print(f"[MCTSAgent] Decision: V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, "
                  f"θ={best_action['theta']:.2f}, a={best_action['a']:.3f}, b={best_action['b']:.3f}")
            
            return {
                'V0': float(best_action['V0']),
                'phi': float(best_action['phi']) % 360,
                'theta': float(best_action['theta']),
                'a': float(best_action['a']),
                'b': float(best_action['b'])
            }
            
        except Exception as e:
            print(f"[MCTSAgent] Error during decision: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class EnsembleVotingAgent(Agent):
    """Ensemble voting agent
    
    Core idea:
    - Call multiple sub-agents (NewAgent, MCTSAgent) to generate candidate actions
    - Evaluate each candidate via physics simulation
    - Select the highest-scoring action as the final decision
    
    Advantages:
    - Combines strengths of multiple decision strategies
    - Reduces mistakes via simulation-based validation
    - Produces more robust decisions
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize sub-agents
        self.new_agent = Enhanced_Bayes_Agent()
        self.mcts_agent = MCTSAgent()
        
        # Evaluation parameters
        self.num_eval_simulations = 3  # Simulations per action evaluation
        
        print("EnsembleVotingAgent (Ensemble Voting) initialized.")
    
    def print_config(self):
        """Print all important hyperparameters"""
        print("\n" + "="*50)
        print("[EnsembleVotingAgent Hyperparameter Configuration]")
        print("  Sub-agents:")
        print("    - NewAgent (geometry + Bayesian optimization)")
        print("    - MCTSAgent (geometry + MCTS search)")
        print("  Evaluation parameters:")
        print(f"    - num_eval_simulations (sims per action): {self.num_eval_simulations}")
        print("  Notes: multi-strategy ensemble voting, selects the highest-scoring action")
        print("="*50)
        print("\n--- NewAgent config ---")
        self.new_agent.print_config()
        print("\n--- MCTSAgent config ---")
        self.mcts_agent.print_config()
    
    def _evaluate_action(self, action, balls, table, my_targets, last_state_snapshot):
        """Evaluate a single action's score
        
        Uses multiple simulations and averages the result to reduce randomness.
        """
        total_score = 0.0
        valid_sims = 0
        
        for _ in range(self.num_eval_simulations):
            sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = pt.Cue(cue_ball_id="cue")
            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
            
            try:
                shot.cue.set_state(
                    V0=action['V0'],
                    phi=action['phi'] % 360,
                    theta=action['theta'],
                    a=action['a'],
                    b=action['b']
                )
                
                if not simulate_with_timeout(shot, timeout=3):
                    continue
                
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                total_score += score
                valid_sims += 1
                
            except Exception as e:
                continue
        
        if valid_sims == 0:
            return -500.0
        
        return total_score / valid_sims
    
    def decision(self, balls=None, my_targets=None, table=None):
        """Ensemble voting decision
        
        Steps:
        1. Call NewAgent to get a candidate action
        2. Call MCTSAgent to get a candidate action
        3. Evaluate each candidate via simulation
        4. Choose the highest-scoring action
        """
        if balls is None or my_targets is None or table is None:
            print("[EnsembleVotingAgent] Missing required inputs; using a random action.")
            return self._random_action()
        
        try:
            # Save state snapshot
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # Check whether all target balls are already pocketed
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[EnsembleVotingAgent] All target balls cleared; switching to the 8-ball.")
            
            # Step 1: Collect candidate actions
            candidate_actions = []
            
            # Get an action from NewAgent
            print("[EnsembleVotingAgent] Calling NewAgent...")
            try:
                new_agent_action = self.new_agent.decision(balls, my_targets, table)
                if new_agent_action:
                    candidate_actions.append({
                        'action': new_agent_action,
                        'source': 'NewAgent'
                    })
            except Exception as e:
                print(f"[EnsembleVotingAgent] NewAgent decision failed: {e}")
            
            # Get an action from MCTSAgent
            print("[EnsembleVotingAgent] Calling MCTSAgent...")
            try:
                mcts_agent_action = self.mcts_agent.decision(balls, my_targets, table)
                if mcts_agent_action:
                    candidate_actions.append({
                        'action': mcts_agent_action,
                        'source': 'MCTSAgent'
                    })
            except Exception as e:
                print(f"[EnsembleVotingAgent] MCTSAgent decision failed: {e}")
            
            if len(candidate_actions) == 0:
                print("[EnsembleVotingAgent] No candidate actions; using a random action.")
                return self._random_action()
            
            # Step 2: Evaluate each candidate action
            print(f"[EnsembleVotingAgent] Evaluating {len(candidate_actions)} candidate actions...")
            
            best_action = None
            best_score = float('-inf')
            best_source = None
            
            for candidate in candidate_actions:
                action = candidate['action']
                source = candidate['source']
                
                score = self._evaluate_action(
                    action, balls, table, my_targets, last_state_snapshot
                )
                
                print(f"[EnsembleVotingAgent] {source} action score: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_source = source
            
            # Step 3: Return the best action
            print(f"[EnsembleVotingAgent] Final selection: {best_source} (score: {best_score:.2f})")
            print(f"[EnsembleVotingAgent] Decision: V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, "
                  f"θ={best_action['theta']:.2f}, a={best_action['a']:.3f}, b={best_action['b']:.3f}")
            
            return best_action
            
        except Exception as e:
            print(f"[EnsembleVotingAgent] Error during decision: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()
