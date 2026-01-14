import pooltool as pt
import numpy as np
import copy
import signal

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

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


# ============ Reward and BasicAgent ============
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
    valid_ball_ids = {str(i) for i in range(1, 16)} | {"8"}

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    # cushion analysis
    cue_hit_cushion = False
    target_hit_cushion = False
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    foul_no_rail = (len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion))

    # scoring
    score = 0
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        if player_targets == ['8']:
            score += 100
        else:
            score -= 150

    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10

    return score


class BasicAgent(Agent):
    """Basic agent using Bayesian optimization to search shot parameters."""
    def __init__(self):
        super().__init__()
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90),
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        print("BasicAgent initialized.")

    def _create_optimizer(self, reward_function, seed):
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )

        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )

        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        return optimizer

    def decision(self, balls=None, my_targets=None, table=None):
        """Search for best shot parameters using Bayesian optimization.

        Returns random action on failure.
        """
        if balls is None:
            print(f"[BasicAgent] decision did not receive balls info, using random action.")
            return self._random_action()

        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] No group balls left; switching target to 8-ball")

            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])

                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)

                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)

                    if not simulate_with_timeout(shot, timeout=3):
                        return 0
                except Exception as e:
                    print(f"[BasicAgent] Simulation failed, applying heavy penalty. Reason: {e}")
                    return -500

                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                return score

            print(f"[BasicAgent] Searching best shot for player (targets: {my_targets})...")

            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )

            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] No good plan found (best: {best_score:.2f}). Using random action.")
                return self._random_action()

            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] Decision (score: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"theta={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] fatal error during decision, using random action. Reason: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()