import pooltool as pt
import numpy as np
import copy

from agent.BasicAgent.basic_agent import BasicAgent

def collect_ball_states(shot):
    """
    Collect ball state information.

    Args:
        shot: System object

    Returns:
        dict: {ball_id: {'position','velocity','spin','state','time','pocketed'}}
    """
    results = {}
    for ball_id, ball in shot.balls.items():
        s = ball.state
        results[ball_id] = {
            "position": s.rvw[0].tolist(),
            "velocity": s.rvw[1].tolist(),
            "spin": s.rvw[2].tolist(),
            "state": int(s.s),
            "time": float(s.t),
            "pocketed": ball.state.s
        }
    return results

def save_balls_state(balls):
    """Save ball states (deep copy).

    Args:
        balls: {ball_id: Ball}

    Returns:
        dict: deep-copied ball states
    """
    return {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

def restore_balls_state(saved_state):
    """Restore ball states (deep copy).

    Args:
        saved_state: previously saved ball states

    Returns:
        dict: restored deep-copied ball states
    """
    return {bid: copy.deepcopy(ball) for bid, ball in saved_state.items()}

class PoolEnv():
    """Pool match environment.

    Usage:
        env = PoolEnv()
        env.reset(target_ball='solid')  # or 'stripe'
        obs = env.get_observation()
        res = env.take_shot(action)
    """

    def __init__(self):
        # Initialize environment (call reset() before use)
        self.table = None
        self.balls = None
        self.cue = None
        # Player A and B target ball IDs
        self.player_targets = None
        # hit count
        self.hit_count = 0
        # last state snapshot (for foul rollback)
        self.last_state = None
        # player names
        self.players = ["A", "B"]
        # current player index (0 = A, 1 = B)
        self.curr_player = 0
        # finished flag
        self.done = False
        # winner: 'A', 'B', or 'SAME'
        self.winner = None
        # maximum allowed hits before forced end
        self.MAX_HIT_COUNT = 60
        # record all shots for post-game rendering or saving
        self.shot_record = pt.MultiSystem()
        # shot parameter noise std (simulate real-world error)
        self.noise_std = {
            'V0': 0.1,      # velocity std
            'phi': 0.1,     # horizontal angle std (degrees)
            'theta': 0.1,   # vertical angle std (degrees)
            'a': 0.003,     # lateral offset std (relative to ball radius)
            'b': 0.003      # longitudinal offset std (relative to ball radius)
        }
        self.enable_noise = True  # whether to apply noise

    def get_observation(self, player=None):
        """
        Get observation for the specified player (deep copy).

        Args:
            player (str, optional): 'A' or 'B'. If None, returns current player.

        Returns:
            tuple: (balls, my_targets, table)

                balls (dict): {ball_id: Ball}
                    ball_id values:
                        - 'cue': cue ball
                        - '1'-'7': solid balls
                        - '8': 8-ball
                        - '9'-'15': stripe balls

                    Ball.state.rvw: np.ndarray shape=(3,3)
                        [0]: position np.array([x,y,z]) in meters
                        [1]: velocity np.array([vx,vy,vz]) in m/s
                        [2]: spin np.array([wx,wy,wz]) in rad/s

                    Ball.state.s: int status code
                        0 = stationary
                        4 = pocketed
                        1-3 = moving

                    Ball.state.t: float timestamp (seconds)

                my_targets (list[str]): player's target ball IDs
                    - Normal: ['1','2',...] or ['9','10',...]
                    - After clearing group: ['8'] (must pocket 8)

                table (Table): table object with attributes like width/length and pockets
        """
        # If player not given, use current player
        if player is None:
            player = self.get_curr_player()
        # Return deep copies of balls and table, and my target IDs
        return copy.deepcopy(self.balls), self.player_targets[player], copy.deepcopy(self.table)
        
    def get_curr_player(self,):
        """Get current player; returns 'A' or 'B'."""
        return self.players[self.curr_player]
    
    def get_done(self,):
        """Check whether the game is finished.

        Returns:
            (True, {'winner': 'A'/'B'/'SAME', 'hit_count': int}) if finished
            (False, {}) if not finished
        """
        if self.done:
            return True, {'winner':self.winner, 'hit_count':self.hit_count}
        return False, {}
    
    def reset(self, state=None, target_ball:str=None):
        """
        Reset environment.

        Args:
            state: reserved, must be None
            target_ball: Player A target type
                'solid': A plays solids (1-7), B plays stripes (9-15)
                'stripe': A plays stripes (9-15), B plays solids (1-7)
        """
        # restoring to a specific state is not supported yet
        if state is not None:
            raise NotImplementedError("Restoring to a specific state is not supported yet.")
        
        # set up table and rack
        self.table = pt.Table.default()
        self.balls = pt.get_rack(pt.GameType.EIGHTBALL, self.table)
        self.cue = pt.Cue(cue_ball_id="cue") 
        
        # set player target ball lists
        if target_ball == 'solid':
            self.player_targets = {
                "A": [str(i) for i in range(1, 8)],
                "B": [str(i) for i in range(9, 16)],
            }
        elif target_ball == 'stripe':
            self.player_targets = {
                "A": [str(i) for i in range(9, 16)],
                "B": [str(i) for i in range(1, 8)],
            }
        else:
            raise NotImplementedError("unsupported target_ball parameter", target_ball)
        
        # reset counters and records
        self.hit_count = 0
        self.last_state = save_balls_state(self.balls)
        self.curr_player = 0
        self.done = False
        self.winner = None
        self.shot_record = pt.MultiSystem()
        
    def take_shot(self, action:dict):
        """
        Execute a shot action.

        Args:
            action: {'V0': [0.5,8.0], 'phi': [0,360], 'theta': [0,90], 'a': [-0.5,0.5], 'b': [-0.5,0.5]}

        Returns:
            dict (see docstring in get_observation for field descriptions)

        Note: when `enable_noise=True` Gaussian noise is added to the action
        """
        # apply Gaussian noise to simulate real-world error
        if self.enable_noise:
            noisy_action = {
                'V0': action['V0'] + np.random.normal(0, self.noise_std['V0']),
                'phi': action['phi'] + np.random.normal(0, self.noise_std['phi']),
                'theta': action['theta'] + np.random.normal(0, self.noise_std['theta']),
                'a': action['a'] + np.random.normal(0, self.noise_std['a']),
                'b': action['b'] + np.random.normal(0, self.noise_std['b'])
            }
            
            # clip parameters to valid ranges
            noisy_action['V0'] = np.clip(noisy_action['V0'], 0.5, 8.0)
            noisy_action['phi'] = noisy_action['phi'] % 360  # wrap angle
            noisy_action['theta'] = np.clip(noisy_action['theta'], 0, 90)
            noisy_action['a'] = np.clip(noisy_action['a'], -0.5, 0.5)
            noisy_action['b'] = np.clip(noisy_action['b'], -0.5, 0.5)
            
            # Print original and noisy action
            print(f"Player {self.get_curr_player()} original action: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"theta={action['theta']:.2f}°, a={action['a']:.3f}, b={action['b']:.3f}")
            print(f"Player {self.get_curr_player()} actual action: V0={noisy_action['V0']:.2f}, phi={noisy_action['phi']:.2f}, "
                  f"theta={noisy_action['theta']:.2f}°, a={noisy_action['a']:.3f}, b={noisy_action['b']:.3f}")
            
            action = noisy_action
        else:
            # when noise disabled, print the action to be executed
            print(f"Player {self.get_curr_player()} executing action: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"theta={action['theta']:.2f}°, a={action['a']:.3f}, b={action['b']:.3f}")

        # run physics simulation for the shot
        shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)
        self.cue.set_state(V0=action["V0"], phi=action["phi"], theta=action["theta"], a=action['a'], b=action['b'])
        pt.simulate(shot, inplace=True)
        # record shot (for rendering later)
        self.shot_record.append(copy.deepcopy(shot))
        # update balls to the simulation result
        self.balls = shot.balls 
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and self.last_state[bid].state.s != 4]

        events = shot.events
        first_contact_ball_id = None
        # define valid ball ID set
        valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                # filter out 'cue' and non-ball ids
                other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break
        cue_hit_cushion = False
        target_hit_cushion = False
        for e in events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if 'cushion' in et:
                if 'cue' in ids:
                    cue_hit_cushion = True
                if first_contact_ball_id is not None and first_contact_ball_id in ids:
                    target_hit_cushion = True
        
        # categorize pocketed balls
        own_pocketed = [bid for bid in new_pocketed if bid in self.player_targets[self.players[self.curr_player]]]
        enemy_pocketed = [bid for bid in new_pocketed if bid not in self.player_targets[self.players[self.curr_player]] and bid not in ["cue", "8"]]
        
        ##### Rules: check fouls, determine next player, and game end #####

        # cue and 8-ball pocketed together -> immediate loss for current player
        if "cue" in new_pocketed and "8" in new_pocketed:
            print("Cue and 8-ball pocketed together: foul, immediate loss.")
            print(f"Player {self.players[1 - self.curr_player]} wins!")
            self.done = True
            self.winner = self.players[1 - self.curr_player]
            return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': True, 'BLACK_BALL_INTO_POCKET': True, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': False, 'BALLS': copy.deepcopy(self.balls)}

        # cue ball pocketed (foul)
        if "cue" in new_pocketed:
            print("Cue ball pocketed! Foul: revert to previous state and switch turn.")
            # restore pre-shot state
            balls_before_shot = copy.deepcopy(self.last_state)
            self.balls = restore_balls_state(self.last_state)
            self.curr_player = 1 - self.curr_player
            self.done = False
            self.hit_count += 1
            if self.hit_count >= self.MAX_HIT_COUNT:
                print("Reached max hit count, game over.")
                self.done = True
                a_left = len([bid for bid in self.player_targets["A"] if bid != '8' and self.balls[bid].state.s != 4])
                b_left = len([bid for bid in self.player_targets["B"] if bid != '8' and self.balls[bid].state.s != 4])
                if a_left < b_left:
                    self.winner = "A"
                elif b_left < a_left:
                    self.winner = "B"
                else:
                    self.winner = "SAME"
                print(f"Max hit count summary: A remaining {a_left}, B remaining {b_left}, winner: {self.winner}")
            return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': True, 'BLACK_BALL_INTO_POCKET': False, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': False, 'BALLS': balls_before_shot}
        
        player = self.get_curr_player()
        remaining_own_before = [bid for bid in self.player_targets[player] if self.last_state[bid].state.s != 4]
        # 8-ball pocketed: win/loss determination
        if "8" in new_pocketed:
            # check whether player had cleared their group before pocketing 8
            if len(remaining_own_before) == 0:
                print(f"Player {player} pocketed the 8-ball legally and wins!")
                self.winner = self.players[self.curr_player]
            else:
                print(f"Player {player} illegally pocketed the 8-ball before clearing group: immediate loss!")
                print(f"Player {self.players[1 - self.curr_player]} wins!")
                self.winner = self.players[1 - self.curr_player]
            self.done = True
            return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': True, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': False, 'BALLS': copy.deepcopy(self.balls)}

        if first_contact_ball_id is None:
            print(f"⚠️ Miss: cue ball did not contact any ball. Revert to previous state and switch turn.")
            # restore pre-shot state
            balls_before_shot = copy.deepcopy(self.last_state)
            self.balls = restore_balls_state(self.last_state)
            self.curr_player = 1 - self.curr_player
            self.hit_count += 1
            if self.hit_count >= self.MAX_HIT_COUNT:
                print("Reached max hit count, game over.")
                self.done = True
                a_left = len([bid for bid in self.player_targets["A"] if bid != '8' and self.balls[bid].state.s != 4])
                b_left = len([bid for bid in self.player_targets["B"] if bid != '8' and self.balls[bid].state.s != 4])
                if a_left < b_left:
                    self.winner = "A"
                elif b_left < a_left:
                    self.winner = "B"
                else:
                    self.winner = "SAME"
                print(f"Max hit count summary: Player A remaining {a_left}, Player B remaining {b_left}, winner: {self.winner}")
            return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': False, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': False, 'NO_HIT': True, 'BALLS': balls_before_shot}
        if first_contact_ball_id is not None:
            opponent_plus_eight = [bid for bid in self.balls.keys() if bid not in self.player_targets[player] and bid not in ['cue']]
            if ('8' not in opponent_plus_eight):
                opponent_plus_eight.append('8')
            # when player still has group balls left, first contact with opponent's ball or 8 is a foul
            # when only 8 remains, first contact must be 8, otherwise foul
            if (len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight) or \
               (len(remaining_own_before) == 0 and first_contact_ball_id != '8'):
                if len(remaining_own_before) == 0:
                    print(f"⚠️ Player {player} collided first with non-8 when only 8 remains: foul. Revert and switch turn.")
                else:
                    print(f"⚠️ Player {player} first contact was opponent's ball or 8: foul. Revert and switch turn.")
                # restore pre-shot state
                balls_before_shot = copy.deepcopy(self.last_state)
                self.balls = restore_balls_state(self.last_state)
                self.curr_player = 1 - self.curr_player
                self.hit_count += 1
                if self.hit_count >= self.MAX_HIT_COUNT:
                    print("Reached max hit count, game over.")
                    self.done = True
                    a_left = len([bid for bid in self.player_targets["A"] if bid != '8' and self.balls[bid].state.s != 4])
                    b_left = len([bid for bid in self.player_targets["B"] if bid != '8' and self.balls[bid].state.s != 4])
                    if a_left < b_left:
                        self.winner = "A"
                    elif b_left < a_left:
                        self.winner = "B"
                    else:
                        self.winner = "SAME"
                    print(f"Max hit count summary: A remaining {a_left}, B remaining {b_left}, winner: {self.winner}")
                return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': False, 'FOUL_FIRST_HIT': True, 'NO_POCKET_NO_RAIL': False, 'BALLS': copy.deepcopy(self.balls)}

        # handle case: no pocketed balls
        if len(new_pocketed) == 0:
            if (not cue_hit_cushion) and (not target_hit_cushion):
                # No pocket and no cushion contact -> foul
                print(f"⚠️ No pocket and no cushion contact: foul. Revert to previous state and switch turn.")
                # restore pre-shot state
                balls_before_shot = copy.deepcopy(self.last_state)
                self.balls = restore_balls_state(self.last_state)
                self.curr_player = 1 - self.curr_player
                self.hit_count += 1
                if self.hit_count >= self.MAX_HIT_COUNT:
                    print("Reached max hit count, game over.")
                    self.done = True
                    a_left = len([bid for bid in self.player_targets["A"] if bid != '8' and self.balls[bid].state.s != 4])
                    b_left = len([bid for bid in self.player_targets["B"] if bid != '8' and self.balls[bid].state.s != 4])
                    if a_left < b_left:
                        self.winner = "A"
                    elif b_left < a_left:
                        self.winner = "B"
                    else:
                        self.winner = "SAME"
                    print(f"Max hit count summary: A remaining {a_left}, B remaining {b_left}, winner: {self.winner}")
                return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': False, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': True, 'BALLS': balls_before_shot}
            else:
                # No pocket but cushion contact -> switch turn
                print(f"⚠️ No pocket, switch turn.")
                self.curr_player = 1 - self.curr_player
                self.last_state = save_balls_state(self.balls)
                self.hit_count += 1
                if self.hit_count >= self.MAX_HIT_COUNT:
                    print("Reached max hit count, game over.")
                    self.done = True
                    a_left = len([bid for bid in self.player_targets["A"] if bid != '8' and self.balls[bid].state.s != 4])
                    b_left = len([bid for bid in self.player_targets["B"] if bid != '8' and self.balls[bid].state.s != 4])
                    if a_left < b_left:
                        self.winner = "A"
                    elif b_left < a_left:
                        self.winner = "B"
                    else:
                        self.winner = "SAME"
                    print(f"Max hit count summary: A remaining {a_left}, B remaining {b_left}, winner: {self.winner}")
                return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': False, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': False, 'BALLS': copy.deepcopy(self.balls)}
        
        # decide next player based on whether own balls were pocketed
        if own_pocketed:
            print(f"Player {player} pocketed {own_pocketed}, continue turn.")
        else:
            print(f"Player {player} did not pocket own ball, switch turn.")
            self.curr_player = 1 - self.curr_player

        # save current state and increment hit count
        self.last_state = save_balls_state(self.balls)

        self.hit_count += 1
        if self.hit_count >= self.MAX_HIT_COUNT:
            print("Reached max hit count, game over.")
            self.done = True
            a_left = len([bid for bid in self.player_targets["A"] if bid != '8' and self.balls[bid].state.s != 4])
            b_left = len([bid for bid in self.player_targets["B"] if bid != '8' and self.balls[bid].state.s != 4])
            if a_left < b_left:
                self.winner = "A"
            elif b_left < a_left:
                self.winner = "B"
            else:
                self.winner = "SAME"
            print(f"Max hit count summary: A remaining {a_left}, B remaining {b_left}, winner: {self.winner}")
            return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': False, 'BALLS': copy.deepcopy(self.balls)}
        
        # return shot result
        return {'ME_INTO_POCKET': own_pocketed, 'ENEMY_INTO_POCKET': enemy_pocketed, 'WHITE_BALL_INTO_POCKET': False, 'BLACK_BALL_INTO_POCKET': False, 'FOUL_FIRST_HIT': False, 'NO_POCKET_NO_RAIL': False, 'BALLS': copy.deepcopy(self.balls)}
    

if __name__ == '__main__':
    # Initialize environment for manual run
    env = PoolEnv()

    agent_a, agent_b = BasicAgent(), BasicAgent()

    env.reset(target_ball='solid') # choose Player A's target type
    while True:
        player = env.get_curr_player()
        print(f"[Hit {env.hit_count}] Player: {player}")
        balls, my_targets, table = env.get_observation(player)
        if player == 'A': # alternate first/second
            action = agent_a.decision(balls, my_targets, table)
        else:
            action = agent_b.decision(balls, my_targets, table)
        env.take_shot(action)
        
        # view current shot, press ESC to exit
        # pt.show(env.shot_record[-1], title=f"hit count: {env.hit_count}")
        
        done, info = env.get_done()
        if done:
            print("Game over.")
            ## view full shot record, press ESC to step through shots
            # for i in range(len(env.shot_record)):
            #     pt.show(env.shot_record[i], title=f"hit count: {i}")
            
            ## view entire match with p/n to step
            # pt.show(env.shot_record, title=f"all record")
            break