import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import math
import numpy as np
try:
    import pooltool as pt
except Exception:
    pt = None

class FeedbackManager:
    def __init__(self):
        pass

    def give_feedback(self, action: dict, balls: dict, my_targets: list, table) -> tuple:
        # deep-copy environment for simulation
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        # apply action
        V0 = float(action['V0'])
        phi = float(action['phi'])
        theta = float(action['theta'])
        a = float(action['a'])
        b = float(action['b'])
        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)

        # run simulation
        pt.simulate(shot, inplace=True)

        # analyze simulation results
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and balls[bid].state.s != 4]

        # find first contact ball
        first_contact_ball_id = None
        for e in shot.events:
            et = str(e.event_type).lower()
            ids = list(e.ids) if hasattr(e, 'ids') else []
            if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                other_ids = [i for i in ids if i != 'cue']
                if other_ids:
                    first_contact_ball_id = other_ids[0]
                    break

        messages = []

        # no-contact case
        if first_contact_ball_id is None and len(balls) > 2:
            messages.append("No contact: cue ball did not hit any object ball.")

        # first-contact foul checks
        remaining_own_before = [bid for bid in my_targets if balls[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in balls.keys() if bid not in my_targets and bid != 'cue']
        if '8' not in opponent_plus_eight:
            opponent_plus_eight.append('8')

        FOUL_FIRST_HIT = False
        if first_contact_ball_id is None:
            if len(balls) > 2:
                FOUL_FIRST_HIT = True
        else:
            if len(remaining_own_before) > 0 and first_contact_ball_id in opponent_plus_eight:
                FOUL_FIRST_HIT = True
            if len(remaining_own_before) == 0 and first_contact_ball_id != '8':
                FOUL_FIRST_HIT = True

        # no-rail foul checks
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

        NO_POCKET_NO_RAIL = (len(new_pocketed) == 0 and (not cue_hit_cushion) and (not target_hit_cushion))
        NO_HIT = (first_contact_ball_id is None)

        # pocketing summary
        ME_INTO_POCKET = [bid for bid in new_pocketed if bid in my_targets]
        ENEMY_INTO_POCKET = [bid for bid in new_pocketed if bid not in my_targets and bid not in ["cue", "8"]]
        WHITE_BALL_INTO_POCKET = ('cue' in new_pocketed)
        BLACK_BALL_INTO_POCKET = ('8' in new_pocketed)

        # immediate loss checks
        immediate_loss = False
        immediate_reasons = []
        if WHITE_BALL_INTO_POCKET and BLACK_BALL_INTO_POCKET:
            immediate_loss = True
            immediate_reasons.append('cue_and_8_same_shot')
        if BLACK_BALL_INTO_POCKET and len(remaining_own_before) > 0:
            immediate_loss = True
            immediate_reasons.append('8_before_clearing')

        # build feedback dictionary (flags)
        # flags = {
        #     'ME_INTO_POCKET': ME_INTO_POCKET,
        #     'ENEMY_INTO_POCKET': ENEMY_INTO_POCKET,
        #     'WHITE_BALL_INTO_POCKET': WHITE_BALL_INTO_POCKET,
        #     'BLACK_BALL_INTO_POCKET': BLACK_BALL_INTO_POCKET,
        #     'FOUL_FIRST_HIT': FOUL_FIRST_HIT,
        #     'NO_POCKET_NO_RAIL': NO_POCKET_NO_RAIL,
        #     'NO_HIT': NO_HIT,
        #     'IMMEDIATE_GAME_LOSS': immediate_loss,
        #     'IMMEDIATE_REASONS': immediate_reasons,
        # }

        # whether specified target was pocketed
        target = action.get('Target') if isinstance(action, dict) else None
        target_pocketed = False
        if target:
            target_pocketed = (target in ME_INTO_POCKET) or (target in ENEMY_INTO_POCKET) or (target == 'cue' and WHITE_BALL_INTO_POCKET) or (target == '8' and BLACK_BALL_INTO_POCKET)

        # immediate loss messages
        if immediate_loss:
            if 'cue_and_8_same_shot' in immediate_reasons:
                messages.append('Cue and 8-ball pocketed together: immediate game loss.')
            if '8_before_clearing' in immediate_reasons:
                messages.append('8-ball pocketed before clearing your group: immediate game loss.')

        # non-immediate fouls or warnings
        if WHITE_BALL_INTO_POCKET and not BLACK_BALL_INTO_POCKET:
            messages.append('Cue ball pocketed: foul.')
        if FOUL_FIRST_HIT:
            messages.append('Illegal first contact: foul.')
        if NO_HIT:
            messages.append('Cue ball did not contact any ball: foul')
        if NO_POCKET_NO_RAIL:
            messages.append('No pocket and no cushion contact: foul.')

        # messages for legal outcomes
        if not messages and not immediate_loss:
            if ME_INTO_POCKET:
                messages.append(f'Pocketed own balls: {ME_INTO_POCKET}. Continue turn.')
            elif ENEMY_INTO_POCKET:
                messages.append(f'Pocketed opponent balls: {ENEMY_INTO_POCKET}. Turn switches.')
            else:
                if (not NO_POCKET_NO_RAIL) and (len(new_pocketed) == 0):
                    messages.append('No pocket but cushion contact: legal, turn switches.')
                else:
                    messages.append('No balls pocketed.')

        # If a target was specified: consider execution valid if other own balls were pocketed
        if target:
            if target_pocketed:
                messages.append(f'Target {target} pocketed.')
            else:
                if ME_INTO_POCKET:
                    messages.append(f'Target {target} NOT pocketed, but other own balls {ME_INTO_POCKET} were pocketede.')
                else:
                    messages.append(f'Target {target} NOT pocketed.')

        target_ok = True
        if target:
            target_ok = target_pocketed or (len(ME_INTO_POCKET) > 0)

        ready = (not immediate_loss and not FOUL_FIRST_HIT and not NO_POCKET_NO_RAIL and not NO_HIT and not WHITE_BALL_INTO_POCKET and target_ok)

        # construct [Simulation Feedback]
        feedback_lines = []
        feedback_lines.append("[Simulation Feedback]")
        feedback_lines.append(f"- Previous Action: V0={V0}, phi={phi}, theta={theta}, a={a}, b={b}")
        # feedback_lines.append(f"- New pocketed: {new_pocketed}")
        # feedback_lines.append(f"- First contact: {first_contact_ball_id}")
        # feedback_lines.append(f"- Flags: {flags}")
        feedback_lines.append("- Messages:")
        for m in messages:
            feedback_lines.append(f"  - {m}")

        feedback_str = "\n".join(feedback_lines)
        
        return ready, feedback_str
    
if __name__ == "__main__":
    # Add simple local tests using a fake pooltool to avoid real dependency
    class FakeState:
        def __init__(self, s):
            self.s = s

    class FakeBall:
        def __init__(self, s=1):
            self.state = FakeState(s)

    class FakeEvent:
        def __init__(self, event_type, ids):
            self.event_type = event_type
            self.ids = ids

    class FakeCue:
        def __init__(self, cue_ball_id="cue"):
            self.cue_ball_id = cue_ball_id

        def set_state(self, V0=None, phi=None, theta=None, a=None, b=None):
            pass

    class FakeSystem:
        def __init__(self, table=None, balls=None, cue=None, events=None):
            self.cue = cue or FakeCue()
            self.balls = balls or {}
            self.events = events or []

    class FakePt:
        def __init__(self):
            self.Cue = FakeCue
            self.System = FakeSystem

        @staticmethod
        def simulate(shot, inplace=True):
            return None

    def make_balls_dict(pairs):
        return {bid: FakeBall(s) for bid, s in pairs.items()}

    def run_test(name, orig_balls, sim_balls, events, my_targets, action):
        print(f"--- {name} ---")
        fake_pt = FakePt()

        # System constructor: ignore passed-in balls, return FakeSystem using sim_balls and events
        def system_ctor(table=None, balls=None, cue=None):
            return FakeSystem(table=table, balls=sim_balls, cue=FakeCue(), events=events)

        fake_pt.System = system_ctor

        # replace module-level pt with fake_pt for testing
        globals()['pt'] = fake_pt

        fm = FeedbackManager()
        ready, fbstr = fm.give_feedback(action, orig_balls, my_targets, table=None)
        print(f"ready={ready}")
        print(fbstr)
        print()

    # Test1: own ball pocketed -> legal, continue
    orig = make_balls_dict({'1': 1, 'cue': 1})
    sim = make_balls_dict({'1': 4, 'cue': 1})
    ev = [FakeEvent('collision', ('cue', '1'))]
    act = {'V0': 1.0, 'phi': 0.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0}
    run_test('Own ball pocketed', orig, sim, ev, ['1'], act)

    # Test2: cue ball pocketed -> foul
    orig2 = make_balls_dict({'1': 1, 'cue': 1})
    sim2 = make_balls_dict({'1': 1, 'cue': 4})
    ev2 = [FakeEvent('collision', ('cue', '1'))]
    run_test('Cue ball pocketed (foul)', orig2, sim2, ev2, ['1'], act)

    # Test3: target not pocketed but other own ball pocketed -> acceptable
    orig3 = make_balls_dict({'1': 1, '2': 1, 'cue': 1})
    sim3 = make_balls_dict({'1': 4, '2': 1, 'cue': 1})
    ev3 = [FakeEvent('collision', ('cue', '1'))]
    act3 = {'V0': 1.0, 'phi': 0.0, 'theta': 0.0, 'a': 0.0, 'b': 0.0, 'Target': '2'}
    run_test('Target not pocketed but other own pocketed', orig3, sim3, ev3, ['1', '2'], act3)

    # Test4: cue and 8 pocketed together -> immediate loss
    orig4 = make_balls_dict({'1': 1, '8': 1, 'cue': 1})
    sim4 = make_balls_dict({'1': 1, '8': 4, 'cue': 4})
    ev4 = [FakeEvent('collision', ('cue', '8'))]
    run_test('Cue and 8 pocketed (immediate loss)', orig4, sim4, ev4, ['1'], act)