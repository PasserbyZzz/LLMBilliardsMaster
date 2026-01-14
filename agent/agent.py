import random

# ============ Agent ============
class Agent():
    # Base Agent class
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """Decision method.

        Returns:
            dict with keys 'V0', 'phi', 'theta', 'a', 'b'.
        """
        pass
    
    def _random_action(self,):
        """Generate a random shot action.

        Returns:
            dict with ranges:
                V0: [0.5, 8.0] m/s
                phi: [0, 360] degrees
                theta: [0, 90] degrees
                a, b: [-0.5, 0.5] (fractions of ball radius)
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # initial speed (0.5–8.0 m/s)
            'phi': round(random.uniform(0, 360), 2),    # horizontal angle (0°–360°)
            'theta': round(random.uniform(0, 90), 2),   # vertical angle (elevation)
            'a': round(random.uniform(-0.5, 0.5), 3),   # cue lateral offset (fraction of ball radius)
            'b': round(random.uniform(-0.5, 0.5), 3)    # cue longitudinal offset (fraction of ball radius)
        }
        return action