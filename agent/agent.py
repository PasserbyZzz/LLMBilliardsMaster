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
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移 (单位：球半径比例)
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移 (单位：球半径比例)
        }
        return action