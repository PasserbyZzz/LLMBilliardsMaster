import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

def set_random_seed(enable=False, seed=42):
    # Set random seed for reproducibility
    if enable:
        # Python random
        random.seed(seed)
        # NumPy random
        np.random.seed(seed)
        # PyTorch random (if available)
        if torch is not None:
            torch.manual_seed(seed)  # CPU seed
            torch.cuda.manual_seed(seed)  # current GPU seed
            torch.cuda.manual_seed_all(seed)  # all GPU seeds
            # Make CUDA deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Random seed set to: {seed}")
    # Reset to non-deterministic behavior (use system time)
    else:
        random.seed()
        np.random.seed(None)
        
        print("Random seed disabled, using fully random mode")

def check_path_to_pocket(cue_pos, target_pos, pocket_pos, balls, ball_radius=0.0286, blocking_threshold=None):
    """
    Check whether the path from cue -> target -> pocket is blocked.

    Args:
        cue_pos: cue ball position [x,y,z]
        target_pos: target ball position [x,y,z]
        pocket_pos: pocket position [x,y,z]
        balls: dict of ball objects (uses state.rvw[0] for coords)
        ball_radius: ball radius in meters (default 0.0286)
        blocking_threshold: detection threshold (default 2 * ball_radius)

    Returns:
        (cue_blocked, target_blocked, blocking_balls_cue, blocking_balls_target)
    """
    if blocking_threshold is None:
        blocking_threshold = ball_radius * 2
    
    target_2d = np.array(target_pos[:2])
    pocket_2d = np.array(pocket_pos[:2])
    direction = pocket_2d - target_2d
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return False, False, [], []
    
    direction = direction / dist
    ghost_ball_pos = target_2d - direction * ball_radius * 2
    
    cue_2d = np.array(cue_pos[:2])
    
    blocking_balls_cue = []
    for ball_id, ball in balls.items():
        if ball_id == 'cue' or ball.state.s == 4:
            continue
        ball_pos = ball.state.rvw[0][:2]
        if is_point_near_line_segment(cue_2d, ghost_ball_pos, ball_pos, threshold=blocking_threshold):
            blocking_balls_cue.append(ball_id)
    
    blocking_balls_target = []
    for ball_id, ball in balls.items():
        if ball_id == 'cue' or ball.state.s == 4:
            continue
        ball_pos = ball.state.rvw[0][:2]
        if is_point_near_line_segment(target_2d, pocket_2d, ball_pos, threshold=blocking_threshold):
            blocking_balls_target.append(ball_id)
    
    cue_blocked = len(blocking_balls_cue) > 0
    target_blocked = len(blocking_balls_target) > 0
    
    return cue_blocked, target_blocked, blocking_balls_cue, blocking_balls_target

def is_point_near_line_segment(p1, p2, point, threshold=0.1):
    """
    Determine whether a point is near a line segment.

    Args:
        p1, p2: segment endpoints
        point: point to check
        threshold: distance threshold

    Returns:
        bool: whether point is near the segment
    """
    # Vector calculation
    v = p2 - p1
    w = point - p1
    
    # Projection parameter
    c1 = np.dot(w, v)
    if c1 <= 0:  # before p1
        return np.linalg.norm(w) < threshold
    
    c2 = np.dot(v, v)
    if c1 >= c2:  # after p2
        return np.linalg.norm(point - p2) < threshold
    
    # On the segment
    b = c1 / c2
    pb = p1 + b * v
    return np.linalg.norm(point - pb) < threshold