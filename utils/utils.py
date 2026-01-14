import random
import numpy as np
import math

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


##——————————————————————————————————————————————————————————————————————————————————————###
### This part below is for AlgorithmicAgent's  geometry calculations, not for LLMAgents ###
##——————————————————————————————————————————————————————————————————————————————————————###
def normalize(vec):
    """Normalize a vector"""
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def calculate_distance(pos1, pos2):
    """Compute distance between two points (x/y plane only)"""
    return np.linalg.norm(pos1[:2] - pos2[:2])


def calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos, ball_radius=0.028575):
    """
    Compute the aim point so the target ball travels along the target→pocket direction
    
    Args:
        cue_pos: Cue ball position [x, y, z]
        target_pos: Target ball position [x, y, z]
        pocket_pos: Pocket position [x, y, z]
        ball_radius: Ball radius in meters
    
    Returns:
        aim_point: The point where the cue ball should contact the target ball [x, y, z]
        None: If the target ball is already near the pocket position
    """
    target_2d = target_pos[:2]
    pocket_2d = pocket_pos[:2]
    
    # Direction from the target ball to the pocket
    direction = pocket_2d - target_2d
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return None
    
    direction = direction / dist
    
    # Aim point: the side of the target ball opposite the pocket
    aim_point_2d = target_2d - direction * ball_radius * 2
    aim_point = np.array([aim_point_2d[0], aim_point_2d[1], target_pos[2]])
    
    return aim_point


def calculate_angle_to_aim_point(cue_pos, aim_point):
    """
    Compute the horizontal angle (phi) from the cue ball to the aim point
    
    Args:
        cue_pos: Cue ball position [x, y, z]
        aim_point: Aim point [x, y, z]
    
    Returns:
        phi: Horizontal angle in degrees, range [0, 360)
    """
    dx = aim_point[0] - cue_pos[0]
    dy = aim_point[1] - cue_pos[1]
    
    # Use atan2 to compute the angle
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Convert to [0, 360) range
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def check_eight_ball_in_path(cue_pos, target_pos, balls, ball_radius=0.028575):
    """
    Check whether the 8-ball lies on the path from the cue ball to the target ball
    
    Args:
        cue_pos: Cue ball position
        target_pos: Target ball position
        balls: Dict of all ball states
        ball_radius: Ball radius
    
    Returns:
        bool: True if the 8-ball is on/near the path
        float: Distance from the 8-ball to the path (inf if not on/near the path)
    """
    if '8' not in balls or balls['8'].state.s == 4:  # 8-ball already pocketed
        return False, float('inf')
    
    eight_pos = balls['8'].state.rvw[0]
    
    # Use a stricter threshold for the 8-ball:
    # touching the 8-ball is a serious foul, so we keep a larger safety margin.
    safety_threshold = ball_radius * 4  # 4x ball radius as safety distance
    
    if is_point_near_line_segment(cue_pos[:2], target_pos[:2], eight_pos[:2], threshold=safety_threshold):
        # Compute the exact distance from the 8-ball to the path
        v = target_pos[:2] - cue_pos[:2]
        w = eight_pos[:2] - cue_pos[:2]
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
        if c2 > 1e-9:
            b = c1 / c2
            closest_point = cue_pos[:2] + b * v
            dist = np.linalg.norm(eight_pos[:2] - closest_point)
            return True, dist
        return True, 0.0
    
    return False, float('inf')


def check_other_balls_in_path(cue_pos, target_pos, my_targets, balls, ball_radius=0.028575):
    """
    Check whether any non-target balls (opponent balls or the 8-ball) lie on the path
    
    Args:
        cue_pos: Cue ball position
        target_pos: Target ball position
        my_targets: List of this player's target ball IDs
        balls: Dict of all ball states
        ball_radius: Ball radius
    
    Returns:
        list: IDs of non-target balls that are on/near the path
        float: Distance to the closest blocking ball
    """
    blocking_balls = []
    min_dist = float('inf')
    
    for ball_id, ball in balls.items():
        if ball_id == 'cue' or ball.state.s == 4:
            continue
        
        # If the ball is on/near the path and is not the intended target
        if ball_id not in my_targets or ball_id == target_pos:
            ball_pos = ball.state.rvw[0]
            
            # Check whether it lies on/near the path
            if is_point_near_line_segment(cue_pos[:2], target_pos[:2], ball_pos[:2], 
                                         threshold=ball_radius * 3):
                blocking_balls.append(ball_id)
                
                # Compute distance to the cue ball
                dist = calculate_distance(cue_pos, ball_pos)
                min_dist = min(min_dist, dist)
    
    return blocking_balls, min_dist


def calculate_shot_difficulty(cue_pos, target_pos, pocket_pos, balls, target_id=None, 
                             my_targets=None, ball_radius=0.028575):
    """
    Compute shot difficulty (enhanced: includes 8-ball avoidance logic)
    
    Args:
        cue_pos: Cue ball position
        target_pos: Target ball position
        pocket_pos: Pocket position
        balls: Dict of all ball states
        target_id: Target ball ID (used to skip checks)
        my_targets: This player's target ball IDs (used to classify obstacles)
        ball_radius: Ball radius
    
    Returns:
        difficulty: Difficulty score (lower is easier)
    """
    # 1. Distance factor
    cue_to_target = calculate_distance(cue_pos, target_pos)
    target_to_pocket = calculate_distance(target_pos, pocket_pos)
    
    # 2. Angle factor
    vec_in = normalize((target_pos - cue_pos)[:2])
    vec_out = normalize((pocket_pos - target_pos)[:2])
    
    dot_product = np.clip(np.dot(vec_in, vec_out), -1.0, 1.0)
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)
    
    # Angle penalty (0° easiest; larger angles are harder)
    angle_penalty = 1.0 + abs(angle_deg) / 90.0
    
    # 3. Obstruction factor
    obstruction_penalty = 1.0
    eight_ball_penalty = 1.0  # Separate penalty for the 8-ball
    
    for ball_id, ball in balls.items():
        # Skip the cue ball, the target ball, and pocketed balls
        if ball_id == 'cue' or ball_id == target_id or ball.state.s == 4:
            continue
        
        ball_pos = ball.state.rvw[0]
        
        # Check whether it lies on/near the cue-to-target path
        if is_point_near_line_segment(cue_pos[:2], target_pos[:2], ball_pos[:2], 
                                     threshold=ball_radius * 3):
            if ball_id == '8':
                # 8-ball on the path: large penalty
                eight_ball_penalty = 10.0
            elif my_targets is not None and ball_id not in my_targets:
                # Opponent ball on the path: strong penalty
                obstruction_penalty += 2.0
            else:
                # Another of our balls on the path
                obstruction_penalty += 0.5
        
        # Check whether it lies on/near the target-to-pocket path
        if is_point_near_line_segment(target_pos[:2], pocket_pos[:2], ball_pos[:2], 
                                     threshold=ball_radius * 3):
            if ball_id == '8':
                eight_ball_penalty = max(eight_ball_penalty, 5.0)
            else:
                obstruction_penalty += 0.3
    
    # 4. Check whether the 8-ball is too close (risk of accidental contact)
    if '8' in balls and balls['8'].state.s != 4:
        eight_pos = balls['8'].state.rvw[0]
        dist_to_eight = calculate_distance(target_pos, eight_pos)
        if dist_to_eight < ball_radius * 6:  # 8-ball too close to target ball
            eight_ball_penalty = max(eight_ball_penalty, 3.0)
    
    # Combined difficulty
    base_difficulty = (cue_to_target * 0.5 + target_to_pocket * 0.3)
    difficulty = base_difficulty * angle_penalty * obstruction_penalty * eight_ball_penalty
    
    return difficulty


def is_point_near_line_segment(p1, p2, point, threshold=0.1):
    """
    Determine whether a point is near a line segment
    
    Args:
        p1, p2: Segment endpoints
        point: Point to check
        threshold: Distance threshold
    
    Returns:
        bool: Whether the point is near the segment
    """
    # Vector calculation
    v = p2 - p1
    w = point - p1
    
    # Projection parameter
    c1 = np.dot(w, v)
    if c1 <= 0:  # Before p1
        return np.linalg.norm(w) < threshold
    
    c2 = np.dot(v, v)
    if c1 >= c2:  # After p2
        return np.linalg.norm(point - p2) < threshold
    
    # On the segment
    b = c1 / c2
    pb = p1 + b * v
    return np.linalg.norm(point - pb) < threshold


def select_best_target(cue_pos, my_targets, balls, table, avoid_eight=True):
    """
    Select the easiest target ball to pot (enhanced: avoids 8-ball interference)
    
    Args:
        cue_pos: Cue ball position
        my_targets: Target ball ID list
        balls: All ball states
        table: Table object
        avoid_eight: Whether to avoid paths near the 8-ball
    
    Returns:
        (best_target_id, best_pocket_id, min_difficulty)
    """
    best_target = None
    best_pocket = None
    min_difficulty = float('inf')
    
    # Collect all candidate options
    candidates = []
    
    for target_id in my_targets:
        if balls[target_id].state.s == 4:  # Already pocketed
            continue
        
        target_pos = balls[target_id].state.rvw[0]
        
        # Try each pocket
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center
            
            # Pass my_targets to classify blocking balls
            difficulty = calculate_shot_difficulty(
                cue_pos, target_pos, pocket_pos, balls, 
                target_id=target_id, my_targets=my_targets
            )
            
            # Check whether the 8-ball is on/near the path
            eight_in_path, eight_dist = check_eight_ball_in_path(
                cue_pos, target_pos, balls
            )
            
            # If the 8-ball is on the path, apply an extra penalty
            if eight_in_path and avoid_eight:
                difficulty *= 20.0  # Large penalty, but do not fully exclude
            
            candidates.append({
                'target_id': target_id,
                'pocket_id': pocket_id,
                'difficulty': difficulty,
                'eight_in_path': eight_in_path
            })
            
            if difficulty < min_difficulty:
                min_difficulty = difficulty
                best_target = target_id
                best_pocket = pocket_id
    
    # If all candidates have the 8-ball on the path, select the lowest difficulty
    if best_target is None and candidates:
        candidates.sort(key=lambda x: x['difficulty'])
        best = candidates[0]
        best_target = best['target_id']
        best_pocket = best['pocket_id']
        min_difficulty = best['difficulty']
    
    return best_target, best_pocket, min_difficulty


def calculate_recommended_velocity(distance):
    """
    Recommend shot speed based on distance
    
    Args:
        distance: Distance from cue ball to target ball (meters)
    
    Returns:
        V0: Recommended speed (m/s)
    """
    # Use less power for short distances and more for longer distances,
    # but avoid excessive speed to reduce loss of control.
    if distance < 0.3:
        return 1.0
    elif distance < 0.6:
        return 2.0
    elif distance < 1.0:
        return 3.0
    elif distance < 1.5:
        return 4.0
    else:
        return 5.0


def check_cue_ball_pocket_risk(cue_pos, target_pos, table, ball_radius=0.028575):
    """
    Check whether the cue ball has a pocketing (scratch) risk after striking the target ball
    
    Rationale: after impact, the cue ball may scratch along a rebound direction or via follow-through
    
    Args:
        cue_pos: Cue ball position
        target_pos: Target ball position
        table: Table object
        ball_radius: Ball radius
    
    Returns:
        risk_level: Risk level (0-1, higher is riskier)
        risky_pocket_id: ID of the riskiest pocket
    """
    max_risk = 0.0
    risky_pocket = None
    
    # Shot direction (cue → target)
    shot_dir = normalize((target_pos - cue_pos)[:2])
    
    for pocket_id, pocket in table.pockets.items():
        pocket_pos = pocket.center[:2]
        
        # 1) Follow-through risk: cue ball continues along shot direction and may scratch
        # Check whether the extended line passes near the pocket mouth
        extended_pos = target_pos[:2] + shot_dir * 0.5  # Extend by 0.5m
        dist_to_pocket = np.linalg.norm(extended_pos - pocket_pos)
        
        if dist_to_pocket < 0.15:  # Pocket mouth radius ~0.05–0.06m, plus margin
            follow_risk = 1.0 - (dist_to_pocket / 0.15)
            max_risk = max(max_risk, follow_risk * 0.8)
            if follow_risk * 0.8 > max_risk - 0.01:
                risky_pocket = pocket_id
        
        # 2) Direct risk: if the target ball is near the pocket, the cue ball may follow in
        target_to_pocket = np.linalg.norm(target_pos[:2] - pocket_pos)
        if target_to_pocket < 0.2:
            # Target is close to the pocket; check whether geometry suggests a follow-in scratch
            cue_to_pocket = np.linalg.norm(cue_pos[:2] - pocket_pos)
            if cue_to_pocket < 0.4:
                direct_risk = 1.0 - (cue_to_pocket / 0.4)
                max_risk = max(max_risk, direct_risk * 0.6)
                if direct_risk * 0.6 > max_risk - 0.01:
                    risky_pocket = pocket_id
    
    return max_risk, risky_pocket


def predict_first_contact_ball(cue_pos, phi_deg, balls, ball_radius=0.028575):
    """
    Geometric prediction: given a shot angle, determine which ball the cue ball hits first
    
    Uses ray–circle intersection testing.
    
    Args:
        cue_pos: Cue ball position [x, y, z]
        phi_deg: Shot horizontal angle (degrees)
        balls: Dict of all ball states
        ball_radius: Ball radius
    
    Returns:
        first_ball_id: ID of the first ball contacted; None if no ball is contacted
        distance: Distance to the first-contact ball
    """
    phi_rad = math.radians(phi_deg)
    direction = np.array([math.cos(phi_rad), math.sin(phi_rad)])
    
    cue_2d = cue_pos[:2]
    collision_radius = ball_radius * 2  # Center distance when two balls are tangent
    
    min_dist = float('inf')
    first_ball = None
    
    for ball_id, ball in balls.items():
        if ball_id == 'cue' or ball.state.s == 4:  # Skip cue ball and pocketed balls
            continue
        
        ball_pos = ball.state.rvw[0][:2]
        
        # Ray–circle intersection test
        # Ray: P = cue_2d + t * direction, t >= 0
        # Circle: |P - ball_pos| = collision_radius
        
        oc = cue_2d - ball_pos  # Vector from circle center to ray origin
        
        a = np.dot(direction, direction)  # Typically 1
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - collision_radius ** 2
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # Has intersection(s)
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2 * a)
            t2 = (-b + sqrt_d) / (2 * a)
            
            # Choose the nearest positive t
            t = t1 if t1 > 0.001 else t2
            
            if t > 0.001 and t < min_dist:
                min_dist = t
                first_ball = ball_id
    
    return first_ball, min_dist if first_ball else float('inf')


def check_eight_ball_scratch_risk(cue_pos, target_pos, pocket_pos, balls, ball_radius=0.028575):
    """
    Check the risk of scratching the cue ball when shooting the 8-ball
    
    This is the most critical foul when shooting the 8-ball and should be avoided.
    
    Args:
        cue_pos: Cue ball position
        target_pos: 8-ball position
        pocket_pos: Target pocket position
        balls: All ball states
        ball_radius: Ball radius
    
    Returns:
        risk_level: Risk level (0-1)
        risk_type: Risk type description
    """
    shot_dir = normalize((target_pos - cue_pos)[:2])
    target_to_pocket_dir = normalize((pocket_pos[:2] - target_pos[:2]))
    
    # Compute the shot angle (incident angle)
    cos_angle = np.dot(shot_dir, target_to_pocket_dir)
    
    risk_level = 0.0
    risk_type = "safe"
    
    # 1) Full-hit risk: near-straight shot, the cue ball may follow through and scratch
    if cos_angle > 0.85:  # Near-straight shot
        # Check whether there is space near the target pocket
        dist_target_to_pocket = np.linalg.norm(target_pos[:2] - pocket_pos[:2])
        if dist_target_to_pocket < 0.3:
            risk_level = max(risk_level, 0.8 * cos_angle)
            risk_type = "follow-through"
    
    # 2) Cross-pocket risk: the cue ball may be deflected toward another pocket
    # Compute a simplified reflection direction
    reflect_dir = shot_dir - 2 * np.dot(shot_dir, target_to_pocket_dir) * target_to_pocket_dir
    
    # Check whether the reflection direction points toward a pocket
    for pocket_id, pocket in balls.items():
        # This requires a table object; currently skipped.
        pass
    
    return risk_level, risk_type
