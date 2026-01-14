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
    """向量归一化"""
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


def calculate_distance(pos1, pos2):
    """计算两点间距离（只考虑x,y平面）"""
    return np.linalg.norm(pos1[:2] - pos2[:2])


def calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos, ball_radius=0.028575):
    """
    计算瞄准点：使目标球沿着 target→pocket 方向运动
    
    参数：
        cue_pos: 白球位置 [x, y, z]
        target_pos: 目标球位置 [x, y, z]
        pocket_pos: 球袋位置 [x, y, z]
        ball_radius: 球半径（米）
    
    返回：
        aim_point: 白球应该击中目标球的点 [x, y, z]
        None: 如果目标球已在球袋位置附近
    """
    target_2d = target_pos[:2]
    pocket_2d = pocket_pos[:2]
    
    # 目标球到球袋的方向
    direction = pocket_2d - target_2d
    dist = np.linalg.norm(direction)
    
    if dist < 1e-6:
        return None
    
    direction = direction / dist
    
    # 瞄准点：目标球背向球袋的一侧
    aim_point_2d = target_2d - direction * ball_radius * 2
    aim_point = np.array([aim_point_2d[0], aim_point_2d[1], target_pos[2]])
    
    return aim_point


def calculate_angle_to_aim_point(cue_pos, aim_point):
    """
    计算白球到瞄准点的水平角度 phi
    
    参数：
        cue_pos: 白球位置 [x, y, z]
        aim_point: 瞄准点 [x, y, z]
    
    返回：
        phi: 水平角度（度），范围 [0, 360)
    """
    dx = aim_point[0] - cue_pos[0]
    dy = aim_point[1] - cue_pos[1]
    
    # 使用 atan2 计算角度
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # 转换到 [0, 360) 范围
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def check_eight_ball_in_path(cue_pos, target_pos, balls, ball_radius=0.028575):
    """
    检查黑八是否在白球到目标球的路径上
    
    参数：
        cue_pos: 白球位置
        target_pos: 目标球位置
        balls: 所有球的状态字典
        ball_radius: 球半径
    
    返回：
        bool: True如果黑八在路径上
        float: 黑八到路径的距离（如果不在路径上返回inf）
    """
    if '8' not in balls or balls['8'].state.s == 4:  # 黑八已进袋
        return False, float('inf')
    
    eight_pos = balls['8'].state.rvw[0]
    
    # 使用更严格的阈值检测黑八
    # 因为碰到黑八是严重犯规，需要更大的安全距离
    safety_threshold = ball_radius * 4  # 4倍球半径作为安全距离
    
    if is_point_near_line_segment(cue_pos[:2], target_pos[:2], eight_pos[:2], threshold=safety_threshold):
        # 计算黑八到路径的精确距离
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
    检查是否有非目标球（对方球或黑八）在路径上
    
    参数：
        cue_pos: 白球位置
        target_pos: 目标球位置  
        my_targets: 己方目标球ID列表
        balls: 所有球的状态字典
        ball_radius: 球半径
    
    返回：
        list: 在路径上的非目标球ID列表
        float: 最近的阻挡球的距离
    """
    blocking_balls = []
    min_dist = float('inf')
    
    for ball_id, ball in balls.items():
        if ball_id == 'cue' or ball.state.s == 4:
            continue
        
        # 如果球在路径上且不是我们要打的目标球
        if ball_id not in my_targets or ball_id == target_pos:
            ball_pos = ball.state.rvw[0]
            
            # 检查是否在路径上
            if is_point_near_line_segment(cue_pos[:2], target_pos[:2], ball_pos[:2], 
                                         threshold=ball_radius * 3):
                blocking_balls.append(ball_id)
                
                # 计算到白球的距离
                dist = calculate_distance(cue_pos, ball_pos)
                min_dist = min(min_dist, dist)
    
    return blocking_balls, min_dist


def calculate_shot_difficulty(cue_pos, target_pos, pocket_pos, balls, target_id=None, 
                             my_targets=None, ball_radius=0.028575):
    """
    计算击球难度（增强版：包含黑八避让逻辑）
    
    参数：
        cue_pos: 白球位置
        target_pos: 目标球位置
        pocket_pos: 球袋位置
        balls: 所有球的状态字典
        target_id: 目标球ID（用于跳过检测）
        my_targets: 己方目标球ID列表（用于判断哪些球是障碍）
        ball_radius: 球半径
    
    返回：
        difficulty: 难度分数（越小越容易）
    """
    # 1. 距离因素
    cue_to_target = calculate_distance(cue_pos, target_pos)
    target_to_pocket = calculate_distance(target_pos, pocket_pos)
    
    # 2. 角度因素
    vec_in = normalize((target_pos - cue_pos)[:2])
    vec_out = normalize((pocket_pos - target_pos)[:2])
    
    dot_product = np.clip(np.dot(vec_in, vec_out), -1.0, 1.0)
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)
    
    # 角度惩罚（0°最容易，角度越大越难）
    angle_penalty = 1.0 + abs(angle_deg) / 90.0
    
    # 3. 遮挡因素
    obstruction_penalty = 1.0
    eight_ball_penalty = 1.0  # 黑八单独惩罚
    
    for ball_id, ball in balls.items():
        # 跳过白球、目标球和已进袋的球
        if ball_id == 'cue' or ball_id == target_id or ball.state.s == 4:
            continue
        
        ball_pos = ball.state.rvw[0]
        
        # 检查是否在白球到目标球的路径上
        if is_point_near_line_segment(cue_pos[:2], target_pos[:2], ball_pos[:2], 
                                     threshold=ball_radius * 3):
            if ball_id == '8':
                # 黑八在路径上 - 极大惩罚！
                eight_ball_penalty = 10.0  # 极大惩罚
            elif my_targets is not None and ball_id not in my_targets:
                # 对方球在路径上 - 较大惩罚
                obstruction_penalty += 2.0
            else:
                # 己方其他球在路径上
                obstruction_penalty += 0.5
        
        # 检查是否在目标球到球袋的路径上
        if is_point_near_line_segment(target_pos[:2], pocket_pos[:2], ball_pos[:2], 
                                     threshold=ball_radius * 3):
            if ball_id == '8':
                eight_ball_penalty = max(eight_ball_penalty, 5.0)
            else:
                obstruction_penalty += 0.3
    
    # 4. 检查黑八是否太近（可能被意外碰到）
    if '8' in balls and balls['8'].state.s != 4:
        eight_pos = balls['8'].state.rvw[0]
        dist_to_eight = calculate_distance(target_pos, eight_pos)
        if dist_to_eight < ball_radius * 6:  # 黑八太靠近目标球
            eight_ball_penalty = max(eight_ball_penalty, 3.0)
    
    # 综合难度
    base_difficulty = (cue_to_target * 0.5 + target_to_pocket * 0.3)
    difficulty = base_difficulty * angle_penalty * obstruction_penalty * eight_ball_penalty
    
    return difficulty


def is_point_near_line_segment(p1, p2, point, threshold=0.1):
    """
    判断点是否在线段附近
    
    参数：
        p1, p2: 线段两端点
        point: 待检测的点
        threshold: 距离阈值
    
    返回：
        bool: 是否在线段附近
    """
    # 向量计算
    v = p2 - p1
    w = point - p1
    
    # 投影参数
    c1 = np.dot(w, v)
    if c1 <= 0:  # 在p1之前
        return np.linalg.norm(w) < threshold
    
    c2 = np.dot(v, v)
    if c1 >= c2:  # 在p2之后
        return np.linalg.norm(point - p2) < threshold
    
    # 在线段上
    b = c1 / c2
    pb = p1 + b * v
    return np.linalg.norm(point - pb) < threshold


def select_best_target(cue_pos, my_targets, balls, table, avoid_eight=True):
    """
    选择最容易打进的目标球（增强版：避免黑八干扰）
    
    参数：
        cue_pos: 白球位置
        my_targets: 目标球ID列表
        balls: 所有球状态
        table: 球桌对象
        avoid_eight: 是否避开黑八路径
    
    返回：
        (best_target_id, best_pocket_id, min_difficulty)
    """
    best_target = None
    best_pocket = None
    min_difficulty = float('inf')
    
    # 收集所有候选方案
    candidates = []
    
    for target_id in my_targets:
        if balls[target_id].state.s == 4:  # 已进袋
            continue
        
        target_pos = balls[target_id].state.rvw[0]
        
        # 尝试每个球袋
        for pocket_id, pocket in table.pockets.items():
            pocket_pos = pocket.center
            
            # 传入my_targets用于判断阻挡球
            difficulty = calculate_shot_difficulty(
                cue_pos, target_pos, pocket_pos, balls, 
                target_id=target_id, my_targets=my_targets
            )
            
            # 检查黑八是否在路径上
            eight_in_path, eight_dist = check_eight_ball_in_path(
                cue_pos, target_pos, balls
            )
            
            # 如果黑八在路径上，增加额外惩罚
            if eight_in_path and avoid_eight:
                difficulty *= 20.0  # 极大惩罚，但不完全排除
            
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
    
    # 如果所有候选方案都有黑八在路径上，选择难度最低的
    if best_target is None and candidates:
        candidates.sort(key=lambda x: x['difficulty'])
        best = candidates[0]
        best_target = best['target_id']
        best_pocket = best['pocket_id']
        min_difficulty = best['difficulty']
    
    return best_target, best_pocket, min_difficulty


def calculate_recommended_velocity(distance):
    """
    根据距离推荐击球速度
    
    参数：
        distance: 白球到目标球的距离（米）
    
    返回：
        V0: 推荐速度（m/s）
    """
    # 近距离用小力，远距离用大力
    # 但不要太大，避免失控
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
    检查白球在击打目标球后是否有落袋风险
    
    原理：白球击打目标球后，可能沿着反弹方向或跟进方向落袋
    
    参数：
        cue_pos: 白球位置
        target_pos: 目标球位置
        table: 球桌对象
        ball_radius: 球半径
    
    返回：
        risk_level: 风险等级 (0-1, 越高越危险)
        risky_pocket_id: 最危险的球袋ID
    """
    max_risk = 0.0
    risky_pocket = None
    
    # 击球方向（白球→目标球）
    shot_dir = normalize((target_pos - cue_pos)[:2])
    
    for pocket_id, pocket in table.pockets.items():
        pocket_pos = pocket.center[:2]
        
        # 1. 检查跟进风险：白球沿击球方向继续前进可能落袋
        # 延长线是否经过袋口附近
        extended_pos = target_pos[:2] + shot_dir * 0.5  # 延长0.5米
        dist_to_pocket = np.linalg.norm(extended_pos - pocket_pos)
        
        if dist_to_pocket < 0.15:  # 袋口半径约0.05-0.06m，留余量
            follow_risk = 1.0 - (dist_to_pocket / 0.15)
            max_risk = max(max_risk, follow_risk * 0.8)
            if follow_risk * 0.8 > max_risk - 0.01:
                risky_pocket = pocket_id
        
        # 2. 检查直接风险：目标球距离袋口很近，白球可能直接跟进
        target_to_pocket = np.linalg.norm(target_pos[:2] - pocket_pos)
        if target_to_pocket < 0.2:
            # 目标球离袋口很近，检查击球角度是否会让白球跟进
            cue_to_pocket = np.linalg.norm(cue_pos[:2] - pocket_pos)
            if cue_to_pocket < 0.4:
                direct_risk = 1.0 - (cue_to_pocket / 0.4)
                max_risk = max(max_risk, direct_risk * 0.6)
                if direct_risk * 0.6 > max_risk - 0.01:
                    risky_pocket = pocket_id
    
    return max_risk, risky_pocket


def predict_first_contact_ball(cue_pos, phi_deg, balls, ball_radius=0.028575):
    """
    几何预判：给定击球角度，白球首先会碰到哪个球
    
    使用射线-圆相交检测
    
    参数：
        cue_pos: 白球位置 [x, y, z]
        phi_deg: 击球水平角度（度）
        balls: 所有球的状态字典
        ball_radius: 球半径
    
    返回：
        first_ball_id: 首次碰撞的球ID，None表示未碰到任何球
        distance: 到首次碰撞球的距离
    """
    phi_rad = math.radians(phi_deg)
    direction = np.array([math.cos(phi_rad), math.sin(phi_rad)])
    
    cue_2d = cue_pos[:2]
    collision_radius = ball_radius * 2  # 两球相切时的圆心距
    
    min_dist = float('inf')
    first_ball = None
    
    for ball_id, ball in balls.items():
        if ball_id == 'cue' or ball.state.s == 4:  # 跳过白球和已进袋的球
            continue
        
        ball_pos = ball.state.rvw[0][:2]
        
        # 射线-圆相交检测
        # 射线: P = cue_2d + t * direction, t >= 0
        # 圆: |P - ball_pos| = collision_radius
        
        oc = cue_2d - ball_pos  # 从圆心到射线起点的向量
        
        a = np.dot(direction, direction)  # 通常为1
        b = 2 * np.dot(oc, direction)
        c = np.dot(oc, oc) - collision_radius ** 2
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # 有交点
            sqrt_d = math.sqrt(discriminant)
            t1 = (-b - sqrt_d) / (2 * a)
            t2 = (-b + sqrt_d) / (2 * a)
            
            # 取最近的正值t
            t = t1 if t1 > 0.001 else t2
            
            if t > 0.001 and t < min_dist:
                min_dist = t
                first_ball = ball_id
    
    return first_ball, min_dist if first_ball else float('inf')


def check_eight_ball_scratch_risk(cue_pos, target_pos, pocket_pos, balls, ball_radius=0.028575):
    """
    检查打黑8时白球同时落袋的风险（scratch）
    
    这是打黑8时最致命的犯规，需要特别防范
    
    参数：
        cue_pos: 白球位置
        target_pos: 黑8位置
        pocket_pos: 目标袋位置
        balls: 所有球状态
        ball_radius: 球半径
    
    返回：
        risk_level: 风险等级 (0-1)
        risk_type: 风险类型描述
    """
    shot_dir = normalize((target_pos - cue_pos)[:2])
    target_to_pocket_dir = normalize((pocket_pos[:2] - target_pos[:2]))
    
    # 计算击球角度（入射角）
    cos_angle = np.dot(shot_dir, target_to_pocket_dir)
    
    risk_level = 0.0
    risk_type = "safe"
    
    # 1. 正面碰撞风险：角度接近180°（直线球），白球可能跟进
    if cos_angle > 0.85:  # 接近直线
        # 检查目标袋附近是否有空间
        dist_target_to_pocket = np.linalg.norm(target_pos[:2] - pocket_pos[:2])
        if dist_target_to_pocket < 0.3:
            risk_level = max(risk_level, 0.8 * cos_angle)
            risk_type = "follow-through"
    
    # 2. 对角袋风险：白球可能被弹向另一个袋
    # 计算白球反弹方向
    # 简化模型：白球沿垂直于碰撞点切线方向反弹
    reflect_dir = shot_dir - 2 * np.dot(shot_dir, target_to_pocket_dir) * target_to_pocket_dir
    
    # 检查反弹方向是否朝向某个袋
    for pocket_id, pocket in balls.items():
        # 这里需要table对象，暂时跳过
        pass
    
    return risk_level, risk_type