
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


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
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
        
    # 计算奖励分数
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
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action


class Enhanced_Bayes_Agent(Agent): 
    """NewAgent（几何引导 + 贝叶斯优化 + 增强安全策略）"""
    
    def __init__(self):
        super().__init__()
        
        # 导入几何工具
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
        
        # 贝叶斯优化参数（优化后）
        self.INITIAL_SEARCH = 30  # 增加初始采样
        self.OPT_SEARCH = 15    # 增加优化迭代
        self.ALPHA = 5e-4         # 降低噪声
        
        # 几何搜索参数（收紧搜索范围）
        self.phi_range = 40       # 角度搜索范围 ±40°（从75°缩小）
        self.V0_range = 2.0       # 速度搜索范围 ±2.0 m/s（从3.0缩小）
        self.theta_max = 45       # 最大仰角（从60°缩小）
        self.offset_range = 0.30  # 击球点偏移范围（从0.40缩小）
        
        # 黑八避让参数（增强惩罚）
        self.eight_ball_penalty = 400    # 误打黑八的惩罚（提高）
        self.wrong_ball_penalty = 250    # 误打对方球的惩罚（提高）
        self.cue_pocket_penalty = 350    # 白球落袋惩罚（新增）
        self.scratch_eight_penalty = 600 # 白球+黑8同时落袋惩罚（新增，致命）
        
        # 打黑8时的特殊保守参数
        self.eight_ball_V0_max = 5.0     # 打黑8时最大速度
        self.eight_ball_theta_max = 25   # 打黑8时最大仰角
        self.eight_ball_offset_range = 0.20  # 打黑8时偏移范围
        
        print("Enhanced_BayesAgent (Geometry-Guided Bayesian Optimization v2) 已初始化。")
    
    def print_config(self):
        """打印所有重要超参数"""
        print("\n" + "="*50)
        print("[Enhanced_BayesAgent 超参数配置]")
        print(f"  贝叶斯优化参数:")
        print(f"    - INITIAL_SEARCH (初始随机采样): {self.INITIAL_SEARCH}")
        print(f"    - OPT_SEARCH (贝叶斯优化迭代): {self.OPT_SEARCH}")
        print(f"    - ALPHA (高斯过程噪声): {self.ALPHA}")
        print(f"  几何启发式参数:")
        print(f"    - phi_range (角度搜索范围): ±{self.phi_range}°")
        print(f"    - V0_range (速度搜索范围): ±{self.V0_range} m/s")
        print(f"    - theta_max (最大仰角): {self.theta_max}°")
        print(f"    - offset_range (击球点偏移): ±{self.offset_range}")
        print(f"  惩罚参数:")
        print(f"    - eight_ball_penalty (误打黑8): {self.eight_ball_penalty}")
        print(f"    - wrong_ball_penalty (误打对方球): {self.wrong_ball_penalty}")
        print(f"    - cue_pocket_penalty (白球落袋): {self.cue_pocket_penalty}")
        print(f"    - scratch_eight_penalty (白球+黑8同落): {self.scratch_eight_penalty}")
        print(f"  打黑8特殊参数:")
        print(f"    - eight_ball_V0_max: {self.eight_ball_V0_max} m/s")
        print(f"    - eight_ball_theta_max: {self.eight_ball_theta_max}°")
        print(f"    - eight_ball_offset_range: ±{self.eight_ball_offset_range}")
        print("="*50)
    
    def _analyze_first_contact(self, shot, valid_ball_ids):
        """分析击球后的首次碰撞
        
        返回：
            first_contact_ball_id: 首次碰撞的球ID
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
        """基于几何启发式的决策（增强版v2：白球保护 + 黑八安全击打）
        
        步骤：
        1. 用几何学选择最容易的目标球和球袋（考虑黑八）
        2. 计算理想瞄准角度
        3. 几何预检测：首撞球验证、白球落袋风险
        4. 在瞄准角度附近用贝叶斯优化搜索（带增强惩罚）
        5. 打黑8时采用特殊保守策略
        
        参数：
            balls: 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
        
        返回：
            dict: 击球动作
        """
        if balls is None or my_targets is None or table is None:
            print("[NewAgent] 缺少必要信息，使用随机动作。")
            return self._random_action()
        
        try:
            # 保存状态快照
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 检查目标球是否已清空
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_eight = False
            if len(remaining_own) == 0:
                my_targets = ["8"]
                is_shooting_eight = True
                print("[NewAgent] 目标球已清空，切换到黑8。⚠️ 进入谨慎模式！")
            
            # 步骤1: 几何启发式选择目标（考虑黑八避让）
            cue_pos = balls['cue'].state.rvw[0]
            best_target_id, best_pocket_id, difficulty = self.select_best_target(
                cue_pos, my_targets, balls, table, avoid_eight=True
            )
            
            if best_target_id is None:
                print("[NewAgent] 未找到有效目标球，使用随机动作。")
                return self._random_action()
            
            target_pos = balls[best_target_id].state.rvw[0]
            pocket_pos = table.pockets[best_pocket_id].center
            
            # 检查路径上是否有黑八（用于调整策略）
            eight_in_path, eight_dist = self.check_eight_ball_in_path(cue_pos, target_pos, balls)
            
            # 检查白球落袋风险
            cue_pocket_risk, risky_pocket = self.check_cue_ball_pocket_risk(cue_pos, target_pos, table)
            
            # 几何预判首撞球
            predicted_first, _ = self.predict_first_contact_ball(
                cue_pos, 
                self.calculate_angle_to_aim_point(cue_pos, self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)),
                balls
            )
            
            warning_msg = ""
            if eight_in_path:
                warning_msg += " (警告：黑八在路径附近!)"
            if cue_pocket_risk > 0.5:
                warning_msg += f" (警告：白球落袋风险高={cue_pocket_risk:.2f})"
            if predicted_first and predicted_first != best_target_id:
                warning_msg += f" (警告：预判首撞={predicted_first})"
            
            print(f"[NewAgent] 选择目标: {best_target_id} → 球袋: {best_pocket_id}, 难度: {difficulty:.2f}{warning_msg}")
            
            # 步骤2: 计算几何瞄准参数
            aim_point = self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)
            
            if aim_point is None:
                print("[NewAgent] 无法计算瞄准点，使用随机动作。")
                return self._random_action()
            
            # 计算理想角度
            ideal_phi = self.calculate_angle_to_aim_point(cue_pos, aim_point)
            
            # 计算推荐速度
            distance = self.calculate_distance(cue_pos, target_pos)
            ideal_V0 = self.calculate_recommended_velocity(distance)
            
            # 打黑8时额外检查scratch风险
            scratch_risk = 0.0
            if is_shooting_eight:
                scratch_risk, risk_type = self.check_eight_ball_scratch_risk(
                    cue_pos, target_pos, pocket_pos, balls
                )
                if scratch_risk > 0.3:
                    print(f"[NewAgent] ⚠️ 打黑8 scratch风险: {scratch_risk:.2f} ({risk_type})，降低力度")
                    ideal_V0 = min(ideal_V0, self.eight_ball_V0_max * (1 - scratch_risk * 0.5))
            
            print(f"[NewAgent] 几何解: phi={ideal_phi:.2f}°, V0={ideal_V0:.2f} m/s, 距离={distance:.3f}m")
            
            # 步骤3: 构建搜索空间（根据路径风险和是否打黑8调整）
            if is_shooting_eight:
                # 打黑8时使用更保守的参数
                phi_range = self.phi_range * 0.4
                V0_range = self.V0_range * 0.5
                V0_max = self.eight_ball_V0_max
                theta_max = self.eight_ball_theta_max
                offset_range = self.eight_ball_offset_range
                print(f"[NewAgent] 黑8模式：收紧搜索范围 phi±{phi_range:.1f}°, V0±{V0_range:.1f}, θ≤{theta_max}°")
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
            
            # 如果白球落袋风险高，进一步限制
            if cue_pocket_risk > 0.5:
                V0_max = min(V0_max, 3.5)
                theta_max = min(theta_max, 25)
                offset_range *= 0.7
            
            # 为避免边界收缩为同一点，添加安全边界函数
            def _safe_bounds(center, span, min_v, max_v, eps=1e-3):
                low = max(min_v, center - span)
                high = min(max_v, center + span)
                if high - low < eps:
                    mid = (low + high) / 2
                    low = max(min_v, mid - eps / 2)
                    high = min(max_v, mid + eps / 2)
                    if high - low < eps:  # 极端情况下再扩一点点
                        high = min(max_v, low + eps)
                return low, high

            V0_low, V0_high = _safe_bounds(ideal_V0, V0_range, 0.8, V0_max)  # 提升最小力度，避免0或极小力度导致数值问题
            phi_low, phi_high = _safe_bounds(ideal_phi, phi_range, -720.0, 1080.0)  # 放宽允许范围，内部会再取模

            pbounds = {
                'V0': (V0_low, V0_high),
                'phi': (phi_low, phi_high),
                'theta': (0, max(theta_max, 1e-2)),
                'a': (-offset_range, offset_range),
                'b': (-offset_range, offset_range)
            }
            
            # 步骤4: 定义增强奖励函数（考虑首次碰撞、白球落袋、黑8+白球同落）
            valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
            
            def reward_fn_wrapper(V0, phi, theta, a, b):
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")
                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    # 处理角度边界
                    phi_normalized = phi % 360
                    shot.cue.set_state(V0=V0, phi=phi_normalized, theta=theta, a=a, b=b)
                    
                    # 使用超时保护（避免物理引擎卡死）
                    if not simulate_with_timeout(shot, timeout=15):
                        return 0  # 超时返回中性分数
                except Exception as e:
                    return -500
                
                # 基础得分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )
                
                # 额外检测：分析模拟结果
                first_contact = self._analyze_first_contact(shot, valid_ball_ids)
                
                # 检测白球和黑8是否都落袋（致命犯规）
                cue_pocketed = shot.balls['cue'].state.s == 4
                eight_pocketed = '8' in shot.balls and shot.balls['8'].state.s == 4 and last_state_snapshot['8'].state.s != 4
                
                if cue_pocketed and eight_pocketed:
                    # 白球+黑8同时落袋 - 最严重的犯规
                    score -= self.scratch_eight_penalty
                elif cue_pocketed:
                    # 仅白球落袋
                    score -= self.cue_pocket_penalty
                
                # 首次碰撞惩罚
                if first_contact is not None:
                    if first_contact not in my_targets:
                        if first_contact == '8' and my_targets != ['8']:
                            score -= self.eight_ball_penalty  # 误打黑八
                        else:
                            score -= self.wrong_ball_penalty  # 误打对方球
                
                # 打黑8时的额外保护：即使没落袋，速度太大也要惩罚
                if is_shooting_eight and V0 > self.eight_ball_V0_max:
                    score -= (V0 - self.eight_ball_V0_max) * 20
                
                return score
            
            # 步骤5: 贝叶斯优化
            print(f"[NewAgent] 在几何解附近优化...")
            
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
            
            # 添加几何解作为初始探测点
            optimizer.probe(
                params={
                    'V0': np.clip(ideal_V0, V0_low, V0_high),
                    'phi': np.clip(ideal_phi, phi_low, phi_high),
                    'theta': 5.0,  # 小仰角
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
                print(f"[NewAgent] 优化后分数仍然很低 ({best_score:.2f})，使用几何解。")
                # 回退到几何解（打黑8时使用更保守的速度）
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
            
            print(f"[NewAgent] 最终决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[NewAgent] 决策时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class GeometricAgent(Agent):
    """纯几何决策 Agent（无贝叶斯优化）
    
    特点：
    - 完全基于几何计算，无需物理仿真优化
    - 速度极快，适合实时决策
    - 使用启发式规则确保安全性和准确性
    """
    
    def __init__(self):
        super().__init__()
        
        # 导入几何工具
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
        
        # 几何参数调整系数
        self.angle_adjustment_factor = 0.98  # 角度微调系数（应对系统性偏差）
        self.velocity_safety_factor = 0.9    # 速度安全系数（避免过猛）
        
        # 打黑8时的保守参数
        self.eight_ball_velocity_limit = 4.5  # 打黑8最大速度
        self.eight_ball_theta_limit = 20      # 打黑8最大仰角
        
        # 一般情况下的参数
        self.normal_theta_max = 35            # 普通击球最大仰角
        self.risky_velocity_limit = 3.0       # 高风险情况速度限制
        
        print("GeometricAgent (Pure Geometry, No Bayesian Optimization) 已初始化。")
    
    def print_config(self):
        """打印所有重要超参数"""
        print("\n" + "="*50)
        print("[GeometricAgent 超参数配置]")
        print(f"  几何调整参数:")
        print(f"    - angle_adjustment_factor (角度微调系数): {self.angle_adjustment_factor}")
        print(f"    - velocity_safety_factor (速度安全系数): {self.velocity_safety_factor}")
        print(f"  打黑8特殊参数:")
        print(f"    - eight_ball_velocity_limit: {self.eight_ball_velocity_limit} m/s")
        print(f"    - eight_ball_theta_limit: {self.eight_ball_theta_limit}°")
        print(f"  一般击球参数:")
        print(f"    - normal_theta_max (普通最大仰角): {self.normal_theta_max}°")
        print(f"    - risky_velocity_limit (高风险速度限制): {self.risky_velocity_limit} m/s")
        print(f"  特点: 纯几何计算，无贝叶斯优化，速度极快")
        print("="*50)
    
    def _calculate_cut_angle_adjustment(self, cue_pos, target_pos, pocket_pos):
        """计算切球角度调整
        
        根据白球-目标球-球袋的几何关系，计算需要的切球角度
        返回建议的theta和offset调整
        """
        # 计算入射向量和出射向量的夹角
        vec_in = (target_pos - cue_pos)[:2]
        vec_out = (pocket_pos - target_pos)[:2]
        
        vec_in_norm = vec_in / (np.linalg.norm(vec_in) + 1e-9)
        vec_out_norm = vec_out / (np.linalg.norm(vec_out) + 1e-9)
        
        dot_product = np.clip(np.dot(vec_in_norm, vec_out_norm), -1.0, 1.0)
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # 根据角度决定theta和offset
        if angle_deg < 15:  # 直线球
            theta = 3.0
            a_offset = 0.0
            b_offset = 0.0
        elif angle_deg < 45:  # 小角度切球
            theta = 8.0
            # 轻微的侧旋帮助控球
            a_offset = 0.05 if angle_deg < 30 else 0.10
            b_offset = 0.0
        elif angle_deg < 90:  # 中等角度切球
            theta = 15.0
            a_offset = 0.15
            b_offset = 0.05
        else:  # 大角度切球（困难球）
            theta = 25.0
            a_offset = 0.20
            b_offset = 0.10
        
        return theta, a_offset, b_offset, angle_deg
    
    def _adjust_for_distance(self, distance, base_velocity):
        """根据距离微调速度
        
        考虑摩擦力损耗，距离越远需要略微增加速度
        """
        if distance < 0.3:
            # 近距离：轻柔击球
            return base_velocity * 0.85
        elif distance < 0.6:
            return base_velocity * 0.95
        elif distance < 1.0:
            return base_velocity
        elif distance < 1.5:
            # 中等距离：稍微增加力度
            return base_velocity * 1.1
        else:
            # 远距离：显著增加力度，但不超过安全上限
            return min(base_velocity * 1.25, 6.5)
    
    def decision(self, balls=None, my_targets=None, table=None):
        """纯几何决策（无贝叶斯优化）
        
        步骤：
        1. 选择最佳目标球和球袋
        2. 计算几何瞄准点和角度
        3. 根据距离、角度、风险等因素计算速度和击球参数
        4. 应用安全性检查和调整
        5. 直接返回几何解（无优化步骤）
        """
        if balls is None or my_targets is None or table is None:
            print("[GeometricAgent] 缺少必要信息，使用随机动作。")
            return self._random_action()
        
        try:
            # 检查目标球是否已清空
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_eight = False
            if len(remaining_own) == 0:
                my_targets = ["8"]
                is_shooting_eight = True
                print("[GeometricAgent] 目标球已清空，切换到黑8。⚠️ 进入保守模式！")
            
            # 步骤1: 选择目标球和球袋
            cue_pos = balls['cue'].state.rvw[0]
            best_target_id, best_pocket_id, difficulty = self.select_best_target(
                cue_pos, my_targets, balls, table, avoid_eight=True
            )
            
            if best_target_id is None:
                print("[GeometricAgent] 未找到有效目标球，使用随机动作。")
                return self._random_action()
            
            target_pos = balls[best_target_id].state.rvw[0]
            pocket_pos = table.pockets[best_pocket_id].center
            
            # 步骤2: 风险评估
            eight_in_path, eight_dist = self.check_eight_ball_in_path(cue_pos, target_pos, balls)
            cue_pocket_risk, risky_pocket = self.check_cue_ball_pocket_risk(cue_pos, target_pos, table)
            
            # 步骤3: 计算几何瞄准参数
            aim_point = self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)
            
            if aim_point is None:
                print("[GeometricAgent] 无法计算瞄准点，使用随机动作。")
                return self._random_action()
            
            # 计算理想角度
            ideal_phi = self.calculate_angle_to_aim_point(cue_pos, aim_point)
            
            # 应用角度调整因子
            phi = ideal_phi * self.angle_adjustment_factor
            
            # 步骤4: 计算速度
            distance = self.calculate_distance(cue_pos, target_pos)
            base_velocity = self.calculate_recommended_velocity(distance)
            
            # 根据距离微调
            velocity = self._adjust_for_distance(distance, base_velocity)
            
            # 应用速度安全系数
            velocity = velocity * self.velocity_safety_factor
            
            # 步骤5: 计算切球参数
            theta, a_offset, b_offset, cut_angle = self._calculate_cut_angle_adjustment(
                cue_pos, target_pos, pocket_pos
            )
            
            # 步骤6: 根据特殊情况调整
            warning_msg = ""
            
            # 打黑8时的特殊处理
            if is_shooting_eight:
                scratch_risk, risk_type = self.check_eight_ball_scratch_risk(
                    cue_pos, target_pos, pocket_pos, balls
                )
                
                # 极度保守的黑8策略
                velocity = min(velocity, self.eight_ball_velocity_limit)
                theta = min(theta, self.eight_ball_theta_limit)
                
                # 如果scratch风险高，进一步降低速度
                if scratch_risk > 0.3:
                    velocity = velocity * (1.0 - scratch_risk * 0.5)
                    warning_msg += f" (Scratch风险={scratch_risk:.2f})"
                
                # 黑8时尽量中心击球
                a_offset = a_offset * 0.5
                b_offset = b_offset * 0.5
                
                print(f"[GeometricAgent] 黑8模式：V0={velocity:.2f}, θ={theta:.2f}°{warning_msg}")
            
            # 黑八在路径上的处理
            elif eight_in_path:
                # 尝试通过调整角度避开黑八
                # 这里使用简单策略：略微偏离理想角度
                phi_adjustment = 2.0 if eight_dist < 0.1 else 1.0
                phi = (phi + phi_adjustment) % 360
                
                # 降低速度以减少意外碰撞
                velocity = min(velocity, 4.0)
                warning_msg += " (避让黑8)"
            
            # 白球落袋风险高的处理
            if cue_pocket_risk > 0.5:
                # 大幅降低速度
                velocity = min(velocity, self.risky_velocity_limit)
                # 减小仰角，避免跟进
                theta = min(theta, 15.0)
                # 施加后旋（负b值）帮助白球停止
                b_offset = -0.15
                warning_msg += f" (白球风险={cue_pocket_risk:.2f})"
            
            # 步骤7: 预判首撞球验证
            predicted_first, _ = self.predict_first_contact_ball(cue_pos, phi, balls)
            
            if predicted_first and predicted_first != best_target_id:
                # 预判首撞不是目标球，尝试微调角度
                if predicted_first in my_targets:
                    # 首撞是己方其他球，可以接受但不理想
                    warning_msg += f" (预判首撞={predicted_first})"
                else:
                    # 首撞是对方球或黑8，需要调整
                    # 尝试小幅度调整角度
                    for angle_offset in [1.5, -1.5, 3.0, -3.0, 5.0, -5.0]:
                        test_phi = (phi + angle_offset) % 360
                        test_first, _ = self.predict_first_contact_ball(cue_pos, test_phi, balls)
                        if test_first == best_target_id:
                            phi = test_phi
                            warning_msg += f" (角度调整{angle_offset:+.1f}°)"
                            break
                    else:
                        # 无法调整，使用原角度但降低速度
                        velocity = min(velocity, 2.5)
                        warning_msg += f" (⚠️首撞风险={predicted_first})"
            
            # 最终边界检查
            velocity = np.clip(velocity, 0.5, 8.0)
            phi = phi % 360
            theta = np.clip(theta, 0, 90)
            a_offset = np.clip(a_offset, -0.5, 0.5)
            b_offset = np.clip(b_offset, -0.5, 0.5)
            
            # 构建动作
            action = {
                'V0': float(velocity),
                'phi': float(phi),
                'theta': float(theta),
                'a': float(a_offset),
                'b': float(b_offset)
            }
            
            print(f"[GeometricAgent] 目标: {best_target_id}→{best_pocket_id}, "
                  f"距离={distance:.3f}m, 切角={cut_angle:.1f}°{warning_msg}")
            print(f"[GeometricAgent] 决策: V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            
            return action
            
        except Exception as e:
            print(f"[GeometricAgent] 决策时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class MCTSAgent(Agent):
    """基于蒙特卡洛树搜索(MCTS)的Agent
    
    核心思想：
    - 使用几何学生成候选动作（目标球+球袋组合）
    - 对每个候选动作进行多次蒙特卡洛模拟
    - 使用UCB公式平衡探索(exploration)与利用(exploitation)
    - 选择平均奖励最高的动作
    
    特点：
    - 结合几何先验知识缩小搜索空间
    - 通过物理仿真评估动作质量
    - 比纯贝叶斯优化更注重采样多样性
    """
    
    def __init__(self):
        super().__init__()
        
        # 导入几何工具
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
        
        # MCTS核心参数
        self.num_simulations = 30          # 总模拟次数
        self.exploration_weight = 1.41     # UCB探索权重 (sqrt(2))
        self.num_candidates_per_target = 5 # 每个目标生成的候选动作数
        
        # 动作参数扰动范围
        self.phi_noise_range = 5.0         # 角度扰动范围 ±5°
        self.V0_noise_range = 1.0          # 速度扰动范围 ±1.0 m/s
        self.theta_noise_range = 10.0      # 仰角扰动范围 ±10°
        self.offset_noise_range = 0.15     # 偏移扰动范围 ±0.15
        
        # 打黑8的保守参数
        self.eight_ball_V0_max = 4.5
        self.eight_ball_theta_max = 20
        
        print("MCTSAgent (Monte Carlo Tree Search) 已初始化。")
    
    def print_config(self):
        """打印所有重要超参数"""
        print("\n" + "="*50)
        print("[MCTSAgent 超参数配置]")
        print(f"  MCTS核心参数:")
        print(f"    - num_simulations (模拟次数): {self.num_simulations}")
        print(f"    - exploration_weight (UCB探索权重): {self.exploration_weight}")
        print(f"    - num_candidates_per_target (每目标候选数): {self.num_candidates_per_target}")
        print(f"  动作扰动范围:")
        print(f"    - phi_noise_range: ±{self.phi_noise_range}°")
        print(f"    - V0_noise_range: ±{self.V0_noise_range} m/s")
        print(f"    - theta_noise_range: ±{self.theta_noise_range}°")
        print(f"    - offset_noise_range: ±{self.offset_noise_range}")
        print(f"  打黑8特殊参数:")
        print(f"    - eight_ball_V0_max: {self.eight_ball_V0_max} m/s")
        print(f"    - eight_ball_theta_max: {self.eight_ball_theta_max}°")
        print(f"  特点: 几何先验 + MCTS搜索 + 物理仿真评估")
        print("="*50)
    
    def _generate_candidate_actions(self, balls, my_targets, table, is_shooting_eight=False):
        """
        使用几何学生成候选动作列表
        
        思路：
        1. 遍历所有未进袋的目标球
        2. 对每个目标球，尝试每个球袋
        3. 计算几何瞄准参数
        4. 对基础参数进行随机扰动，生成多个变体
        
        返回：
            list of dict: 候选动作列表
        """
        candidates = []
        cue_pos = balls['cue'].state.rvw[0]
        
        for target_id in my_targets:
            if balls[target_id].state.s == 4:  # 已进袋
                continue
            
            target_pos = balls[target_id].state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 计算难度
                difficulty = self.calculate_shot_difficulty(
                    cue_pos, target_pos, pocket_pos, balls,
                    target_id=target_id, my_targets=my_targets
                )
                
                # 检查黑8是否在路径上
                eight_in_path, _ = self.check_eight_ball_in_path(cue_pos, target_pos, balls)
                if eight_in_path and not is_shooting_eight:
                    difficulty *= 10.0
                
                # 计算几何瞄准参数
                aim_point = self.calculate_aim_point_for_pocket(cue_pos, target_pos, pocket_pos)
                if aim_point is None:
                    continue
                
                base_phi = self.calculate_angle_to_aim_point(cue_pos, aim_point)
                distance = self.calculate_distance(cue_pos, target_pos)
                base_V0 = self.calculate_recommended_velocity(distance)
                
                if is_shooting_eight:
                    base_V0 = min(base_V0, self.eight_ball_V0_max)
                
                # 生成多个扰动变体
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
        """模拟一个动作并返回奖励"""
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
        """计算UCB分数: avg_reward + c * sqrt(ln(N) / n)"""
        if visit_count == 0:
            return float('inf')
        exploitation = avg_reward
        exploration = self.exploration_weight * math.sqrt(math.log(total_visits + 1) / visit_count)
        return exploitation + exploration
    
    def decision(self, balls=None, my_targets=None, table=None):
        """MCTS决策主流程"""
        if balls is None or my_targets is None or table is None:
            print("[MCTSAgent] 缺少必要信息，使用随机动作。")
            return self._random_action()
        
        try:
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            is_shooting_eight = False
            if len(remaining_own) == 0:
                my_targets = ["8"]
                is_shooting_eight = True
                print("[MCTSAgent] 目标球已清空，切换到黑8。⚠️ 进入谨慎模式！")
            
            # 步骤1: 生成候选动作
            candidates = self._generate_candidate_actions(balls, my_targets, table, is_shooting_eight)
            
            if len(candidates) == 0:
                print("[MCTSAgent] 无候选动作，使用随机动作。")
                return self._random_action()
            
            print(f"[MCTSAgent] 生成 {len(candidates)} 个候选动作，开始MCTS搜索...")
            
            # 初始化统计
            visit_counts = [0] * len(candidates)
            total_rewards = [0.0] * len(candidates)
            
            # 步骤2: MCTS迭代
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
            
            # 步骤3: 选择最佳动作
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
                print(f"[MCTSAgent] MCTS最佳奖励较低 ({best_avg_reward:.2f})，回退到最简单目标。")
                best_candidate = candidates[0]
                best_action = best_candidate['action']
            
            print(f"[MCTSAgent] 目标: {best_candidate['target_id']}→{best_candidate['pocket_id']}, "
                  f"平均奖励={best_avg_reward:.2f}, 访问次数={best_visits}")
            print(f"[MCTSAgent] 决策: V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, "
                  f"θ={best_action['theta']:.2f}, a={best_action['a']:.3f}, b={best_action['b']:.3f}")
            
            return {
                'V0': float(best_action['V0']),
                'phi': float(best_action['phi']) % 360,
                'theta': float(best_action['theta']),
                'a': float(best_action['a']),
                'b': float(best_action['b'])
            }
            
        except Exception as e:
            print(f"[MCTSAgent] 决策时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()


class EnsembleVotingAgent(Agent):
    """集体投票Agent（Ensemble Voting Agent）
    
    核心思想：
    - 调用多个子Agent（NewAgent, MCTSAgent）分别生成候选动作
    - 对每个候选动作进行物理仿真评估
    - 选择得分最高的动作作为最终决策
    
    优势：
    - 结合多种决策策略的优点
    - 通过模拟验证减少失误
    - 更加稳健的决策
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始化子Agent
        self.new_agent = Enhanced_Bayes_Agent()
        self.mcts_agent = MCTSAgent()
        
        # 评估参数
        self.num_eval_simulations = 3  # 每个动作评估的模拟次数
        
        print("EnsembleVotingAgent (Ensemble Voting) 已初始化。")
    
    def print_config(self):
        """打印所有重要超参数"""
        print("\n" + "="*50)
        print("[EnsembleVotingAgent 超参数配置]")
        print(f"  子Agent:")
        print(f"    - NewAgent (几何+贝叶斯优化)")
        print(f"    - MCTSAgent (几何+MCTS搜索)")
        print(f"  评估参数:")
        print(f"    - num_eval_simulations (每动作模拟次数): {self.num_eval_simulations}")
        print(f"  特点: 多策略集成投票，选择最高分动作")
        print("="*50)
        print("\n--- NewAgent 配置 ---")
        self.new_agent.print_config()
        print("\n--- MCTSAgent 配置 ---")
        self.mcts_agent.print_config()
    
    def _evaluate_action(self, action, balls, table, my_targets, last_state_snapshot):
        """评估单个动作的得分
        
        通过多次模拟取平均分，减少随机性影响
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
        """集体投票决策
        
        步骤：
        1. 调用NewAgent获取候选动作
        2. 调用MCTSAgent获取候选动作
        3. 对每个候选动作进行模拟评估
        4. 选择得分最高的动作
        """
        if balls is None or my_targets is None or table is None:
            print("[EnsembleVotingAgent] 缺少必要信息，使用随机动作。")
            return self._random_action()
        
        try:
            # 保存状态快照
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
            
            # 检查目标球是否已清空
            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[EnsembleVotingAgent] 目标球已清空，切换到黑8。")
            
            # 步骤1: 收集候选动作
            candidate_actions = []
            
            # 从NewAgent获取动作
            print("[EnsembleVotingAgent] 调用 NewAgent...")
            try:
                new_agent_action = self.new_agent.decision(balls, my_targets, table)
                if new_agent_action:
                    candidate_actions.append({
                        'action': new_agent_action,
                        'source': 'NewAgent'
                    })
            except Exception as e:
                print(f"[EnsembleVotingAgent] NewAgent决策失败: {e}")
            
            # 从MCTSAgent获取动作
            print("[EnsembleVotingAgent] 调用 MCTSAgent...")
            try:
                mcts_agent_action = self.mcts_agent.decision(balls, my_targets, table)
                if mcts_agent_action:
                    candidate_actions.append({
                        'action': mcts_agent_action,
                        'source': 'MCTSAgent'
                    })
            except Exception as e:
                print(f"[EnsembleVotingAgent] MCTSAgent决策失败: {e}")
            
            if len(candidate_actions) == 0:
                print("[EnsembleVotingAgent] 无候选动作，使用随机动作。")
                return self._random_action()
            
            # 步骤2: 评估每个候选动作
            print(f"[EnsembleVotingAgent] 评估 {len(candidate_actions)} 个候选动作...")
            
            best_action = None
            best_score = float('-inf')
            best_source = None
            
            for candidate in candidate_actions:
                action = candidate['action']
                source = candidate['source']
                
                score = self._evaluate_action(
                    action, balls, table, my_targets, last_state_snapshot
                )
                
                print(f"[EnsembleVotingAgent] {source} 动作得分: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_source = source
            
            # 步骤3: 返回最佳动作
            print(f"[EnsembleVotingAgent] 最终选择: {best_source} (得分: {best_score:.2f})")
            print(f"[EnsembleVotingAgent] 决策: V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, "
                  f"θ={best_action['theta']:.2f}, a={best_action['a']:.3f}, b={best_action['b']:.3f}")
            
            return best_action
            
        except Exception as e:
            print(f"[EnsembleVotingAgent] 决策时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

