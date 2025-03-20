import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple

class CryptoTradingEnv(gym.Env):
    """加密货币交易环境"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000,
        transaction_cost: float = 0.001,
        reward_scaling: float = 1e-4,
        state_space: int = 30,
        action_space: int = 3,  # 买入、卖出、持有
        tech_indicator_list: List[str] = None,
        turbulence_threshold: float = None,
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: List = None,
        model_name: str = "",
        mode: str = "",
        iteration: str = ""
    ):
        """
        初始化加密货币交易环境
        
        Args:
            df: 包含价格和技术指标的数据
            initial_balance: 初始资金
            transaction_cost: 交易成本
            reward_scaling: 奖励缩放因子
            state_space: 状态空间维度
            action_space: 动作空间维度
            tech_indicator_list: 技术指标列表
            turbulence_threshold: 市场波动阈值
            make_plots: 是否绘制图表
            print_verbosity: 打印频率
            day: 当前交易日
            initial: 是否为初始状态
            previous_state: 前一个状态
            model_name: 模型名称
            mode: 模式（训练/测试）
            iteration: 迭代次数
        """
        super().__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.day = day
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        
        # 定义动作空间（买入、卖出、持有）
        self.action_space = spaces.Discrete(action_space)
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_space,),
            dtype=np.float32
        )
        
        # 初始化状态
        self.state = self._initiate_state()
        
        # 初始化其他变量
        self.terminal = False
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        
        # 记录交易历史
        self.asset_memory = [self.initial_balance]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        
        # 设置随机种子
        self._seed()
        
    def _seed(self, seed=None):
        """设置随机种子"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _initiate_state(self) -> np.ndarray:
        """初始化状态"""
        if self.initial:
            # 获取当前时间步的数据
            self.data = self.df.loc[self.day, :]
            
            # 构建状态向量
            state = []
            
            # 添加账户余额
            state.append(self.initial_balance)
            
            # 添加持仓信息
            for symbol in self.df['symbol'].unique():
                state.append(0)  # 持仓数量
                state.append(0)  # 持仓成本
                
            # 添加技术指标
            for indicator in self.tech_indicator_list:
                state.append(self.data[indicator])
                
            return np.array(state)
        else:
            return self.previous_state
    
    def _get_date(self) -> str:
        """获取当前日期"""
        if len(self.df) > self.day:
            return self.df.index[self.day]
        else:
            return self.df.index[-1]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步交易
        
        Args:
            action: 交易动作（0: 买入, 1: 卖出, 2: 持有）
            
        Returns:
            状态、奖励、是否结束、是否截断、信息字典
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            # 计算最终资产
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1:self.stock_dim + 1]) *
                np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1])
            )
            
            # 计算总回报
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            
            # 计算夏普比率
            if df_total_value['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            else:
                sharpe = 0
                
            # 打印结果
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {self.reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")
            
            return self.state, self.reward, self.terminal, False, {}
            
        else:
            # 获取当前价格
            current_price = self.data['close']
            
            # 执行交易
            if action == 0:  # 买入
                # 计算买入数量
                buy_amount = self.state[0] * 0.95  # 使用95%的资金买入
                shares = buy_amount / current_price
                
                # 计算交易成本
                cost = buy_amount * self.transaction_cost
                
                # 更新状态
                self.state[0] -= (buy_amount + cost)
                self.state[1] += shares
                self.state[2] = current_price
                
                self.cost += cost
                self.trades += 1
                
            elif action == 1:  # 卖出
                # 计算卖出数量
                shares = self.state[1]
                sell_amount = shares * current_price
                
                # 计算交易成本
                cost = sell_amount * self.transaction_cost
                
                # 更新状态
                self.state[0] += (sell_amount - cost)
                self.state[1] = 0
                self.state[2] = 0
                
                self.cost += cost
                self.trades += 1
            
            # 计算奖励（收益率）
            total_asset = self.state[0] + self.state[1] * current_price
            self.reward = (total_asset - self.asset_memory[-1]) / self.asset_memory[-1]
            
            # 更新记忆
            self.asset_memory.append(total_asset)
            self.rewards_memory.append(self.reward)
            self.actions_memory.append(action)
            self.state_memory.append(self.state)
            self.date_memory.append(self._get_date())
            
            # 更新到下一个时间步
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            return self.state, self.reward, self.terminal, False, {}
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            
        Returns:
            初始状态和信息字典
        """
        self._seed(seed)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.initial = True
        self.state = self._initiate_state()
        self.terminal = False
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode += 1
        
        # 重置记忆
        self.asset_memory = [self.initial_balance]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        
        return self.state, {}
    
    def render(self, mode='human'):
        """渲染环境"""
        pass
    
    def _make_plot(self):
        """绘制交易图表"""
        pass 