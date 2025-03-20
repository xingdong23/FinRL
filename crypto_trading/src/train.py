import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import tensorboard
from datetime import datetime

from data_processor import CryptoDataProcessor
from env_crypto_trading import CryptoTradingEnv

class TensorboardCallback(BaseCallback):
    """自定义回调，用于记录训练过程中的指标"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])
            except BaseException as inner_error:
                self.logger.record(key="train/reward", value=None)
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

def train(
    symbol_list: list,
    start_date: str,
    end_date: str,
    timeframe: str = '1h',
    initial_balance: float = 10000,
    transaction_cost: float = 0.001,
    reward_scaling: float = 1e-4,
    total_timesteps: int = 100000,
    learning_rate: float = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    verbose: int = 1,
    tensorboard_log: str = "./tensorboard_logs/",
    model_save_path: str = "./models/"
):
    """
    训练加密货币交易智能体
    
    Args:
        symbol_list: 交易对列表
        start_date: 开始日期
        end_date: 结束日期
        timeframe: 时间周期
        initial_balance: 初始资金
        transaction_cost: 交易成本
        reward_scaling: 奖励缩放因子
        total_timesteps: 总训练步数
        learning_rate: 学习率
        n_steps: 每次更新的步数
        batch_size: 批次大小
        n_epochs: 每次更新的训练轮数
        gamma: 折扣因子
        gae_lambda: GAE参数
        clip_range: PPO裁剪范围
        ent_coef: 熵系数
        verbose: 打印级别
        tensorboard_log: TensorBoard日志目录
        model_save_path: 模型保存目录
    """
    # 创建必要的目录
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # 准备数据
    print("准备数据...")
    data_processor = CryptoDataProcessor()
    train_data, test_data = data_processor.prepare_data(
        symbol_list=symbol_list,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # 定义技术指标列表
    tech_indicator_list = [
        'MA5', 'MA10', 'MA20',
        'RSI', 'MACD', 'Signal_Line',
        'BB_middle', 'BB_upper', 'BB_lower'
    ]
    
    # 创建训练环境
    print("创建训练环境...")
    train_env = CryptoTradingEnv(
        df=train_data,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        reward_scaling=reward_scaling,
        state_space=1 + 2*len(symbol_list) + len(tech_indicator_list),
        action_space=3,
        tech_indicator_list=tech_indicator_list,
        make_plots=False,
        print_verbosity=10
    )
    
    # 包装环境
    train_env = DummyVecEnv([lambda: train_env])
    
    # 设置模型参数
    model_kwargs = {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "verbose": verbose
    }
    
    # 创建模型
    print("创建模型...")
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=tensorboard_log,
        **model_kwargs
    )
    
    # 训练模型
    print("开始训练...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback(),
        progress_bar=True
    )
    
    # 保存模型
    print("保存模型...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{model_save_path}/ppo_crypto_trading_{timestamp}")
    
    return model, train_env

if __name__ == "__main__":
    # 设置训练参数
    symbol_list = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    # 训练模型
    model, env = train(
        symbol_list=symbol_list,
        start_date=start_date,
        end_date=end_date,
        timeframe='1h',
        initial_balance=10000,
        total_timesteps=100000
    ) 