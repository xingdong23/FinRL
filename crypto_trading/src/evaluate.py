import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from datetime import datetime

from data_processor import CryptoDataProcessor
from env_crypto_trading import CryptoTradingEnv

def evaluate(
    model_path: str,
    symbol_list: list,
    start_date: str,
    end_date: str,
    timeframe: str = '1h',
    initial_balance: float = 10000,
    transaction_cost: float = 0.001,
    reward_scaling: float = 1e-4,
    render: bool = True
):
    """
    评估训练好的加密货币交易智能体
    
    Args:
        model_path: 模型文件路径
        symbol_list: 交易对列表
        start_date: 开始日期
        end_date: 结束日期
        timeframe: 时间周期
        initial_balance: 初始资金
        transaction_cost: 交易成本
        reward_scaling: 奖励缩放因子
        render: 是否显示交易图表
    """
    # 准备数据
    print("准备数据...")
    data_processor = CryptoDataProcessor()
    _, test_data = data_processor.prepare_data(
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
    
    # 创建测试环境
    print("创建测试环境...")
    test_env = CryptoTradingEnv(
        df=test_data,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        reward_scaling=reward_scaling,
        state_space=1 + 2*len(symbol_list) + len(tech_indicator_list),
        action_space=3,
        tech_indicator_list=tech_indicator_list,
        make_plots=render,
        print_verbosity=10
    )
    
    # 包装环境
    test_env = DummyVecEnv([lambda: test_env])
    
    # 加载模型
    print("加载模型...")
    model = PPO.load(model_path)
    
    # 评估模型
    print("开始评估...")
    obs = test_env.reset()
    done = False
    total_reward = 0
    episode_rewards = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward[0]
        episode_rewards.append(total_reward)
    
    # 计算评估指标
    final_balance = test_env.envs[0].total_asset
    returns = (final_balance - initial_balance) / initial_balance * 100
    sharpe_ratio = np.mean(episode_rewards) / np.std(episode_rewards) if len(episode_rewards) > 1 else 0
    
    print("\n评估结果:")
    print(f"初始资金: ${initial_balance:,.2f}")
    print(f"最终资金: ${final_balance:,.2f}")
    print(f"总收益率: {returns:.2f}%")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    
    # 绘制收益曲线
    if render:
        plt.figure(figsize=(12, 6))
        plt.plot(episode_rewards)
        plt.title('训练过程中的累积奖励')
        plt.xlabel('步数')
        plt.ylabel('累积奖励')
        plt.grid(True)
        plt.show()
    
    return {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'returns': returns,
        'sharpe_ratio': sharpe_ratio,
        'episode_rewards': episode_rewards
    }

if __name__ == "__main__":
    # 设置评估参数
    model_path = "./models/ppo_crypto_trading_20240315_120000"  # 替换为实际的模型路径
    symbol_list = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # 评估模型
    results = evaluate(
        model_path=model_path,
        symbol_list=symbol_list,
        start_date=start_date,
        end_date=end_date,
        timeframe='1h',
        initial_balance=10000,
        render=True
    ) 