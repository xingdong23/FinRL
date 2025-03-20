import os
import argparse
from datetime import datetime
import json

from train import train
from evaluate import evaluate

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='加密货币交易智能体训练和评估')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'both'],
                      help='运行模式：train（训练）, evaluate（评估）, both（训练和评估）')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                      help='交易对列表')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                      help='训练数据开始日期')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                      help='训练数据结束日期')
    parser.add_argument('--timeframe', type=str, default='1h',
                      help='时间周期')
    parser.add_argument('--initial_balance', type=float, default=10000,
                      help='初始资金')
    parser.add_argument('--total_timesteps', type=int, default=100000,
                      help='总训练步数')
    parser.add_argument('--model_path', type=str, default=None,
                      help='模型文件路径（用于评估模式）')
    
    args = parser.parse_args()
    
    # 创建结果目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 训练模式
    if args.mode in ['train', 'both']:
        print("\n开始训练...")
        model, env = train(
            symbol_list=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            initial_balance=args.initial_balance,
            total_timesteps=args.total_timesteps
        )
        
        # 保存训练配置
        config = {
            'mode': 'train',
            'symbols': args.symbols,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'timeframe': args.timeframe,
            'initial_balance': args.initial_balance,
            'total_timesteps': args.total_timesteps,
            'timestamp': timestamp
        }
        
        with open(f'{results_dir}/train_config_{timestamp}.json', 'w') as f:
            json.dump(config, f, indent=4)
    
    # 评估模式
    if args.mode in ['evaluate', 'both']:
        print("\n开始评估...")
        
        # 确定模型路径
        model_path = args.model_path
        if args.mode == 'both':
            model_path = f'./models/ppo_crypto_trading_{timestamp}'
        
        if not model_path:
            raise ValueError("评估模式需要提供模型路径（--model_path）")
        
        # 运行评估
        results = evaluate(
            model_path=model_path,
            symbol_list=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            initial_balance=args.initial_balance,
            render=True
        )
        
        # 保存评估结果
        results['timestamp'] = timestamp
        with open(f'{results_dir}/evaluation_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n评估结果已保存到: {results_dir}/evaluation_results_{timestamp}.json")

if __name__ == "__main__":
    main() 