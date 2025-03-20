# 加密货币交易智能体

这是一个基于强化学习的加密货币交易智能体项目，使用FinRL框架实现。该项目实现了以下功能：

1. 数据获取和预处理
2. 技术指标计算
3. 交易环境模拟
4. 强化学习模型训练
5. 模型评估和回测
6. 结果可视化和分析

## 环境要求

- Python 3.8+
- 依赖包：
  - ccxt
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - stable-baselines3
  - gym
  - tensorboard

## 安装

1. 克隆项目：
```bash
git clone <repository_url>
cd crypto_trading
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
crypto_trading/
├── data/               # 数据存储目录
├── models/            # 模型保存目录
├── results/           # 结果保存目录
├── src/               # 源代码
│   ├── data_processor.py    # 数据处理模块
│   ├── env_crypto_trading.py # 交易环境
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   └── main.py             # 主程序
└── notebooks/         # Jupyter notebooks
```

## 使用方法

### 训练模型

```bash
python src/main.py --mode train \
    --symbols BTC/USDT ETH/USDT BNB/USDT \
    --start_date 2020-01-01 \
    --end_date 2023-12-31 \
    --timeframe 1h \
    --initial_balance 10000 \
    --total_timesteps 100000
```

### 评估模型

```bash
python src/main.py --mode evaluate \
    --model_path ./models/ppo_crypto_trading_20240315_120000 \
    --symbols BTC/USDT ETH/USDT BNB/USDT \
    --start_date 2023-01-01 \
    --end_date 2023-12-31
```

### 训练并评估

```bash
python src/main.py --mode both \
    --symbols BTC/USDT ETH/USDT BNB/USDT \
    --start_date 2020-01-01 \
    --end_date 2023-12-31
```

## 参数说明

- `--mode`: 运行模式，可选值：train（训练）, evaluate（评估）, both（训练和评估）
- `--symbols`: 交易对列表
- `--start_date`: 训练数据开始日期
- `--end_date`: 训练数据结束日期
- `--timeframe`: 时间周期
- `--initial_balance`: 初始资金
- `--total_timesteps`: 总训练步数
- `--model_path`: 模型文件路径（用于评估模式）

## 技术指标

项目使用以下技术指标：

1. 移动平均线（MA5, MA10, MA20）
2. 相对强弱指标（RSI）
3. 移动平均收敛散度（MACD）
4. 布林带（BB）

## 结果分析

训练和评估结果将保存在 `results` 目录下，包括：

1. 训练配置（train_config_{timestamp}.json）
2. 评估结果（evaluation_results_{timestamp}.json）
3. TensorBoard日志（tensorboard_logs/）
4. 交易图表（如果启用render选项）

## 注意事项

1. 确保有足够的磁盘空间存储数据和模型
2. 训练过程可能需要较长时间，建议使用GPU加速
3. 回测结果仅供参考，实际交易可能受到滑点、手续费等因素影响
4. 建议在实盘交易前进行充分的回测和风险评估

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License 