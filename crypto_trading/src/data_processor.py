import ccxt
import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta
import time
import requests

class CryptoDataProcessor:
    def __init__(self, exchange_name: str = 'gateio'):
        """
        初始化加密货币数据处理器
        
        Args:
            exchange_name: 交易所名称，默认为'gateio'
        """
        self.exchange_name = exchange_name
        if exchange_name.lower() == 'gateio':
            # 对于Gate.io，我们将使用其REST API直接获取历史数据
            self.base_url = "https://api.gateio.ws/api/v4"
        else:
            # 对于其他交易所，使用ccxt库
            self.exchange = getattr(ccxt, exchange_name)({
                'timeout': 30000,  # 增加超时时间到30秒
                'enableRateLimit': True,  # 启用请求频率限制
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
        
    def download_data(self, 
                     symbol_list: List[str], 
                     start_date: str, 
                     end_date: str, 
                     timeframe: str = '1h') -> pd.DataFrame:
        """
        下载加密货币历史数据
        
        Args:
            symbol_list: 交易对列表，如 ['BTC/USDT', 'ETH/USDT']
            start_date: 开始日期，格式：'YYYY-MM-DD'
            end_date: 结束日期，格式：'YYYY-MM-DD'
            timeframe: 时间周期，如 '1h', '4h', '1d'
            
        Returns:
            包含历史数据的DataFrame
        """
        all_data = []
        
        # 转换日期格式
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        
        if self.exchange_name.lower() == 'gateio':
            # 使用Gate.io的REST API
            for symbol in symbol_list:
                print(f"Downloading {symbol} data using Gate.io API...")
                
                # Gate.io需要将BTC/USDT格式转换为BTC_USDT格式
                formatted_symbol = symbol.replace('/', '_').upper()
                
                # 时间间隔映射
                interval_map = {
                    '1m': '1m',
                    '5m': '5m',
                    '15m': '15m',
                    '30m': '30m',
                    '1h': '1h',
                    '4h': '4h',
                    '8h': '8h',
                    '1d': '1d',
                    '7d': '7d'
                }
                
                gate_interval = interval_map.get(timeframe, '1h')
                
                # 添加重试机制
                max_retries = 3
                retry_delay = 5  # 秒
                
                # 计算每个K线的时间跨度(秒)
                if gate_interval == '1m':
                    seconds_per_candle = 60
                elif gate_interval == '5m':
                    seconds_per_candle = 300
                elif gate_interval == '15m':
                    seconds_per_candle = 900
                elif gate_interval == '30m':
                    seconds_per_candle = 1800
                elif gate_interval == '1h':
                    seconds_per_candle = 3600
                elif gate_interval == '4h':
                    seconds_per_candle = 14400
                elif gate_interval == '8h':
                    seconds_per_candle = 28800
                elif gate_interval == '1d':
                    seconds_per_candle = 86400
                elif gate_interval == '7d':
                    seconds_per_candle = 604800
                else:
                    seconds_per_candle = 3600  # 默认1小时
                
                # 每次最多请求100条数据，以避免超出API限制
                # Gate.io文档说明，如果使用from和to参数，不应该使用limit参数
                max_points = 100
                batch_duration = seconds_per_candle * max_points
                
                # 批量获取数据
                current_start = start_timestamp
                symbol_data = []
                
                while current_start < end_timestamp:
                    for attempt in range(max_retries):
                        try:
                            # 确定当前批次的结束时间
                            current_end = min(current_start + batch_duration, end_timestamp)
                            
                            # 构建URL
                            url = f"{self.base_url}/spot/candlesticks"
                            
                            # 准备请求参数
                            params = {
                                'currency_pair': formatted_symbol,
                                'interval': gate_interval,
                                'from': current_start,
                                'to': current_end
                            }
                            
                            # 添加Accept头
                            headers = {
                                'Accept': 'application/json'
                            }
                            
                            # 发起请求
                            response = requests.get(url, params=params, headers=headers)
                            response.raise_for_status()  # 如果请求失败，抛出异常
                            data = response.json()
                            
                            if data:
                                # Gate.io返回的数据格式为:
                                # ["timestamp", "volume", "close", "high", "low", "open", "?"]
                                temp_df = pd.DataFrame(data)
                                
                                if not temp_df.empty:
                                    # 确保有足够的列
                                    if len(temp_df.columns) >= 6:
                                        # 重命名列
                                        temp_df.columns = ['timestamp', 'volume', 'close', 'high', 'low', 'open'] + (
                                            ['extra'] * (len(temp_df.columns) - 6) if len(temp_df.columns) > 6 else []
                                        )
                                        
                                        # 转换时间戳为datetime
                                        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'].astype(int), unit='s')
                                        
                                        # 转换所有价格列为浮点数
                                        for col in ['open', 'high', 'low', 'close', 'volume']:
                                            temp_df[col] = temp_df[col].astype(float)
                                        
                                        # 重新排序列以匹配我们的标准格式
                                        temp_df = temp_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                                        
                                        # 添加交易对信息
                                        temp_df['symbol'] = symbol
                                        
                                        symbol_data.append(temp_df)
                            
                            # 更新起始时间为当前批次的结束时间
                            current_start = current_end
                            
                            # 成功获取数据，跳出重试循环
                            break
                            
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Attempt {attempt + 1} failed: {str(e)}")
                                print(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                print(f"Failed to download {symbol} data after {max_retries} attempts")
                                print(f"Error: {str(e)}")
                                # 不是所有请求都必须成功，继续下一个时间段
                                current_start = current_end
                                break
                    
                    # 添加延迟以避免触发API限制
                    time.sleep(1)
                
                # 合并该交易对的所有批次数据
                if symbol_data:
                    combined_symbol_df = pd.concat(symbol_data, ignore_index=True)
                    all_data.append(combined_symbol_df)
        else:
            # 使用ccxt库获取数据
            for symbol in symbol_list:
                print(f"Downloading {symbol} data using ccxt...")
                
                # 添加重试机制
                max_retries = 3
                retry_delay = 5  # 秒
                
                for attempt in range(max_retries):
                    try:
                        # 获取历史数据
                        ohlcv = self.exchange.fetch_ohlcv(
                            symbol,
                            timeframe=timeframe,
                            since=start_timestamp * 1000,  # ccxt需要毫秒时间戳
                            limit=1000
                        )
                        
                        # 转换为DataFrame
                        df = pd.DataFrame(
                            ohlcv,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        
                        # 添加交易对信息
                        df['symbol'] = symbol
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        all_data.append(df)
                        break  # 如果成功，跳出重试循环
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed: {str(e)}")
                            print(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"Failed to download {symbol} data after {max_retries} attempts")
                            raise
                
                # 添加延迟以避免触发API限制
                time.sleep(1)
        
        # 合并所有数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp')
            return combined_df
        else:
            raise Exception("No data was downloaded")
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 按交易对分组处理
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # 计算移动平均线
            symbol_data['MA5'] = symbol_data['close'].rolling(window=5).mean()
            symbol_data['MA10'] = symbol_data['close'].rolling(window=10).mean()
            symbol_data['MA20'] = symbol_data['close'].rolling(window=20).mean()
            
            # 计算RSI
            delta = symbol_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            symbol_data['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = symbol_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = symbol_data['close'].ewm(span=26, adjust=False).mean()
            symbol_data['MACD'] = exp1 - exp2
            symbol_data['Signal_Line'] = symbol_data['MACD'].ewm(span=9, adjust=False).mean()
            
            # 计算布林带
            symbol_data['BB_middle'] = symbol_data['close'].rolling(window=20).mean()
            bb_std = symbol_data['close'].rolling(window=20).std()
            symbol_data['BB_upper'] = symbol_data['BB_middle'] + (bb_std * 2)
            symbol_data['BB_lower'] = symbol_data['BB_middle'] - (bb_std * 2)
            
            # 更新原始DataFrame
            df.loc[df['symbol'] == symbol] = symbol_data
            
        return df
    
    def prepare_data(self, 
                    symbol_list: List[str], 
                    start_date: str, 
                    end_date: str, 
                    timeframe: str = '1h') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备训练和测试数据
        
        Args:
            symbol_list: 交易对列表
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间周期
            
        Returns:
            训练数据和测试数据的元组
        """
        # 下载数据
        df = self.download_data(symbol_list, start_date, end_date, timeframe)
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 分割训练集和测试集（使用最后20%的数据作为测试集）
        train_size = int(len(df) * 0.8)
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        return train_data, test_data 