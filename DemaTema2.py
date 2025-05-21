import backtrader as bt
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Define default ticker symbol
default_ticker = 'CHILE.SN'

# Strategy: DEMA Crossover with ATR filter and SMA200 trend filter
class DemaCrossStrategy(bt.Strategy):
    params = (
        ('fast_dema', 10),    # Optimized parameters
        ('slow_dema', 22),
        ('order_percentage', 0.95),
        ('atr_period', 14),
        ('atr_sma_period', 14),
        ('trend_period', 200),
        ('ticker', default_ticker)
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        # Indicators
        self.dema_fast = bt.indicators.DEMA(self.data_close, period=self.params.fast_dema)
        self.dema_slow = bt.indicators.DEMA(self.data_close, period=self.params.slow_dema)
        self.crossover = bt.indicators.CrossOver(self.dema_fast, self.dema_slow)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
        self.atr_sma = bt.indicators.SMA(self.atr, period=self.params.atr_sma_period)
        self.sma_trend = bt.indicators.SMA(self.data_close, period=self.params.trend_period)
        # Signal storage for plotting
        self.buy_signals = []  # list of (datetime, price)
        self.sell_signals = []
        self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} [{self.params.ticker}] - {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status is order.Completed:
            dt = self.datas[0].datetime.datetime(0)
            price = order.executed.price
            side = 'BUY' if order.isbuy() else 'SELL'
            self.log(f'{side} EXECUTED | Price: {price:.2f} Size: {order.executed.size}')
            # record for plotting
            if order.isbuy():
                self.buy_signals.append((dt, price))
            else:
                self.sell_signals.append((dt, price))
        self.order = None

    def next(self):
        # Print signals in real-time for live execution
        if not self.position and self.crossover[0] > 0 and \
           self.data_close[0] > self.sma_trend[0] and \
           self.atr[0] > self.atr_sma[0]:
            self.log(f'BUY SIGNAL BAR | Close: {self.data_close[0]:.2f}, ATR: {self.atr[0]:.4f}, SMA200: {self.sma_trend[0]:.2f}')
            cash = self.broker.get_cash()
            size = int((cash * self.params.order_percentage) / self.data_close[0])
            if size > 0:
                self.order = self.buy(size=size)
        elif self.position and self.crossover[0] < 0 and self.atr[0] > self.atr_sma[0]:
            self.log(f'SELL SIGNAL BAR | Close: {self.data_close[0]:.2f}, ATR: {self.atr[0]:.4f}')
            self.order = self.close()

# Full backtest with plotting of buy/sell signals
def full_backtest(df):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(DemaCrossStrategy)
    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed)
    cerebro.broker.set_cash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    strat = cerebro.run()[0]
    # Extract results
    sr = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    ann_sr = sr * (252**0.5) if sr else None
    dd = strat.analyzers.drawdown.get_analysis().max.drawdown
    trades = strat.analyzers.trades.get_analysis()
    total = trades.total.total or 0
    won = trades.won.total or 0
    win_rate = won/total*100 if total else 0
    print(f"Final Value: {cerebro.broker.getvalue():.2f}, Sharpe: {ann_sr:.2f}, Drawdown: {dd:.2f}%, Trades: {total}, Win Rate: {win_rate:.2f}%")

    # Prepare plot
    plt.figure(figsize=(14,8))
    plt.plot(df.index, df['close'], label='Close Price')
    # Plot buy/sell signals
    if strat.buy_signals:
        bx, by = zip(*strat.buy_signals)
        plt.scatter(bx, by, marker='^', color='green', s=100, label='Buy Signal')
    if strat.sell_signals:
        sx, sy = zip(*strat.sell_signals)
        plt.scatter(sx, sy, marker='v', color='red', s=100, label='Sell Signal')
    plt.title(f"Strategy Signals: {default_ticker}")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load data
    df = yf.download(default_ticker, period='2y')
    if df.empty:
        raise ValueError(f"No data for {default_ticker}")
    if isinstance(df.columns, pd.MultiIndex) and default_ticker in df.columns.levels[1]:
        df = df.xs(default_ticker, axis=1, level=1)
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)

    # Run full backtest and show plot
    full_backtest(df)
