"""
CSV Logger — автоматическое логирование рыночных данных.

Вдохновлено:
- txbabaxyz/polyrec: automatic CSV logging per 15-minute market
"""
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.data_pipeline import MarketSnapshot, OrderBook, PriceTick
from core.strategy_engine import Signal

logger = logging.getLogger(__name__)


class CSVLogger:
    """Логирование данных в CSV для бэктестинга и анализа."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writers: Dict[str, csv.writer] = {}
        self._files: Dict[str, object] = {}

    def _get_filepath(self, category: str, date: Optional[str] = None) -> Path:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"{category}_{date}.csv"

    def log_market(self, market: MarketSnapshot):
        """Логировать снимок рынка."""
        path = self._get_filepath("markets")
        is_new = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    "timestamp", "market_id", "question",
                    "yes_price", "no_price", "spread",
                    "volume_24h", "liquidity",
                ])
            writer.writerow([
                datetime.now().isoformat(),
                market.market_id,
                market.question[:100],
                f"{market.yes_price:.4f}",
                f"{market.no_price:.4f}",
                f"{market.spread:.4f}",
                f"{market.volume_24h:.2f}",
                f"{market.liquidity:.2f}",
            ])

    def log_signal(self, signal: Signal):
        """Логировать сигнал стратегии."""
        path = self._get_filepath("signals")
        is_new = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    "timestamp", "strategy", "market_id",
                    "signal_type", "confidence",
                    "price", "size", "reason",
                ])
            writer.writerow([
                datetime.now().isoformat(),
                signal.strategy_name,
                signal.market_id,
                signal.signal_type.value,
                signal.confidence.value,
                f"{signal.suggested_price:.4f}",
                f"{signal.suggested_size:.4f}",
                signal.reason,
            ])

    def log_trade(self, trade_data: Dict):
        """Логировать сделку."""
        path = self._get_filepath("trades")
        is_new = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    "timestamp", "id", "market_id", "side",
                    "entry_price", "exit_price", "shares",
                    "pnl", "pnl_pct", "strategy",
                ])
            writer.writerow([
                datetime.now().isoformat(),
                trade_data.get("id", ""),
                trade_data.get("market_id", ""),
                trade_data.get("side", ""),
                trade_data.get("entry_price", ""),
                trade_data.get("exit_price", ""),
                trade_data.get("shares", ""),
                trade_data.get("pnl", ""),
                trade_data.get("pnl_pct", ""),
                trade_data.get("strategy", ""),
            ])

    def log_price_tick(self, tick: PriceTick):
        """Логировать ценовой тик (Binance и т.д.)."""
        path = self._get_filepath(f"prices_{tick.symbol.lower()}")
        is_new = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "symbol", "price", "volume", "source"])
            writer.writerow([
                datetime.now().isoformat(),
                tick.symbol,
                f"{tick.price:.2f}",
                f"{tick.volume:.4f}",
                tick.source,
            ])

    def get_log_stats(self) -> str:
        """Статистика по логам."""
        files = list(self.log_dir.glob("*.csv"))
        if not files:
            return "📂 Логи пусты."

        total_size = sum(f.stat().st_size for f in files)
        lines = [f"📂 *Логи* ({len(files)} файлов, {total_size / 1024:.1f} KB)\n"]

        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            size = f.stat().st_size / 1024
            # Подсчёт строк
            with open(f) as fp:
                row_count = sum(1 for _ in fp) - 1  # минус заголовок
            lines.append(f"  `{f.name}` — {row_count} записей ({size:.1f} KB)")

        return "\n".join(lines)
