"""Run the portfolio SQL files and assert the headline KPI scope."""

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    connection = duckdb.connect()
    for sql_file in sorted((ROOT / "sql").glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8")
        relation = connection.sql(sql)
        frame = relation.df()
        print(f"[{sql_file.name}] rows={len(frame):,}")
        print(frame.head(10).to_string(index=False))

    strategy_orders, late_orders, late_rate = connection.execute(
        """
        SELECT COUNT(*), SUM(is_late::INTEGER), AVG(is_late::INTEGER)
        FROM read_parquet(?)
        WHERE is_strategy_eligible = 1
        """,
        [str(ROOT / "data" / "processed" / "order_fulfillment_mart.parquet")],
    ).fetchone()
    assert strategy_orders == 436_062
    assert late_orders == 59_456
    assert abs(late_rate - 0.136347577) < 1e-6
    print("SQL CHECKS PASSED")


if __name__ == "__main__":
    main()
