-- Core fulfillment KPIs on the order-level mart.
-- Run from the project root with DuckDB. One row in the mart equals one order.

CREATE OR REPLACE VIEW strategy_orders AS
SELECT *
FROM read_parquet('data/processed/order_fulfillment_mart.parquet')
WHERE is_strategy_eligible = 1;

SELECT
    COUNT(*) AS strategy_orders,
    SUM(is_late) AS late_orders,
    ROUND(AVG(is_late::INTEGER) * 100, 4) AS late_rate_pct,
    ROUND(AVG(total_fulfillment_minutes), 2) AS avg_fulfillment_minutes,
    ROUND(AVG(promise_minutes), 2) AS avg_promise_minutes
FROM strategy_orders;

SELECT
    service_date,
    COUNT(*) AS orders,
    SUM(is_late) AS late_orders,
    ROUND(AVG(is_late::INTEGER) * 100, 4) AS late_rate_pct
FROM strategy_orders
GROUP BY service_date
ORDER BY service_date;
