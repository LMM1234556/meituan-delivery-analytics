-- District × 30-minute operating table.
-- The slot is based on order_push_time: the point when the order enters delivery.

CREATE OR REPLACE VIEW strategy_orders AS
SELECT *
FROM read_parquet('data/processed/order_fulfillment_mart.parquet')
WHERE is_strategy_eligible = 1;

WITH district_slot AS (
    SELECT
        service_date,
        da_id,
        push_time_slot_30m,
        COUNT(*) AS orders,
        SUM(is_late) AS late_orders,
        AVG(is_late::INTEGER) AS late_rate,
        AVG(order_push_to_accept_minutes) AS avg_push_to_accept_minutes,
        AVG(accept_to_pickup_minutes) AS avg_accept_to_pickup_minutes,
        AVG(pickup_to_delivery_minutes) AS avg_pickup_to_delivery_minutes
    FROM strategy_orders
    GROUP BY 1, 2, 3
)
SELECT
    *,
    late_orders * 1.0 / SUM(late_orders) OVER () AS late_order_contribution,
    orders * 1.0 / SUM(orders) OVER () AS order_contribution
FROM district_slot
ORDER BY late_orders DESC, orders DESC;
