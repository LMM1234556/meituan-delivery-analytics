-- Descriptive decomposition: mean duration gap between late and on-time orders.
-- This is not a causal attribution of lateness.

CREATE OR REPLACE VIEW strategy_orders AS
SELECT *
FROM read_parquet('data/processed/order_fulfillment_mart.parquet')
WHERE is_strategy_eligible = 1;

WITH stage_means AS (
    SELECT
        is_late,
        AVG(order_to_push_minutes) AS order_to_push,
        AVG(order_push_to_accept_minutes) AS push_to_accept,
        AVG(accept_to_pickup_minutes) AS accept_to_pickup,
        AVG(pickup_to_delivery_minutes) AS pickup_to_delivery
    FROM strategy_orders
    GROUP BY is_late
), gaps AS (
    SELECT
        MAX(CASE WHEN is_late = 1 THEN order_to_push END)
          - MAX(CASE WHEN is_late = 0 THEN order_to_push END) AS order_to_push,
        MAX(CASE WHEN is_late = 1 THEN push_to_accept END)
          - MAX(CASE WHEN is_late = 0 THEN push_to_accept END) AS push_to_accept,
        MAX(CASE WHEN is_late = 1 THEN accept_to_pickup END)
          - MAX(CASE WHEN is_late = 0 THEN accept_to_pickup END) AS accept_to_pickup,
        MAX(CASE WHEN is_late = 1 THEN pickup_to_delivery END)
          - MAX(CASE WHEN is_late = 0 THEN pickup_to_delivery END) AS pickup_to_delivery
    FROM stage_means
)
SELECT stage, ROUND(mean_gap_minutes, 4) AS mean_gap_minutes
FROM gaps
UNPIVOT (mean_gap_minutes FOR stage IN (
    order_to_push,
    push_to_accept,
    accept_to_pickup,
    pickup_to_delivery
));

