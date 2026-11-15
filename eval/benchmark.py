"""
benchmark.py
Evaluation Benchmark: 20 standardized questions for data/sample.csv spanning
aggregations, group-by breakdowns, filtering, percentages/shares, and top-N rankings.
Includes verified reference codes and plausible, discriminating wrong analyst variants.
"""

from typing import Any

BENCHMARK: list[dict[str, Any]] = [
    # ------------------------------------------------------------------------
    # Category 1: Simple Aggregations (5 questions)
    # ------------------------------------------------------------------------
    {
        "id": 0,
        "category": "simple aggregation",
        "difficulty": "easy",
        "question": "What is the total number of orders in the dataset?",
        "reference_code": "result = len(df)",
        "wrong_variants": [
            {
                "label": "Summed order_value_usd instead of counting orders",
                "code": "result = df['order_value_usd'].sum()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Counted distinct metro_city values instead of total orders",
                "code": "result = df['metro_city'].nunique()",
                "kind": "wrong_aggregation",
            },
        ],
    },
    {
        "id": 1,
        "category": "simple aggregation",
        "difficulty": "easy",
        "question": "What is the total sum of order_value_usd across all orders?",
        "reference_code": "result = df['order_value_usd'].sum()",
        "wrong_variants": [
            {
                "label": "Calculated average order value instead of sum",
                "code": "result = df['order_value_usd'].mean()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Summed net_effective_order_price instead of order_value_usd",
                "code": "result = df['net_effective_order_price'].sum()",
                "kind": "wrong_column",
            },
        ],
    },
    {
        "id": 2,
        "category": "simple aggregation",
        "difficulty": "easy",
        "question": "What is the mean customer_rating across all orders?",
        "reference_code": "result = df['customer_rating'].mean()",
        "wrong_variants": [
            {
                "label": "Calculated median rating instead of mean",
                "code": "result = df['customer_rating'].median()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Averaged confidence_score instead of customer_rating",
                "code": "result = df['confidence_score'].mean()",
                "kind": "wrong_column",
            },
        ],
    },
    {
        "id": 3,
        "category": "simple aggregation",
        "difficulty": "medium",
        "question": "How many distinct metro_city values are present in the dataset?",
        "reference_code": "result = df['metro_city'].nunique()",
        "wrong_variants": [
            {
                "label": "Counted total rows instead of distinct cities",
                "code": "result = df['metro_city'].count()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Counted distinct cuisine_category instead of metro_city",
                "code": "result = df['cuisine_category'].nunique()",
                "kind": "wrong_column",
            },
        ],
    },
    {
        "id": 4,
        "category": "simple aggregation",
        "difficulty": "medium",
        "question": "What is the overall ratio of total delivery_fee_usd to total order_value_usd across the entire dataset?",
        "reference_code": "result = df['delivery_fee_usd'].sum() / df['order_value_usd'].sum()",
        "wrong_variants": [
            {
                "label": "Averaged the row-level delivery fee ratio instead of taking ratio of sums",
                "code": "result = (df['delivery_fee_usd'] / df['order_value_usd']).mean()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Used net_effective_order_price as denominator instead of order_value_usd",
                "code": "result = df['delivery_fee_usd'].sum() / df['net_effective_order_price'].sum()",
                "kind": "wrong_denominator",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # Category 2: Group-by Aggregations (5 questions)
    # ------------------------------------------------------------------------
    {
        "id": 5,
        "category": "group-by aggregation",
        "difficulty": "easy",
        "question": "What is the average delivery_distance_km for each order_channel?",
        "reference_code": "result = df.groupby('order_channel')['delivery_distance_km'].mean().reset_index()",
        "wrong_variants": [
            {
                "label": "Summed distance instead of calculating average",
                "code": "result = df.groupby('order_channel')['delivery_distance_km'].sum().reset_index()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Grouped by customer_spending_tier instead of order_channel",
                "code": "result = df.groupby('customer_spending_tier')['delivery_distance_km'].mean().reset_index()",
                "kind": "wrong_groupby_key",
            },
        ],
    },
    {
        "id": 6,
        "category": "group-by aggregation",
        "difficulty": "easy",
        "question": "What is the total order_value_usd for each restaurant_price_tier?",
        "reference_code": "result = df.groupby('restaurant_price_tier')['order_value_usd'].sum().reset_index()",
        "wrong_variants": [
            {
                "label": "Calculated mean order value instead of total sum",
                "code": "result = df.groupby('restaurant_price_tier')['order_value_usd'].mean().reset_index()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Summed discount_applied_usd instead of order_value_usd",
                "code": "result = df.groupby('restaurant_price_tier')['discount_applied_usd'].sum().reset_index()",
                "kind": "wrong_column",
            },
        ],
    },
    {
        "id": 7,
        "category": "group-by aggregation",
        "difficulty": "medium",
        "question": "What is the median actual_delivery_time_min for each weather_condition?",
        "reference_code": "result = df.groupby('weather_condition')['actual_delivery_time_min'].median().reset_index()",
        "wrong_variants": [
            {
                "label": "Calculated mean instead of median delivery time",
                "code": "result = df.groupby('weather_condition')['actual_delivery_time_min'].mean().reset_index()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Used estimated_prep_time_min instead of actual_delivery_time_min",
                "code": "result = df.groupby('weather_condition')['estimated_prep_time_min'].median().reset_index()",
                "kind": "wrong_column",
            },
        ],
    },
    {
        "id": 8,
        "category": "group-by aggregation",
        "difficulty": "medium",
        "question": "What is the average order_value_usd for each metro_city, counting only orders placed through the 'Online Delivery App' channel?",
        "reference_code": "result = df[df['order_channel'] == 'Online Delivery App'].groupby('metro_city')['order_value_usd'].mean().reset_index()",
        "wrong_variants": [
            {
                "label": "Omitted the order_channel filter for Online Delivery App",
                "code": "result = df.groupby('metro_city')['order_value_usd'].mean().reset_index()",
                "kind": "missing_filter",
            },
            {
                "label": "Summed order_value_usd instead of taking the average",
                "code": "result = df[df['order_channel'] == 'Online Delivery App'].groupby('metro_city')['order_value_usd'].sum().reset_index()",
                "kind": "wrong_aggregation",
            },
        ],
    },
    {
        "id": 9,
        "category": "group-by aggregation",
        "difficulty": "hard",
        "question": "What is the mean net_effective_order_price for each combination of is_peak_hour and restaurant_price_tier?",
        "reference_code": "result = df.groupby(['is_peak_hour', 'restaurant_price_tier'])['net_effective_order_price'].mean().reset_index()",
        "wrong_variants": [
            {
                "label": "Grouped only by restaurant_price_tier, omitting is_peak_hour",
                "code": "result = df.groupby('restaurant_price_tier')['net_effective_order_price'].mean().reset_index()",
                "kind": "wrong_groupby_key",
            },
            {
                "label": "Averaged order_value_usd instead of net_effective_order_price",
                "code": "result = df.groupby(['is_peak_hour', 'restaurant_price_tier'])['order_value_usd'].mean().reset_index()",
                "kind": "wrong_column",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # Category 3: Filter then Aggregate (4 questions)
    # ------------------------------------------------------------------------
    {
        "id": 10,
        "category": "filter then aggregate",
        "difficulty": "easy",
        "question": "What is the total discount_applied_usd for all orders where is_peak_hour is True?",
        "reference_code": "result = df[df['is_peak_hour'] == True]['discount_applied_usd'].sum()",
        "wrong_variants": [
            {
                "label": "Omitted the is_peak_hour filter, summing across all orders",
                "code": "result = df['discount_applied_usd'].sum()",
                "kind": "missing_filter",
            },
            {
                "label": "Filtered for is_peak_hour is False instead of True",
                "code": "result = df[df['is_peak_hour'] == False]['discount_applied_usd'].sum()",
                "kind": "missing_filter",
            },
        ],
    },
    {
        "id": 11,
        "category": "filter then aggregate",
        "difficulty": "medium",
        "question": "What is the average total_fulfillment_time_min for orders delivered in 'Rainy' or 'Heavy Rain / Storm' weather_condition?",
        "reference_code": "result = df[df['weather_condition'].isin(['Rainy', 'Heavy Rain / Storm'])]['total_fulfillment_time_min'].mean()",
        "wrong_variants": [
            {
                "label": "Included only Rainy weather, missing Heavy Rain / Storm",
                "code": "result = df[df['weather_condition'] == 'Rainy']['total_fulfillment_time_min'].mean()",
                "kind": "missing_filter",
            },
            {
                "label": "Averaged actual_delivery_time_min instead of total_fulfillment_time_min",
                "code": "result = df[df['weather_condition'].isin(['Rainy', 'Heavy Rain / Storm'])]['actual_delivery_time_min'].mean()",
                "kind": "wrong_column",
            },
        ],
    },
    {
        "id": 12,
        "category": "filter then aggregate",
        "difficulty": "medium",
        "question": "What is the mean customer_rating for each cuisine_category among orders with order_value_usd greater than 50?",
        "reference_code": "result = df[df['order_value_usd'] > 50].groupby('cuisine_category')['customer_rating'].mean().reset_index()",
        "wrong_variants": [
            {
                "label": "Omitted the order_value_usd > 50 filter entirely",
                "code": "result = df.groupby('cuisine_category')['customer_rating'].mean().reset_index()",
                "kind": "missing_filter",
            },
            {
                "label": "Calculated median rating instead of mean rating for each cuisine",
                "code": "result = df[df['order_value_usd'] > 50].groupby('cuisine_category')['customer_rating'].median().reset_index()",
                "kind": "wrong_aggregation",
            },
        ],
    },
    {
        "id": 13,
        "category": "filter then aggregate",
        "difficulty": "hard",
        "question": "What is the maximum logistics_friction_index among orders placed through 'Online Delivery App' with delivery_distance_km greater than or equal to 5?",
        "reference_code": "result = df[(df['order_channel'] == 'Online Delivery App') & (df['delivery_distance_km'] >= 5)]['logistics_friction_index'].max()",
        "wrong_variants": [
            {
                "label": "Computed mean logistics_friction_index instead of maximum",
                "code": "result = df[(df['order_channel'] == 'Online Delivery App') & (df['delivery_distance_km'] >= 5)]['logistics_friction_index'].mean()",
                "kind": "wrong_aggregation",
            },
            {
                "label": "Extracted maximum confidence_score instead of logistics_friction_index",
                "code": "result = df[(df['order_channel'] == 'Online Delivery App') & (df['delivery_distance_km'] >= 5)]['confidence_score'].max()",
                "kind": "wrong_column",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # Category 4: Percentages or Shares (4 questions)
    # ------------------------------------------------------------------------
    {
        "id": 14,
        "category": "percentages or shares",
        "difficulty": "medium",
        "question": "What is the percentage share of total orders for each order_channel (from 0 to 100)?",
        "reference_code": "result = (df['order_channel'].value_counts(normalize=True) * 100).reset_index()",
        "wrong_variants": [
            {
                "label": "Forgot to multiply by 100, leaving proportions between 0 and 1",
                "code": "result = df['order_channel'].value_counts(normalize=True).reset_index()",
                "kind": "wrong_denominator",
            },
            {
                "label": "Reported raw order counts instead of percentage share",
                "code": "result = df['order_channel'].value_counts().reset_index()",
                "kind": "wrong_aggregation",
            },
        ],
    },
    {
        "id": 15,
        "category": "percentages or shares",
        "difficulty": "medium",
        "question": "What percentage (from 0 to 100) of all orders received a discount (discount_applied_usd greater than 0)?",
        "reference_code": "result = (df['discount_applied_usd'] > 0).mean() * 100",
        "wrong_variants": [
            {
                "label": "Used greater than or equal to 0 (>=) instead of strictly greater than 0 (>)",
                "code": "result = (df['discount_applied_usd'] >= 0).mean() * 100",
                "kind": "off_by_one_threshold",
            },
            {
                "label": "Forgot to multiply by 100, leaving a proportion from 0 to 1",
                "code": "result = float((df['discount_applied_usd'] > 0).mean())",
                "kind": "wrong_denominator",
            },
        ],
    },
    {
        "id": 16,
        "category": "percentages or shares",
        "difficulty": "hard",
        "question": "What percentage share (from 0 to 100) of total order_value_usd was contributed by each cuisine_category?",
        "reference_code": "result = (df.groupby('cuisine_category')['order_value_usd'].sum() / df['order_value_usd'].sum() * 100).reset_index()",
        "wrong_variants": [
            {
                "label": "Calculated order count percentage instead of order_value_usd share",
                "code": "result = (df['cuisine_category'].value_counts(normalize=True) * 100).reset_index()",
                "kind": "wrong_column",
            },
            {
                "label": "Forgot to multiply by 100, leaving fractions between 0 and 1",
                "code": "result = (df.groupby('cuisine_category')['order_value_usd'].sum() / df['order_value_usd'].sum()).reset_index()",
                "kind": "wrong_denominator",
            },
        ],
    },
    {
        "id": 19,
        "category": "percentages or shares",
        "difficulty": "hard",
        "question": "Among orders where is_peak_hour is True, what percentage (from 0 to 100) were placed through the 'Online Delivery App' channel?",
        "reference_code": "result = (df[df['is_peak_hour'] == True]['order_channel'] == 'Online Delivery App').mean() * 100",
        "wrong_variants": [
            {
                "label": "Used all orders as the denominator instead of only peak-hour orders",
                "code": "result = (df['order_channel'] == 'Online Delivery App').mean() * 100",
                "kind": "wrong_denominator",
            },
            {
                "label": "Forgot to multiply by 100, leaving a proportion from 0 to 1",
                "code": "result = float((df[df['is_peak_hour'] == True]['order_channel'] == 'Online Delivery App').mean())",
                "kind": "wrong_denominator",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # Category 5: Top-N or Highest/Lowest by Group (2 questions)
    # ------------------------------------------------------------------------
    {
        "id": 17,
        "category": "top-N or highest/lowest by group",
        "difficulty": "medium",
        "question": "What are the top 3 metro_city destinations with the highest total order_value_usd?",
        "reference_code": "result = df.groupby('metro_city')['order_value_usd'].sum().nlargest(3).reset_index()",
        "wrong_variants": [
            {
                "label": "Selected bottom 3 (smallest) instead of top 3 (largest)",
                "code": "result = df.groupby('metro_city')['order_value_usd'].sum().nsmallest(3).reset_index()",
                "kind": "wrong_top_n_direction",
            },
            {
                "label": "Ranked by average order value instead of total sum",
                "code": "result = df.groupby('metro_city')['order_value_usd'].mean().nlargest(3).reset_index()",
                "kind": "wrong_aggregation",
            },
        ],
    },
    {
        "id": 18,
        "category": "top-N or highest/lowest by group",
        "difficulty": "medium",
        "question": "What are the 5 cuisine_category options with the lowest average customer_rating, sorted from lowest to highest rating?",
        "reference_code": "result = df.groupby('cuisine_category')['customer_rating'].mean().nsmallest(5).reset_index()",
        "wrong_variants": [
            {
                "label": "Selected top 5 highest ratings instead of lowest 5",
                "code": "result = df.groupby('cuisine_category')['customer_rating'].mean().nlargest(5).reset_index()",
                "kind": "wrong_top_n_direction",
            },
            {
                "label": "Ranked by total sum of ratings instead of average rating",
                "code": "result = df.groupby('cuisine_category')['customer_rating'].sum().nsmallest(5).reset_index()",
                "kind": "wrong_aggregation",
            },
        ],
    },
]
