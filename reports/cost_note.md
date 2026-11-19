# Cost note

Status: NOT CALCULATED. Measure actual prompt/completion tokens and provider prices for each model call, including input classifier and output audit. Formula: `daily_cost = 1000 * (input_tokens/1e6 * input_price_per_1m + output_tokens/1e6 * output_price_per_1m)`. Add embedding, vector database and hosting costs separately. Record price date and region.
