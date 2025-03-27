# Migration from 'side' to 'trade_type'

This document explains the migration from using 'side' and 'direction' fields to the more semantically appropriate 'trade_type' field for indicating buy or sell actions in trades and signals.

## Changes Made

1. Database Models
   - Renamed `OrderSide` enum to `TradeType` in `app/database/models.py`
   - Updated the `Trade` and `Signal` models to use `trade_type` instead of `side`

2. Database Migration
   - Created a new migration file to rename columns in the database
   - Migration handles data conversion between old and new column names

3. Strategy Module
   - Updated all strategies to use `trade_type` instead of `side`
   - Modified `get_stop_loss` and `get_take_profit` methods to use `trade_type` parameter
   - Updated signal generation to use `trade_type` field

4. Main Application
   - Added compatibility layer in `main.py` to handle both legacy 'side'/'direction' fields and the new 'trade_type' field

## How to Apply the Migration

To apply the migration, run:

```bash
python update_to_trade_type.py
```

This will update the database schema to use `trade_type` instead of `side`.

## Compatibility Considerations

The system now supports and standardizes on using `trade_type` for indicating buy or sell trades. However, for backward compatibility:

- Signals with 'side' or 'direction' fields will still work
- These legacy fields are automatically mapped to the new 'trade_type' field
- External systems that integrate with the API should update to use 'trade_type'

## Benefits

1. **Semantic Clarity**: 'trade_type' more clearly represents what the field indicates - the type of trade (buy or sell)
2. **Standardization**: Eliminates the confusion between 'side' and 'direction' fields
3. **Maintainability**: A single, well-named field is easier to maintain and understand 