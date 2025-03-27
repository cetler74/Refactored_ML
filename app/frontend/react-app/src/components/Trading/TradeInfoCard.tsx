import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { formatCurrency, formatDateTime, formatPercentage } from '../../utils/formatters';

interface TradeInfoCardProps {
  trade: {
    symbol: string;
    entry_price?: number;
    current_price?: number;
    amount?: number;
    unrealized_pnl?: number;
    unrealized_pnl_pct?: number;
    type?: string;
    timestamp?: string;
    trade_id: string;
  };
}

export const TradeInfoCard: React.FC<TradeInfoCardProps> = ({ trade }) => {
  // Calculate profit/loss
  const isProfitable = trade.unrealized_pnl && trade.unrealized_pnl > 0;
  
  return (
    <Card variant="outlined" sx={{ mb: 2, borderColor: isProfitable ? 'success.main' : 'error.main' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6">{trade.symbol}</Typography>
          {trade.type && (
            <Chip 
              label={trade.type} 
              color={trade.type.toLowerCase().includes('buy') ? 'success' : 'error'} 
              size="small" 
              variant="outlined"
            />
          )}
        </Box>
        
        <Box display="flex" flexDirection="column" gap={1}>
          {trade.entry_price && (
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">Entry Price:</Typography>
              <Typography variant="body2">{formatCurrency(trade.entry_price)}</Typography>
            </Box>
          )}
          
          {trade.current_price && (
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">Current Price:</Typography>
              <Typography variant="body2">{formatCurrency(trade.current_price)}</Typography>
            </Box>
          )}
          
          {trade.amount && (
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">Amount:</Typography>
              <Typography variant="body2">{trade.amount}</Typography>
            </Box>
          )}
          
          {trade.unrealized_pnl && (
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">Unrealized P&L:</Typography>
              <Typography 
                variant="body2" 
                color={isProfitable ? 'success.main' : 'error.main'}
              >
                {formatCurrency(trade.unrealized_pnl)}
              </Typography>
            </Box>
          )}
          
          {trade.unrealized_pnl_pct && (
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">P&L %:</Typography>
              <Typography 
                variant="body2" 
                color={isProfitable ? 'success.main' : 'error.main'}
              >
                {formatPercentage(trade.unrealized_pnl_pct)}
              </Typography>
            </Box>
          )}
          
          {trade.timestamp && (
            <Box display="flex" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">Entry Time:</Typography>
              <Typography variant="body2">{formatDateTime(trade.timestamp)}</Typography>
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default TradeInfoCard; 