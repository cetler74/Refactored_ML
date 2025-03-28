import React, { useState, useEffect } from 'react';
import { Box, Card, CardContent, Typography, Grid, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, Stack } from '@mui/material';
import { useWebSocket } from '../../hooks/useWebSocket';
import { PortfolioHeatmap } from './PortfolioHeatmap';
import { formatCurrency, formatPercentage, formatDateTime } from '../../utils/formatters';
import ArrowUpward from '@mui/icons-material/ArrowUpward';
import { logger } from '../../utils/logger';

interface MetricCardProps {
  title: string;
  value: number | string;
  change?: number;
  type?: 'currency' | 'percentage' | 'number';
  subtitle?: string;
  icon?: React.ReactNode;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, type = 'number', subtitle, icon }) => {
  const formattedValue = type === 'currency' ? formatCurrency(Number(value))
    : type === 'percentage' ? formatPercentage(Number(value))
    : value;

  return (
    <Card>
      <CardContent>
        <Typography color="textSecondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h5" component="div">
          {formattedValue}
        </Typography>
        {change !== undefined && (
          <Typography 
            color={change >= 0 ? 'success.main' : 'error.main'}
            variant="body2"
          >
            {change >= 0 ? '↑' : '↓'} {formatPercentage(Math.abs(change))}
          </Typography>
        )}
        {subtitle && (
          <Typography variant="caption" color="textSecondary">
            {subtitle}
          </Typography>
        )}
        {icon && (
          <div style={{ marginTop: 10, textAlign: 'center' }}>
            {icon}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export const PortfolioSummary: React.FC = () => {
  const { data, isConnected, error } = useWebSocket();
  
  // Add debug logs to see what data is available
  console.log('PortfolioSummary - WebSocket data:', data);
  
  // Try to access portfolio data, either directly or via the portfolio field
  // This handles both formats: { portfolio: {...} } or direct access
  const portfolio = data?.portfolio || 
    (data && typeof data === 'object' && 'total_balance' in data ? data : null);
  
  console.log('PortfolioSummary - Portfolio data used:', portfolio);

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error">Error: {error.message || String(error)}</Typography>
      </Box>
    );
  }

  if (!isConnected) {
    return (
      <Box p={3}>
        <Typography>Connecting to server...</Typography>
      </Box>
    );
  }

  if (!portfolio) {
    return (
      <Box p={3}>
        <Typography>Loading portfolio data...</Typography>
      </Box>
    );
  }

  // Check if required fields exist - provide safe defaults
  const total_balance = portfolio.total_balance || 0;
  const available_balance = portfolio.available_balance || 0;
  const in_orders = portfolio.in_orders || 0;
  const total_profit = portfolio.total_profit || 0;
  const total_profit_percentage = portfolio.total_profit_percentage || 0;
  const buy_trades_count = portfolio.buy_trades_count || 0;
  const sell_trades_count = portfolio.sell_trades_count || 0;
  const win_rate = portfolio.win_rate || 0;
  const time_trading = portfolio.time_trading || 0; 
  const trading_start_time = portfolio.trading_start_time || new Date().toISOString();
  const positions = portfolio.positions || {};
  const pnl_history = Array.isArray(portfolio.pnl_history) ? portfolio.pnl_history : [];
  const active_pairs = Array.isArray(portfolio.active_pairs) ? portfolio.active_pairs : [];

  // Calculate time trading in human-readable format
  const days = Math.floor(time_trading / (1000 * 60 * 60 * 24));
  const hours = Math.floor((time_trading % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const minutes = Math.floor((time_trading % (1000 * 60 * 60)) / (1000 * 60));
  const timeTradingFormatted = `${days}d ${hours}h ${minutes}m`;

  // Example: Assume you have a section rendering balance history like this:
  const renderBalanceHistory = (history: any[]) => {
    if (!history || !Array.isArray(history) || history.length === 0) {
      return <Typography>No balance history available.</Typography>;
    }

    return (
      <ul>
        {history.map((entry, index) => {
          let formattedDate = 'Invalid Date'; // Default fallback
          const timestamp = entry?.timestamp; // Get the timestamp safely

          if (timestamp) { // Check if timestamp exists
            try {
              const dateObj = new Date(timestamp);
              // Check if the date object is valid AND finite
              if (dateObj instanceof Date && !isNaN(dateObj.getTime()) && isFinite(dateObj.getTime())) {
                formattedDate = dateObj.toLocaleTimeString(); // Or .toLocaleDateString(), etc.
              } else {
                 logger.warn(`Invalid timestamp value encountered in balance history: ${timestamp}`);
              }
            } catch (e) {
              logger.error(`Error parsing timestamp ${timestamp}:`, e);
            }
          } else {
             logger.warn(`Missing timestamp in balance history entry:`, entry);
          }

          return (
            <li key={entry?.id || index}>
              {formattedDate}: ${entry?.balance?.toFixed(2) ?? 'N/A'}
            </li>
          );
        })}
      </ul>
    );
  };

  return (
    <Box p={3}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h4" gutterBottom>
            Portfolio Overview
          </Typography>
        </Grid>

        {/* Key Metrics - First Row */}
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Total Balance"
            value={total_balance}
            type="currency"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Available Balance"
            value={available_balance}
            type="currency"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="In Orders"
            value={in_orders}
            type="currency"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Total Profit"
            value={total_profit}
            type="currency"
            change={total_profit_percentage}
          />
        </Grid>

        {/* Key Metrics - Second Row */}
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Active Buy Positions"
            value={buy_trades_count}
            type="number"
            icon={<ArrowUpward color="success" />}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Sell Trades"
            value={sell_trades_count}
            type="number"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Win Rate"
            value={win_rate}
            type="percentage"
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Time Trading"
            value={timeTradingFormatted}
            subtitle={`Since ${formatDateTime(trading_start_time)}`}
          />
        </Grid>

        {/* Active Trading Pairs */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Trading Pairs
              </Typography>
              {active_pairs.length > 0 ? (
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {active_pairs.map((pair) => (
                    <Chip key={pair} label={pair} variant="outlined" size="small" sx={{ mb: 1 }} />
                  ))}
                </Stack>
              ) : (
                <Typography color="textSecondary">No active pairs specified.</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Portfolio Heatmap */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Position Distribution
              </Typography>
              <PortfolioHeatmap positions={positions} />
            </CardContent>
          </Card>
        </Grid>

        {/* PnL Over Time Chart */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Profit/Loss Over Time
              </Typography>
              {pnl_history.length > 0 ? (
                <Box height={300}>
                  {/* Chart component would go here */}
                  <Typography color="textSecondary">
                    Chart displaying PnL history over time
                  </Typography>
                </Box>
              ) : (
                <Typography color="textSecondary">No PnL history available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Active Positions */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Positions
              </Typography>
              {Object.keys(positions).length > 0 ? (
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Entry Price</TableCell>
                        <TableCell>Current Price</TableCell>
                        <TableCell>Amount</TableCell>
                        <TableCell>P&L</TableCell>
                        <TableCell>Time Open</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(positions).map(([symbol, position]: [string, any]) => {
                        let formattedOpenTime = '-'; // Default fallback
                        const positionTimestamp = position.open_time || position.timestamp; // Get potential timestamp

                        if (positionTimestamp) { // Check if it exists
                          try {
                            const dateObj = new Date(positionTimestamp);
                            // Check if the date object is valid AND finite
                            if (dateObj instanceof Date && !isNaN(dateObj.getTime()) && isFinite(dateObj.getTime())) {
                              // If valid, format it (this assumes formatDateTime handles valid Date objects or ISO strings)
                              formattedOpenTime = formatDateTime(positionTimestamp);
                            } else {
                               logger.warn(`Invalid position timestamp for ${symbol}: ${positionTimestamp}`);
                            }
                          } catch (e) {
                            logger.error(`Error parsing position timestamp for ${symbol} (${positionTimestamp}):`, e);
                          }
                        } else {
                           logger.warn(`Missing open_time/timestamp for position ${symbol}:`, position);
                        }

                        return (
                          <TableRow key={symbol}>
                            <TableCell>{symbol}</TableCell>
                            <TableCell>{position.side || position.trade_type || 'Unknown'}</TableCell>
                            <TableCell>{formatCurrency(position.entry_price || 0)}</TableCell>
                            <TableCell>{formatCurrency(position.current_price || 0)}</TableCell>
                            <TableCell>{position.amount || position.size || 0}</TableCell>
                            <TableCell
                              style={{ color: (position.unrealized_pnl || 0) >= 0 ? 'green' : 'red' }}
                            >
                              {formatCurrency(position.unrealized_pnl || 0)} ({formatPercentage(position.unrealized_pnl_pct || 0)})
                            </TableCell>
                            <TableCell>{formattedOpenTime}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Typography color="textSecondary">No active positions</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Typography variant="h6">Balance History</Typography>
          {/* Ensure portfolio and balance_history are checked before mapping */}
          {portfolio?.balance_history ? (
             renderBalanceHistory(portfolio.balance_history)
          ) : (
             <Typography>Loading balance history...</Typography>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}; 