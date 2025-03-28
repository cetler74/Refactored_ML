// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Paper, 
  Tab, 
  Tabs, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Card, 
  CardContent, 
  Chip
} from '@mui/material';
import { useWebSocket } from '../../hooks/useWebSocket';
import { formatDuration } from '../../utils/formatters';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import { logger } from '../../utils/logger';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface Trade {
  id: string;
  trade_id?: string;
  symbol: string;
  type: string;
  entryPrice: number;
  entry_price?: number;
  currentPrice: number;
  current_price?: number;
  quantity: number;
  amount?: number; 
  value: number;
  pnl: number;
  unrealized_pnl?: number;
  pnlPercentage: number;
  unrealized_pnl_pct?: number;
  entryTime: string;
  timestamp?: string;
  status: string;
  side?: string;
  trade_type?: string;
  size_invested?: number;
  duration?: string;
}

interface Order {
  id: string;
  symbol: string;
  type: string;
  side: string;
  price: number;
  amount: number;
  filled_percentage: number;
  status: string;
  created_at: string;
}

interface TradeTypeChipProps {
  type: string;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const TradeTypeChip: React.FC<TradeTypeChipProps> = ({ type }) => {
  const isBuy = type.toLowerCase() === 'buy';
  return (
    <Chip
      icon={isBuy ? <ArrowUpwardIcon /> : <ArrowDownwardIcon />}
      label={type}
      color={isBuy ? 'success' : 'error'}
      size="small"
      variant="outlined"
    />
  );
};

// Function for accessibility props
function a11yProps(index: number) {
  return {
    id: `tab-${index}`,
    'aria-controls': `tabpanel-${index}`,
  };
}

export const TradingView: React.FC = () => {
  const [value, setValue] = useState(0);
  const { data: wsData, error: wsError } = useWebSocket();
  const [buyPositions, setBuyPositions] = useState<Trade[]>([]);
  const [sellPositions, setSellPositions] = useState<Trade[]>([]);
  const [componentError, setComponentError] = useState<string | null>(null);
  
  useEffect(() => {
    setComponentError(null);
    logger.log("TradingView useEffect triggered. Raw wsData:", wsData);
    
    try {
      // Check if we have data
      if (!wsData) {
        logger.log("TradingView: No WebSocket data available yet.");
        return;
      }
      
      // Extract portfolio data robustly
      const portfolio = wsData?.portfolio ||
        (wsData && typeof wsData === 'object' && ('open_positions' in wsData || 'positions' in wsData) ? wsData : null);
      
      if (!portfolio) {
        logger.warn("TradingView: No portfolio data found in wsData.", wsData);
        return;
      }
      
      logger.log("TradingView: Portfolio data extracted:", portfolio);
      
      // Check if open_positions or positions exists
      const positions = portfolio?.open_positions ?? portfolio?.positions ?? {};
      logger.log("TradingView: Positions object:", positions);
      
      if (typeof positions !== 'object' || positions === null || Object.keys(positions).length === 0) {
        logger.log("TradingView: No valid positions found in portfolio data. Clearing positions.");
        setBuyPositions([]);
        setSellPositions([]);
        return;
      }
      
      const buyPos: Trade[] = [];
      const sellPos: Trade[] = [];
      
      // Process positions
      logger.log("TradingView: Processing positions...");
      Object.entries(positions).forEach(([symbol, posData]: [string, any]) => {
        if (!posData || typeof posData !== 'object') {
          logger.warn(`TradingView: Invalid position data for symbol ${symbol}`, posData);
          return;
        }
        
        const trade: Trade = {
          id: posData.position_id ?? posData.trade_id ?? `pos_${symbol}_${Date.now()}`,
          symbol: symbol,
          type: posData.trade_type ?? posData.type ?? 'long',
          entryPrice: posData.entry_price,
          currentPrice: posData.current_price,
          quantity: posData.position_size ?? posData.amount,
          value: posData.position_value ??
                 ((posData.position_size ?? posData.amount ?? 0) * (posData.current_price ?? 0)),
          pnl: posData.pnl ?? posData.unrealized_pnl,
          pnlPercentage: posData.pnl_percentage ?? posData.unrealized_pnl_pct,
          entryTime: posData.entry_time ?? posData.timestamp ?? new Date().toISOString(),
          status: posData.status ?? 'open',
          entry_price: posData.entry_price,
          current_price: posData.current_price,
          size_invested: posData.size_invested,
          amount: posData.amount ?? posData.position_size,
          unrealized_pnl: posData.unrealized_pnl ?? posData.pnl,
          unrealized_pnl_pct: posData.unrealized_pnl_pct ?? posData.pnl_percentage,
          trade_id: posData.trade_id ?? posData.position_id,
          duration: posData.duration
        };
        
        const tradeType = (trade.type || '').toLowerCase();
        if (tradeType === 'buy' || tradeType === 'long') {
          buyPos.push(trade);
        } else if (tradeType === 'sell' || tradeType === 'short') {
          sellPos.push(trade);
        } else {
           logger.warn(`TradingView: Unknown trade type '${trade.type}' for symbol ${symbol}`);
        }
      });
      
      logger.log("TradingView: Processed positions. Buy:", buyPos.length, "Sell:", sellPos.length);
      setBuyPositions(buyPos);
      setSellPositions(sellPos);
      
    } catch (err) {
      logger.error("TradingView: Error processing data in useEffect:", err);
      setComponentError(`Error processing data: ${err instanceof Error ? err.message : String(err)}`);
    }
    
  }, [wsData]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  // Helper function to format currency with color based on value
  const currencyFormat = (value: number | undefined, prefix = '$') => {
    if (value === undefined) return `${prefix}0.00`;
    return `${prefix}${value.toFixed(2)}`;
  };

  // Helper function to format percentage with color based on value
  const percentFormat = (value: number | undefined) => {
    if (value === undefined) return '0.00%';
    return `${value.toFixed(2)}%`;
  };

  // Helper function to determine trade type display
  const getTradeType = (trade: Trade) => {
    // Use type field from our updated interface first
    if (trade.type) {
      return <TradeTypeChip type={trade.type} />;
    }
    
    // Fallbacks for older data format
    if (trade.trade_type) {
      return <TradeTypeChip type={trade.trade_type} />;
    }
    
    if (trade.side) {
      return <TradeTypeChip type={trade.side} />;
    }
    
    return <TradeTypeChip type="unknown" />;
  };

  // Helper function to get trade value color
  const getTradeColor = (trade: Trade) => {
    // For the new field names
    if (trade.pnl !== undefined) {
      return trade.pnl > 0 ? 'success.main' : 'error.main';
    }
    
    // For the old field names
    if (trade.unrealized_pnl !== undefined) {
      return trade.unrealized_pnl > 0 ? 'success.main' : 'error.main';
    }
    
    return 'text.primary';
  };

  // Helper function to get trade PnL value
  const getTradePnL = (trade: Trade) => {
    // For the new field names
    if (trade.pnl !== undefined) {
      return currencyFormat(trade.pnl);
    }
    
    // For the old field names
    if (trade.unrealized_pnl !== undefined) {
      return currencyFormat(trade.unrealized_pnl);
    }
    
    return '$0.00';
  };

  // Helper function to get trade PnL percentage
  const getTradePnLPercentage = (trade: Trade) => {
    // For the new field names
    if (trade.pnlPercentage !== undefined) {
      return percentFormat(trade.pnlPercentage);
    }
    
    // For the old field names
    if (trade.unrealized_pnl_pct !== undefined) {
      return percentFormat(trade.unrealized_pnl_pct);
    }
    
    return '0.00%';
  };

  const renderBuyPositions = () => {
    if (componentError) return null;
    if (!buyPositions || buyPositions.length === 0) {
      return <Typography>No active buy positions</Typography>;
    }

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Active Buy Positions
        </Typography>
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Entry Price</TableCell>
                <TableCell>Current Price</TableCell>
                <TableCell>Amount</TableCell>
                <TableCell>Size Invested</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Unrealized P&L</TableCell>
                <TableCell>P&L %</TableCell>
                <TableCell>Duration</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {buyPositions.map((trade: Trade) => (
                <TableRow key={trade.id || `trade-${trade.symbol}-${Math.random()}`}>
                  <TableCell>{trade.symbol}</TableCell>
                  <TableCell>{currencyFormat(trade.entryPrice || trade.entry_price)}</TableCell>
                  <TableCell>{currencyFormat(trade.currentPrice || trade.current_price)}</TableCell>
                  <TableCell>{trade.quantity || trade.amount || 0}</TableCell>
                  <TableCell>{currencyFormat(trade.size_invested)}</TableCell>
                  <TableCell>{getTradeType(trade)}</TableCell>
                  <TableCell style={{ color: getTradeColor(trade) }}>
                    {getTradePnL(trade)}
                  </TableCell>
                  <TableCell style={{ color: getTradeColor(trade) }}>
                    {getTradePnLPercentage(trade)}
                  </TableCell>
                  <TableCell>{trade.duration || '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderSellPositions = () => {
    if (!sellPositions || sellPositions.length === 0) {
      return <Typography>No active sell positions</Typography>;
    }

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Active Sell Positions
        </Typography>
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Entry Price</TableCell>
                <TableCell>Current Price</TableCell>
                <TableCell>Amount</TableCell>
                <TableCell>Size Invested</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Unrealized P&L</TableCell>
                <TableCell>P&L %</TableCell>
                <TableCell>Duration</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sellPositions.map((trade: Trade) => (
                <TableRow key={trade.id || `trade-${trade.symbol}-${Math.random()}`}>
                  <TableCell>{trade.symbol}</TableCell>
                  <TableCell>{currencyFormat(trade.entryPrice || trade.entry_price)}</TableCell>
                  <TableCell>{currencyFormat(trade.currentPrice || trade.current_price)}</TableCell>
                  <TableCell>{trade.quantity || trade.amount || 0}</TableCell>
                  <TableCell>{currencyFormat(trade.size_invested)}</TableCell>
                  <TableCell>{getTradeType(trade)}</TableCell>
                  <TableCell style={{ color: getTradeColor(trade) }}>
                    {getTradePnL(trade)}
                  </TableCell>
                  <TableCell style={{ color: getTradeColor(trade) }}>
                    {getTradePnLPercentage(trade)}
                  </TableCell>
                  <TableCell>{trade.duration || '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  // Loading/Error state based on WebSocket connection and component errors
  const isLoading = !wsData && !wsError && !componentError;
  const displayError = wsError ? `WebSocket Error: ${wsError.message}` : componentError;

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <Typography variant="h6">Loading trading data...</Typography>
      </Box>
    );
  }

  if (displayError) {
     return (
       <Box display="flex" justifyContent="center" alignItems="center" height="80vh" sx={{ color: 'error.main', padding: 2 }}>
         <Typography variant="h6">
           Error loading trading view: {displayError}
         </Typography>
       </Box>
     );
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={value} onChange={handleTabChange} aria-label="trading tabs">
          <Tab label="Overview" {...a11yProps(0)} />
          <Tab label="Trades" {...a11yProps(1)} />
          <Tab label="Performance" {...a11yProps(2)} />
        </Tabs>
      </Box>
      <TabPanel value={value} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Active Positions
                </Typography>
                {renderBuyPositions()}
                {buyPositions && buyPositions.length > 0 && 
                 sellPositions && sellPositions.length > 0 && 
                  <Box my={3} />
                }
                {renderSellPositions()}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
      <TabPanel value={value} index={1}>
        <Typography variant="h6">Trade History</Typography>
        <Typography>Trade history will be displayed here.</Typography>
      </TabPanel>
      <TabPanel value={value} index={2}>
        <Typography variant="h6">Performance Metrics</Typography>
        <Typography>Performance metrics will be displayed here.</Typography>
      </TabPanel>
    </Box>
  );
};

export default TradingView; 