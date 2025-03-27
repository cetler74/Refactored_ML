import React, { useState } from 'react';
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
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import { useWebSocket } from '../../hooks/useWebSocket';
import { formatCurrency, formatDateTime, formatPercentage, formatDuration } from '../../utils/formatters';
import CloseIcon from '@mui/icons-material/Close';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import TradeInfoCard from './TradeInfoCard';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface Trade {
  id: string;
  symbol: string;
  type: string;
  entry_price: number;
  current_price: number;
  exit_price?: number;
  amount: number;
  size_invested: number;
  pnl: number;
  pnl_percentage: number;
  duration: number;
  entry_time: string;
  exit_time?: string;
  trade_id: string;
  side?: string | null;
  trade_type?: string;
  timestamp: string;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
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
  const [tabValue, setTabValue] = useState(0);
  const { data, isConnected, error } = useWebSocket();

  // Create trading data from portfolio.positions if available
  const portfolio = data?.portfolio;
  
  // Generate trading data from portfolio positions
  const tradingData = React.useMemo(() => {
    if (!portfolio || !portfolio.positions) {
      return {
        activePositions: [],
        totalValue: 0,
        totalPnl: 0,
        pnlPercentage: 0,
        buyPositions: [],
        sellPositions: []
      };
    }

    const positions = Object.values(portfolio.positions);
    let totalValue = 0;
    let totalPnl = 0;

    const mappedPositions = positions.map((position: any, index) => {
      const id = `position-${index}`;
      const value = position.current_price * position.amount;
      const unrealizedPnl = position.unrealized_pnl || 0;
      const unrealizedPnlPct = position.unrealized_pnl_pct || 0;
      
      totalValue += value;
      totalPnl += unrealizedPnl;

      // Calculate position duration
      const entryTimestamp = position.timestamp ? new Date(position.timestamp).getTime() : Date.now();
      const currentTimestamp = Date.now();
      const durationMs = currentTimestamp - entryTimestamp;
      const duration = formatDuration(durationMs);

      // Calculate size invested
      const sizeInvested = position.size_invested || position.entry_price * position.amount;

      return {
        ...position,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPnlPct,
        duration,
        entry_time: position.timestamp || new Date().toISOString(),
        trade_id: position.trade_id || id,
        side: position.side,
        trade_type: position.trade_type,
        timestamp: position.timestamp || new Date().toISOString(),
        size_invested: sizeInvested
      };
    });

    // Separate buy and sell positions
    const buyPositions = mappedPositions.filter((trade: Trade) => {
      const tradeId = (trade.trade_id || '').toLowerCase();
      const tradeType = (trade.trade_type || '').toLowerCase();
      const side = (trade.side || '').toLowerCase();
      
      return tradeType === 'buy' || side === 'buy' || tradeId.includes('_buy');
    });

    const sellPositions = mappedPositions.filter((trade: Trade) => {
      const tradeId = (trade.trade_id || '').toLowerCase();
      const tradeType = (trade.trade_type || '').toLowerCase();
      const side = (trade.side || '').toLowerCase();
      
      return tradeType === 'sell' || side === 'sell' || tradeId.includes('_sell');
    });

    return {
      activePositions: mappedPositions,
      totalValue,
      totalPnl,
      pnlPercentage: totalValue > 0 ? (totalPnl / totalValue) * 100 : 0,
      buyPositions,
      sellPositions
    };
  }, [portfolio]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Function to extract trade type from various sources
  const getTradeType = (trade: Trade): string => {
    const tradeId = (trade.trade_id || '').toLowerCase();
    let tradeType = trade.trade_type || trade.side || 'Unknown';
    
    // Extract trade type from trade_id if not available
    if (tradeType === 'Unknown' && tradeId) {
      if (tradeId.includes('_buy')) {
        tradeType = 'Buy';
      } else if (tradeId.includes('_sell')) {
        tradeType = 'Sell';
      }
    }
    
    return tradeType;
  };

  const renderBuyPositions = () => {
    if (!tradingData.buyPositions || tradingData.buyPositions.length === 0) {
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
              {tradingData.buyPositions.map((trade: Trade) => (
                <TableRow key={trade.trade_id}>
                  <TableCell>{trade.symbol}</TableCell>
                  <TableCell>${trade.entry_price.toFixed(2)}</TableCell>
                  <TableCell>${trade.current_price.toFixed(2)}</TableCell>
                  <TableCell>{trade.amount}</TableCell>
                  <TableCell>${trade.size_invested?.toFixed(2)}</TableCell>
                  <TableCell>{getTradeType(trade)}</TableCell>
                  <TableCell 
                    style={{ 
                      color: trade.unrealized_pnl >= 0 ? 'green' : 'red' 
                    }}
                  >
                    ${trade.unrealized_pnl.toFixed(2)}
                  </TableCell>
                  <TableCell 
                    style={{ 
                      color: trade.unrealized_pnl_pct >= 0 ? 'green' : 'red' 
                    }}
                  >
                    {trade.unrealized_pnl_pct.toFixed(2)}%
                  </TableCell>
                  <TableCell>{trade.duration}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderSellPositions = () => {
    if (!tradingData.sellPositions || tradingData.sellPositions.length === 0) {
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
              {tradingData.sellPositions.map((trade: Trade) => (
                <TableRow key={trade.trade_id}>
                  <TableCell>{trade.symbol}</TableCell>
                  <TableCell>${trade.entry_price.toFixed(2)}</TableCell>
                  <TableCell>${trade.current_price.toFixed(2)}</TableCell>
                  <TableCell>{trade.amount}</TableCell>
                  <TableCell>${trade.size_invested?.toFixed(2)}</TableCell>
                  <TableCell>{getTradeType(trade)}</TableCell>
                  <TableCell 
                    style={{ 
                      color: trade.unrealized_pnl >= 0 ? 'green' : 'red' 
                    }}
                  >
                    ${trade.unrealized_pnl.toFixed(2)}
                  </TableCell>
                  <TableCell 
                    style={{ 
                      color: trade.unrealized_pnl_pct >= 0 ? 'green' : 'red' 
                    }}
                  >
                    {trade.unrealized_pnl_pct.toFixed(2)}%
                  </TableCell>
                  <TableCell>{trade.duration}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  // Loading state when waiting for WebSocket data
  if (!isConnected || !data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="80vh">
        <Typography variant="h6">
          {error ? `Error: ${error}` : "Loading trading data..."}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="trading tabs">
          <Tab label="Overview" {...a11yProps(0)} />
          <Tab label="Trades" {...a11yProps(1)} />
          <Tab label="Performance" {...a11yProps(2)} />
        </Tabs>
      </Box>
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Active Positions
                </Typography>
                {renderBuyPositions()}
                {tradingData.buyPositions && tradingData.buyPositions.length > 0 && 
                 tradingData.sellPositions && tradingData.sellPositions.length > 0 && 
                  <Box my={3} />
                }
                {renderSellPositions()}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6">Trade History</Typography>
        <Typography>Trade history will be displayed here.</Typography>
      </TabPanel>
      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6">Performance Metrics</Typography>
        <Typography>Performance metrics will be displayed here.</Typography>
      </TabPanel>
    </Box>
  );
};

export default TradingView; 