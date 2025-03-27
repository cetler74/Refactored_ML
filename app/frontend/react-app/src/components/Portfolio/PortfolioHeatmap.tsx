import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import Plotly from 'plotly.js-dist-min';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

interface Position {
  side?: string;
  trade_type?: string;
  entry_price?: number;
  current_price?: number;
  amount?: number;
  size?: number;
  unrealized_pnl?: number;
  unrealized_pnl_pct?: number;
  value?: number;
}

interface PortfolioHeatmapProps {
  positions: Record<string, Position>;
}

export const PortfolioHeatmap: React.FC<PortfolioHeatmapProps> = ({ positions }) => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const chartElement = chartRef.current;
    if (!chartElement || !positions || Object.keys(positions).length === 0) {
      return;
    }

    const symbols = Object.keys(positions);
    const values = symbols.map(symbol => {
      const pos = positions[symbol] || {};
      // Use unrealized_pnl if available, otherwise fall back to value
      return Math.abs(pos.unrealized_pnl || pos.value || 0);
    });
    
    const texts = symbols.map(symbol => {
      const pos = positions[symbol] || {};
      return `${symbol}<br>` +
        `Side: ${pos.side || pos.trade_type || 'Unknown'}<br>` +
        `Entry: ${formatCurrency(pos.entry_price || 0)}<br>` +
        `Current: ${formatCurrency(pos.current_price || 0)}<br>` +
        `Amount: ${pos.amount || pos.size || 0}<br>` +
        `PnL: ${formatCurrency(pos.unrealized_pnl || 0)} (${formatPercentage(pos.unrealized_pnl_pct || 0)})`;
    });

    const colors = symbols.map(symbol => {
      const pos = positions[symbol] || {};
      return (pos.unrealized_pnl || 0) >= 0 ? 'green' : 'red';
    });

    // Prepare treemap data
    const plotData: Array<Partial<Plotly.PlotData>> = [{
      type: 'treemap' as const,
      labels: symbols,
      parents: symbols.map(() => ''),
      values: values,
      text: texts,
      textinfo: 'label',
      hoverinfo: 'all',
      marker: {
        colors: colors,
        line: { width: 2 }
      }
    }];

    const layout: Partial<Plotly.Layout> = {
      title: 'Portfolio Distribution',
      width: chartElement.offsetWidth,
      height: 400,
      margin: { t: 30, l: 0, r: 0, b: 0 }
    };

    Plotly.newPlot(chartElement, plotData, layout);

    // Cleanup
    return () => {
      if (chartElement) {
        Plotly.purge(chartElement);
      }
    };
  }, [positions]);

  if (!positions || Object.keys(positions).length === 0) {
    return <Box p={2}>No active positions</Box>;
  }

  return (
    <Box ref={chartRef} width="100%" height={400} />
  );
}; 