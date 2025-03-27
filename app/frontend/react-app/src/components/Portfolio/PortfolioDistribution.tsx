import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import Plotly from 'plotly.js-dist-min';
import { formatCurrency } from '../../utils/formatters';

interface PortfolioDistributionProps {
  data: {
    symbol: string;
    value: number;
    pnl: number;
  }[];
}

export const PortfolioDistribution: React.FC<PortfolioDistributionProps> = ({ data }) => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const chartElement = chartRef.current;
    if (!chartElement || !data || data.length === 0) {
      return;
    }

    // Prepare data for sunburst chart
    const plotData: Array<Partial<Plotly.PlotData>> = [{
      type: 'sunburst' as const,
      labels: ['Portfolio', ...data.map(item => item.symbol)],
      parents: ['', ...data.map(() => 'Portfolio')],
      values: [data.reduce((sum, item) => sum + item.value, 0), ...data.map(item => item.value)],
      text: ['', ...data.map(item => formatCurrency(item.value))],
      textinfo: 'text',
      hoverinfo: 'all',
      marker: {
        colors: ['#2196f3', ...data.map(item => item.pnl >= 0 ? 'green' : 'red')],
        line: { width: 2 }
      }
    }];

    const layout: Partial<Plotly.Layout> = {
      title: 'Portfolio Distribution',
      showlegend: false,
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
  }, [data]);

  if (!data || data.length === 0) {
    return <Box p={2}>No portfolio data available</Box>;
  }

  return (
    <Box ref={chartRef} width="100%" height={400} />
  );
}; 