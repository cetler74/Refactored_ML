import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import Plotly from 'plotly.js-dist-min';
import { formatPercentage } from '../../utils/formatters';

interface FeatureImportanceProps {
  features: Record<string, number>;
}

export const FeatureImportance: React.FC<FeatureImportanceProps> = ({ features }) => {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const chartElement = chartRef.current;
    if (!chartElement || !features || Object.keys(features).length === 0) {
      return;
    }

    // Sort features by importance
    const sortedFeatures = Object.entries(features)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 15); // Show top 15 features

    const plotData: Array<Partial<Plotly.PlotData>> = [{
      type: 'bar' as const,  // Use const assertion to specify exact type
      x: sortedFeatures.map(([, value]) => value * 100),
      y: sortedFeatures.map(([name]) => name),
      orientation: 'h',
      marker: {
        color: sortedFeatures.map(([, value]) => `rgba(66, 135, 245, ${value})`),
      },
      text: sortedFeatures.map(([, value]) => formatPercentage(value * 100)),
      textposition: 'auto',
    }];

    const layout: Partial<Plotly.Layout> = {
      title: 'Top Feature Importance',
      font: {
        family: 'Arial, sans-serif',
        size: 12,
        color: '#7f7f7f'
      },
      showlegend: false,
      width: chartElement.offsetWidth,
      height: 400,
      margin: {
        l: 150,
        r: 30,
        t: 30,
        b: 30
      },
      xaxis: {
        title: 'Importance (%)',
        tickformat: '.1f',
        ticksuffix: '%'
      },
      yaxis: {
        automargin: true
      },
      bargap: 0.2
    };

    Plotly.newPlot(chartElement, plotData, layout);

    // Cleanup
    return () => {
      if (chartElement) {
        Plotly.purge(chartElement);
      }
    };
  }, [features]);

  if (!features || Object.keys(features).length === 0) {
    return <Box p={2}>No feature importance data available</Box>;
  }

  return (
    <Box ref={chartRef} width="100%" height={400} />
  );
}; 