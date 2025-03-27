import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { formatDate, formatPercentage } from '../../utils/formatters';

interface Prediction {
  timestamp: string;
  confidence: number;
  accuracy: number;
  symbol: string;
  side: string;
}

interface MLPerformanceChartProps {
  data: Prediction[];
}

export const MLPerformanceChart: React.FC<MLPerformanceChartProps> = ({ data }) => {
  // Process data for the chart
  const chartData = data.map(prediction => ({
    timestamp: formatDate(prediction.timestamp),
    confidence: prediction.confidence * 100,
    accuracy: prediction.accuracy * 100
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={chartData}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="timestamp"
          tick={{ fontSize: 12 }}
          angle={-45}
          textAnchor="end"
        />
        <YAxis
          tickFormatter={(value) => `${value}%`}
          domain={[0, 100]}
          tick={{ fontSize: 12 }}
        />
        <Tooltip
          formatter={(value: number) => formatPercentage(value)}
          labelFormatter={(label) => `Time: ${label}`}
        />
        <Legend />
        <Line
          type="monotone"
          dataKey="confidence"
          name="Confidence"
          stroke="#8884d8"
          activeDot={{ r: 8 }}
        />
        <Line
          type="monotone"
          dataKey="accuracy"
          name="Accuracy"
          stroke="#82ca9d"
          activeDot={{ r: 8 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}; 