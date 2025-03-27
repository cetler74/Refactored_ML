import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box
} from '@mui/material';
import { formatDateTime, formatPercentage } from '../../utils/formatters';

interface Prediction {
  timestamp: string;
  symbol: string;
  prediction: 'buy' | 'sell' | 'hold';
  confidence: number;
  actual_movement?: number;
  accuracy?: number;
}

interface PredictionHistoryProps {
  predictions: Prediction[];
}

export const PredictionHistory: React.FC<PredictionHistoryProps> = ({ predictions }) => {
  if (!predictions || predictions.length === 0) {
    return (
      <Box p={2}>
        <Typography variant="body1">No prediction history available</Typography>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Time</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell>Prediction</TableCell>
            <TableCell align="right">Confidence</TableCell>
            <TableCell align="right">Actual Movement</TableCell>
            <TableCell align="right">Accuracy</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {predictions.map((prediction, index) => (
            <TableRow key={`${prediction.timestamp}-${index}`}>
              <TableCell>{formatDateTime(prediction.timestamp)}</TableCell>
              <TableCell>{prediction.symbol}</TableCell>
              <TableCell>
                <Box
                  component="span"
                  sx={{
                    color: prediction.prediction === 'buy' ? 'success.main' :
                          prediction.prediction === 'sell' ? 'error.main' :
                          'text.secondary',
                    textTransform: 'capitalize',
                    fontWeight: 'medium'
                  }}
                >
                  {prediction.prediction}
                </Box>
              </TableCell>
              <TableCell align="right">{formatPercentage(prediction.confidence * 100)}</TableCell>
              <TableCell align="right">
                {prediction.actual_movement !== undefined ? 
                  <Box
                    component="span"
                    sx={{
                      color: prediction.actual_movement > 0 ? 'success.main' :
                            prediction.actual_movement < 0 ? 'error.main' :
                            'text.secondary'
                    }}
                  >
                    {formatPercentage(prediction.actual_movement * 100)}
                  </Box>
                  : '-'
                }
              </TableCell>
              <TableCell align="right">
                {prediction.accuracy !== undefined ? 
                  formatPercentage(prediction.accuracy * 100)
                  : '-'
                }
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}; 