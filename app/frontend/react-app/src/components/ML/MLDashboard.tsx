import React from 'react';
import { Box, Card, CardContent, Grid, Typography, Chip, LinearProgress, Divider, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import { useWebSocket } from '../../hooks/useWebSocket';
import { formatPercentage, formatDateTime } from '../../utils/formatters';
import { MLPerformanceChart } from './MLPerformanceChart';
import { FeatureImportance } from './FeatureImportance';
import { PredictionHistory } from './PredictionHistory';

type ModelStatus = 'trained' | 'training' | 'failed' | 'not_trained';

interface ModelStatusChipProps {
  status: ModelStatus | string;
}

const ModelStatusChip: React.FC<ModelStatusChipProps> = ({ status }) => {
  let color: 'default' | 'success' | 'warning' | 'error' = 'default';
  if (status === 'trained') color = 'success';
  if (status === 'training') color = 'warning';
  if (status === 'failed') color = 'error';
  if (status === 'not_trained') color = 'default';

  return (
    <Chip 
      label={status.replace('_', ' ')} 
      color={color} 
      size="small" 
      variant="outlined"
      sx={{ textTransform: 'capitalize' }}
    />
  );
};

export const MLDashboard: React.FC = () => {
  const { data, isConnected, error } = useWebSocket();
  const mlMetrics = data?.ml_metrics;

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

  if (!mlMetrics) {
    return (
      <Box p={3}>
        <Typography>Loading ML metrics...</Typography>
      </Box>
    );
  }

  // Check if essential properties exist
  const hasModels = Array.isArray(mlMetrics.models);
  const hasTrainingStatus = mlMetrics.training_status && typeof mlMetrics.training_status === 'object';
  const hasModelStats = mlMetrics.model_stats && typeof mlMetrics.model_stats === 'object';
  const hasPredictionStats = mlMetrics.prediction_stats && typeof mlMetrics.prediction_stats === 'object';
  const hasModelHealth = mlMetrics.model_health && typeof mlMetrics.model_health === 'object';

  return (
    <Box p={3}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h4" gutterBottom>
            ML Model Dashboard
          </Typography>
        </Grid>

        {/* Models Overview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Models Overview
              </Typography>
              {hasModels ? (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Model Type</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Accuracy</TableCell>
                        <TableCell>Last Training</TableCell>
                        <TableCell>Next Training</TableCell>
                        <TableCell>Samples</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {mlMetrics.models.map((model: any) => (
                        <TableRow key={`${model.symbol || 'unknown'}-${model.model_type || 'unknown'}`}>
                          <TableCell>{model.symbol || 'Unknown'}</TableCell>
                          <TableCell>{model.model_type || 'Unknown'}</TableCell>
                          <TableCell>
                            <ModelStatusChip status={model.status || 'not_trained'} />
                          </TableCell>
                          <TableCell>
                            {model.accuracy ? formatPercentage(model.accuracy) : 'N/A'}
                          </TableCell>
                          <TableCell>
                            {model.last_training ? formatDateTime(model.last_training) : 'Never'}
                          </TableCell>
                          <TableCell>
                            {model.next_training ? formatDateTime(model.next_training) : 'N/A'}
                          </TableCell>
                          <TableCell>{model.samples || 0}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Typography color="textSecondary">No model data available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Training Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Status
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Current Operation: {hasTrainingStatus ? (mlMetrics.training_status.current_operation || 'Idle') : 'Idle'}
                </Typography>
                {hasTrainingStatus && mlMetrics.training_status.in_progress && (
                  <>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Typography variant="body2">
                        {mlMetrics.training_status.progress || 0}%
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        ETA: {mlMetrics.training_status.eta || 'Unknown'}
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={mlMetrics.training_status.progress || 0}
                      sx={{ mt: 1, mb: 2 }}
                    />
                  </>
                )}
              </Box>
              
              <Typography variant="body2" gutterBottom>
                <strong>Models Trained:</strong> {hasModelStats ? (`${mlMetrics.model_stats.trained || 0} / ${mlMetrics.model_stats.total || 0}`) : '0 / 0'}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Last Training Cycle:</strong> {mlMetrics.last_training_cycle ? formatDateTime(mlMetrics.last_training_cycle) : 'Never'}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Next Training Cycle:</strong> {mlMetrics.next_training_cycle ? formatDateTime(mlMetrics.next_training_cycle) : 'Not scheduled'}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Training Frequency:</strong> {mlMetrics.training_frequency || 'Every 24 hours'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Stats */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Prediction Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" gutterBottom>
                    <strong>Predictions Made:</strong> {hasPredictionStats ? (mlMetrics.prediction_stats.total || 0) : 0}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Average Confidence:</strong> {hasPredictionStats ? formatPercentage(mlMetrics.prediction_stats.avg_confidence || 0) : '0%'}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Signals Generated:</strong> {hasPredictionStats ? (mlMetrics.prediction_stats.signals_generated || 0) : 0}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" gutterBottom>
                    <strong>Success Rate:</strong> {hasPredictionStats ? formatPercentage(mlMetrics.prediction_stats.success_rate || 0) : '0%'}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Avg. Prediction Time:</strong> {hasPredictionStats ? `${mlMetrics.prediction_stats.avg_prediction_time || 'N/A'} ms` : 'N/A'}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    <strong>Data Points Used:</strong> {hasPredictionStats ? (mlMetrics.prediction_stats.data_points || 0) : 0}
                  </Typography>
                </Grid>
              </Grid>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                Model Health Status
              </Typography>
              {hasModelHealth ? (
                <Typography variant="body2" color={mlMetrics.model_health.status === 'healthy' ? 'success.main' : mlMetrics.model_health.status === 'warning' ? 'warning.main' : 'error.main'}>
                  {mlMetrics.model_health.status || 'unknown'}: {mlMetrics.model_health.message || 'No health data available'}
                </Typography>
              ) : (
                <Typography variant="body2" color="textSecondary">
                  No model health data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Performance Over Time
              </Typography>
              {Array.isArray(mlMetrics.predictions) && mlMetrics.predictions.length > 0 ? (
                <MLPerformanceChart data={mlMetrics.predictions} />
              ) : (
                <Typography color="textSecondary">No performance data available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Feature Importance
              </Typography>
              {mlMetrics.feature_importance && Object.keys(mlMetrics.feature_importance).length > 0 ? (
                <FeatureImportance features={mlMetrics.feature_importance} />
              ) : (
                <Typography color="textSecondary">No feature importance data available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction History */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Predictions
              </Typography>
              {Array.isArray(mlMetrics.predictions) && mlMetrics.predictions.length > 0 ? (
                <PredictionHistory predictions={mlMetrics.predictions} />
              ) : (
                <Typography color="textSecondary">No prediction history available</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}; 