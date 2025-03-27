import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Card, 
  CardContent, 
  Typography, 
  Slider, 
  TextField, 
  Button, 
  Grid, 
  Divider,
  CircularProgress,
  Alert,
  Paper
} from '@mui/material';
import axios from 'axios';

const SettingsPanel = () => {
  const [settings, setSettings] = useState({
    max_positions: 5,
    cooldown_minutes: 60,
    min_volatility: 0.05,
    max_volatility: 0.25,
    min_daily_volume_usd: 1000000,
    risk_per_trade: 0.02,
    stop_loss_percentage: 0.05,
    take_profit_percentage: 0.10
  });

  const [systemInfo, setSystemInfo] = useState({
    database: { status: 'unknown', message: 'Loading...' },
    redis: { status: 'not available', message: 'Loading...' },
    system: {
      cpu_usage: 'N/A',
      memory: 'N/A',
      disk_space: 'N/A'
    }
  });

  const [loading, setLoading] = useState(true);
  const [saveLoading, setSaveLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Fetch settings on component mount
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch settings
        const settingsResponse = await axios.get('/api/settings');
        if (settingsResponse.data) {
          setSettings(settingsResponse.data);
        }
        
        // Fetch system info
        const systemInfoResponse = await axios.get('/api/system_info');
        if (systemInfoResponse.data) {
          setSystemInfo(systemInfoResponse.data);
        }
        
      } catch (err) {
        console.error('Error fetching settings:', err);
        setError('Failed to load settings. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchSettings();
    
    // Refresh system info every 30 seconds
    const intervalId = setInterval(async () => {
      try {
        const systemInfoResponse = await axios.get('/api/system_info');
        if (systemInfoResponse.data) {
          setSystemInfo(systemInfoResponse.data);
        }
      } catch (err) {
        console.error('Error refreshing system info:', err);
      }
    }, 30000);

    return () => clearInterval(intervalId);
  }, []);

  const handleChange = (field) => (event, newValue) => {
    if (field === 'max_positions') {
      // For slider, newValue is directly passed
      setSettings({ ...settings, [field]: newValue });
    } else {
      // For text fields, extract value from event
      const value = event.target.value === '' ? '' : Number(event.target.value);
      setSettings({ ...settings, [field]: value });
    }
  };

  const handleSave = async () => {
    try {
      setSaveLoading(true);
      setError(null);
      setSuccessMessage(null);
      
      // Send updated settings to the server
      const response = await axios.post('/api/settings', settings);
      
      if (response.data && response.data.success) {
        setSuccessMessage('Settings saved successfully!');
        
        // Show success message for 3 seconds then hide it
        setTimeout(() => {
          setSuccessMessage(null);
        }, 3000);
      } else {
        setError('Failed to save settings. Please try again.');
      }
    } catch (err) {
      console.error('Error saving settings:', err);
      setError(`Failed to save settings: ${err.response?.data?.error || err.message}`);
    } finally {
      setSaveLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected':
        return 'success.main';
      case 'error':
        return 'error.main';
      case 'unknown':
        return 'warning.main';
      case 'not available':
        return 'text.disabled';
      default:
        return 'text.secondary';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" component="div" gutterBottom>
          Bot Settings
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        {successMessage && (
          <Alert severity="success" sx={{ mb: 2 }}>
            {successMessage}
          </Alert>
        )}
        
        <Grid container spacing={3}>
          {/* Trading Settings */}
          <Grid item xs={12} md={6}>
            <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
              <Typography variant="h6" gutterBottom>
                Trading Parameters
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography gutterBottom>
                  Max Trading Pairs: {settings.max_positions}
                </Typography>
                <Slider
                  value={settings.max_positions}
                  onChange={handleChange('max_positions')}
                  step={1}
                  marks
                  min={1}
                  max={20}
                  valueLabelDisplay="auto"
                />
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <TextField
                  label="Cooldown Period (minutes)"
                  type="number"
                  value={settings.cooldown_minutes}
                  onChange={handleChange('cooldown_minutes')}
                  inputProps={{ min: 0, max: 1440 }}
                  fullWidth
                  margin="normal"
                  size="small"
                />
              </Box>
              
              <Box sx={{ mb: 2 }}>
                <TextField
                  label="Min Daily Volume (USDC)"
                  type="number"
                  value={settings.min_daily_volume_usd}
                  onChange={handleChange('min_daily_volume_usd')}
                  inputProps={{ min: 10000, max: 10000000 }}
                  fullWidth
                  margin="normal"
                  size="small"
                />
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Min Volatility"
                    type="number"
                    value={settings.min_volatility}
                    onChange={handleChange('min_volatility')}
                    inputProps={{ min: 0.01, max: 0.2, step: 0.01 }}
                    fullWidth
                    margin="normal"
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    label="Max Volatility"
                    type="number"
                    value={settings.max_volatility}
                    onChange={handleChange('max_volatility')}
                    inputProps={{ min: 0.05, max: 0.5, step: 0.01 }}
                    fullWidth
                    margin="normal"
                    size="small"
                  />
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 3 }}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={handleSave}
                  disabled={saveLoading}
                  fullWidth
                >
                  {saveLoading ? <CircularProgress size={24} /> : 'Save Settings'}
                </Button>
              </Box>
            </Paper>
          </Grid>
          
          {/* System Information */}
          <Grid item xs={12} md={6}>
            <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default' }}>
              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>
                Database
              </Typography>
              <Box sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    bgcolor: getStatusColor(systemInfo.database.status),
                    mr: 1
                  }}
                />
                <Typography variant="body2">
                  {systemInfo.database.message}
                </Typography>
              </Box>
              
              <Typography variant="subtitle2" gutterBottom>
                Redis Cache
              </Typography>
              <Box sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    bgcolor: getStatusColor(systemInfo.redis.status),
                    mr: 1
                  }}
                />
                <Typography variant="body2">
                  {systemInfo.redis.message}
                </Typography>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="subtitle2" gutterBottom>
                System Resources
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="body2" color="text.secondary">
                    CPU Usage
                  </Typography>
                  <Typography variant="body1">
                    {typeof systemInfo.system.cpu_usage === 'string' 
                      ? systemInfo.system.cpu_usage 
                      : 'N/A'}
                  </Typography>
                </Grid>
                
                <Grid item xs={4}>
                  <Typography variant="body2" color="text.secondary">
                    Memory
                  </Typography>
                  <Typography variant="body1">
                    {typeof systemInfo.system.memory === 'object' 
                      ? systemInfo.system.memory.percent 
                      : 'N/A'}
                  </Typography>
                </Grid>
                
                <Grid item xs={4}>
                  <Typography variant="body2" color="text.secondary">
                    Disk Space
                  </Typography>
                  <Typography variant="body1">
                    {typeof systemInfo.system.disk_space === 'object' 
                      ? systemInfo.system.disk_space.percent 
                      : 'N/A'}
                  </Typography>
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Last updated: {new Date(systemInfo.timestamp).toLocaleTimeString()}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SettingsPanel; 