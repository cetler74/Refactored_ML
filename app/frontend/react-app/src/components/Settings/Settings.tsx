import React from 'react';
import { Box, Typography } from '@mui/material';
import SettingsPanel from '../../components/SettingsPanel';

export const Settings: React.FC = () => {
  return (
    <Box p={3}>
      <Typography variant="h4" gutterBottom>
        Bot Configuration
      </Typography>
      <SettingsPanel />
    </Box>
  );
}; 