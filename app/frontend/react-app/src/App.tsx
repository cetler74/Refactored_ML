import React from 'react';
// Reintroduce Router
import { BrowserRouter as Router } from 'react-router-dom';
// Reintroduce Routes and Route
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
// Reintroduce Layout
import { Layout } from './components/Layout';
// Reintroduce PortfolioSummary
import { PortfolioSummary } from './components/Portfolio/PortfolioSummary';
// import { MLDashboard } from './components/ML/MLDashboard';
// import { TradingView } from './components/Trading/TradingView';
// import { Settings } from './components/Settings/Settings';

// Keep theme setup
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1e1e1e',
          borderRadius: 8,
        },
      },
    },
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          {/* Reintroduce Routes structure with only the default route */}
          <Routes>
            <Route path="/" element={<PortfolioSummary />} />
            {/* Keep other routes commented out for now
            <Route path="/ml" element={<MLDashboard />} />
            <Route path="/trading" element={<TradingView />} />
            <Route path="/settings" element={<Settings />} />
            */}
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
};

export default App; 