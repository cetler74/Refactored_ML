import { useState, useEffect, useCallback } from 'react';

interface WebSocketData {
  type: string;
  portfolio: {
    total_balance: number;
    available_balance: number;
    positions: any;
    pnl_24h: number;
    win_rate: number;
    timestamp: string;
  };
  ml_metrics: {
    status: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    timestamp: string;
  };
  timestamp: string;
}

interface WebSocketHookOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export const useWebSocket = (options: WebSocketHookOptions = {}) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 5
  } = options;

  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<Error | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  // Extract host and port from current URL to make WebSocket connection to same server
  const host = window.location.hostname;
  const port = window.location.port || '3001'; // Use current port or default to 3001
  // Use dynamic port instead of hardcoded 8501
  const wsUrl = `ws://${host}:${port}/ws`;

  const connect = useCallback(() => {
    try {
      console.log(`Connecting to WebSocket at ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setReconnectAttempts(0);
        setError(null);
      };
      
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          console.log('Received WebSocket data:', parsed);
          setData(parsed);
        } catch (e) {
          console.error('Error parsing WebSocket data:', e);
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket connection error'));
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        if (autoReconnect && reconnectAttempts < maxReconnectAttempts) {
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, reconnectInterval);
        }
      };
      
      setSocket(ws);
      
      return ws;
    } catch (error) {
      setError(error as Error);
      return null;
    }
  }, [autoReconnect, maxReconnectAttempts, reconnectAttempts, reconnectInterval, wsUrl]);

  useEffect(() => {
    const ws = connect();
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: any) => {
    if (socket && isConnected) {
      socket.send(typeof message === 'string' ? message : JSON.stringify(message));
    } else {
      console.warn('Cannot send message, WebSocket not connected');
    }
  }, [socket, isConnected]);

  return { isConnected, data, error, sendMessage, reconnect: connect };
}; 