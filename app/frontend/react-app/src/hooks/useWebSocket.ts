import { useState, useEffect, useCallback, useRef } from 'react';
import { logger } from '../utils/logger'; // Assuming you have a logger utility

interface WebSocketHookOptions {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

// Define a type for the WebSocket data
interface WebSocketData {
  [key: string]: any;
}

export const useWebSocket = (options: WebSocketHookOptions = {}) => {
  const {
    autoReconnect = true,
    reconnectInterval = 5000,
    maxReconnectAttempts = 5
  } = options;

  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState<WebSocketData | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  
  // Use refs to avoid creating multiple connections
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const isConnectingRef = useRef(false);

  // Extract host and port from current URL to make WebSocket connection to same server
  const host = window.location.hostname;
  const port = window.location.port || '3002'; // Use current port or default to 3002
  const wsUrl = `ws://${host}:${port}/ws`;

  const cleanup = useCallback(() => {
    // Clear any pending reconnect timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing socket
    if (socketRef.current) {
      socketRef.current.onopen = null;
      socketRef.current.onmessage = null;
      socketRef.current.onerror = null;
      socketRef.current.onclose = null;
      
      if (socketRef.current.readyState === WebSocket.OPEN || 
          socketRef.current.readyState === WebSocket.CONNECTING) {
        socketRef.current.close();
      }
      socketRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    // Skip if we're already connecting or we've exceeded reconnect attempts
    if (isConnectingRef.current) {
      console.log('Connect called while already connecting - ignoring');
      return null;
    }
    
    if (reconnectAttempts >= maxReconnectAttempts) {
      console.error(`Maximum reconnect attempts (${maxReconnectAttempts}) exceeded`);
      setError(new Error(`Maximum reconnect attempts (${maxReconnectAttempts}) exceeded`));
      return null;
    }

    try {
      // Clean up any existing connections
      cleanup();
      
      // Set connecting flag
      isConnectingRef.current = true;
      
      console.log(`Connecting to WebSocket at ${wsUrl} (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`);
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setReconnectAttempts(0);
        setError(null);
        isConnectingRef.current = false;
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          logger.log('WebSocket message received:', message);

          // Basic validation
          if (!message || typeof message !== 'object' || !message.type || !message.data) {
            logger.warn('Received invalid WebSocket message structure:', message);
            return;
          }

          const messageType = message.type;
          const messageData = message.data;

          setData(prevData => {
            // Create a new state object
            const newData = { ...prevData };

            // Store data under its type
            newData[messageType] = messageData;

            // Special handling for portfolio: maintain backward compatibility
            if (messageType === 'portfolio' && typeof messageData === 'object' && messageData !== null) {
              // Merge portfolio fields into the top level, overwriting existing keys if necessary
              // But prioritize existing keys if they are not from the portfolio message type
              // This logic might need refinement based on exact needs
              Object.keys(messageData).forEach(key => {
                 // Only overwrite if the key doesn't exist or if it originally came from 'portfolio'
                 // This prevents overwriting e.g. 'ml_metrics' if portfolio also had such a key by mistake
                 if (!(key in newData) || newData.hasOwnProperty(key) && newData[key] === prevData?.portfolio?.[key]) {
                    newData[key] = messageData[key];
                 }
              });
               // Ensure 'portfolio' key also holds the data directly
               newData.portfolio = messageData;
               logger.log(`Portfolio data updated. Keys: ${Object.keys(newData.portfolio).join(', ')}`);
            } else {
               logger.log(`Data updated for type: ${messageType}`);
            }

            return newData;
          });
        } catch (e) {
          const errorMsg = `Error processing message: ${e instanceof Error ? e.message : String(e)}`;
          logger.error(errorMsg, 'Raw data:', event.data);
          setError(new Error(errorMsg));
        }
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket connection error'));
        isConnectingRef.current = false;
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        isConnectingRef.current = false;
        socketRef.current = null;
        
        if (autoReconnect && reconnectAttempts < maxReconnectAttempts) {
          console.log(`Scheduling reconnect in ${reconnectInterval}ms`);
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, reconnectInterval);
        }
      };
      
      setSocket(ws);
      return ws;
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setError(error as Error);
      isConnectingRef.current = false;
      return null;
    }
  }, [autoReconnect, maxReconnectAttempts, reconnectAttempts, reconnectInterval, wsUrl, cleanup]);

  useEffect(() => {
    // Only connect on initial mount, not on every state change
    connect();
    
    return () => {
      cleanup();
    };
  }, [connect, cleanup]);

  const sendMessage = useCallback((message: any) => {
    if (socketRef.current && isConnected) {
      socketRef.current.send(typeof message === 'string' ? message : JSON.stringify(message));
    } else {
      console.warn('Cannot send message, WebSocket not connected');
    }
  }, [isConnected]);

  return { isConnected, data, error, sendMessage, reconnect: connect };
}; 