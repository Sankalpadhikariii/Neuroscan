import React, { useEffect, useState } from 'react';

// Centralized API helper utilities (kept for backward compatibility)
const API_CONFIG = {
  possibleUrls: [
    'http://localhost:5000',
    'http://127.0.0.1:5000',
    'http://0.0.0.0:5000',
  ],

  getBaseUrl: () => {
    if (process.env.REACT_APP_API_BASE_URL) {
      return process.env.REACT_APP_API_BASE_URL;
    }
    return 'http://localhost:5000';
  },

  testConnection: async (url) => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const response = await fetch(`${url}/me`, {
        method: 'GET',
        credentials: 'include',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response.ok || response.status === 401;
    } catch (error) {
      return false;
    }
  },

  findWorkingUrl: async () => {
    const configuredUrl = API_CONFIG.getBaseUrl();

    if (await API_CONFIG.testConnection(configuredUrl)) {
      return configuredUrl;
    }

    for (const url of API_CONFIG.possibleUrls) {
      if (url === configuredUrl) continue;
      if (await API_CONFIG.testConnection(url)) return url;
    }

    return null;
  },
};

let API_BASE = API_CONFIG.getBaseUrl();

export const initializeApiBase = async () => {
  const workingUrl = await API_CONFIG.findWorkingUrl();
  if (workingUrl) {
    API_BASE = workingUrl;
    return workingUrl;
  }
  throw new Error('Backend server not reachable');
};

export { API_BASE, API_CONFIG };

// ConnectionErrorHandler component — wraps app and shows a friendly retry UI when backend is unreachable
export default function ConnectionErrorHandler({ children }) {
  const [connected, setConnected] = useState(true);
  const [checking, setChecking] = useState(false);

  const checkConnection = async () => {
    setChecking(true);
    try {
      const base = API_CONFIG.getBaseUrl();
      const ok = await API_CONFIG.testConnection(base);
      setConnected(!!ok);
    } catch (e) {
      console.error('Connection check failed:', e);
      setConnected(false);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    checkConnection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (connected) return <>{children}</>;

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: '#f9fafb',
      padding: '24px'
    }}>
      <div style={{ maxWidth: 520, textAlign: 'center', borderRadius: 8, padding: 24, background: 'white', boxShadow: '0 4px 20px rgba(0,0,0,0.06)'}}>
        <h2 style={{ margin: 0 }}>Unable to reach backend</h2>
        <p style={{ color: '#444' }}>
          The app could not connect to the backend server. Make sure the server is running and accessible at <code>{API_CONFIG.getBaseUrl()}</code>.
        </p>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
          <button onClick={checkConnection} disabled={checking} style={{ padding: '8px 12px', borderRadius: 6 }}>
            {checking ? 'Retrying...' : 'Retry'}
          </button>
          <button onClick={() => window.location.reload()} style={{ padding: '8px 12px', borderRadius: 6 }}>
            Reload
          </button>
        </div>
      </div>
    </div>
  );
}
