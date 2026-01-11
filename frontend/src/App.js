<<<<<<< HEAD
import React from 'react';
=======
import React, { useEffect, useState } from 'react';
>>>>>>> 25cee4587776c53dfb2b0f21018e1885f1f153c1
import UniversalLogin from './UniversalLogin';
import AdminPortal from './AdminPortal';
import HospitalPortal from './HospitalPortal';
import PatientPortal from './PatientPortal';
import PricingPage from './PricingPage';
import SubscriptionSuccess from './SubscriptionSuccess';
import SubscriptionCancelled from './SubscriptionCancelled';
<<<<<<< HEAD
import ConnectionErrorHandler from './ConnectionErrorHandler';
import { useState, useEffect } from 'react';

=======
>>>>>>> 25cee4587776c53dfb2b0f21018e1885f1f153c1

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
<<<<<<< HEAD
  const [currentView, setCurrentView] = useState('main');
  const [connectionError, setConnectionError] = useState(false);
=======
  const [currentView, setCurrentView] = useState('main'); // 'main', 'pricing', 'success', 'cancelled'
>>>>>>> 25cee4587776c53dfb2b0f21018e1885f1f153c1

  useEffect(() => {
    checkAuth();
  }, []);

  async function checkAuth() {
    try {
<<<<<<< HEAD
      // Test connection with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const res = await fetch(`${API_BASE}/me`, { 
        credentials: 'include',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        setUser(data.user);
        setConnectionError(false);
      } else {
        setConnectionError(false); // Server responded but user not authenticated
      }
    } catch (err) {
      console.error('Auth check failed:', err);
      if (err.name === 'AbortError' || err.message.includes('fetch')) {
        setConnectionError(true);
      }
=======
      const res = await fetch(`${API_BASE}/me`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setUser(data.user);
      }
    } catch (err) {
      console.error('Auth check failed:', err);
>>>>>>> 25cee4587776c53dfb2b0f21018e1885f1f153c1
    } finally {
      setLoading(false);
    }
  }

  async function handleLogout() {
    try {
      await fetch(`${API_BASE}/logout`, {
        method: 'POST',
        credentials: 'include'
      });
      setUser(null);
    } catch (err) {
      console.error('Logout failed:', err);
    }
  }

  function handleLogin(userData) {
    setUser(userData);
  }

  function handleNavigateToPricing() {
    setCurrentView('pricing');
  }

  function handleNavigateToMain() {
    setCurrentView('main');
  }

  // Check URL for success/cancelled params
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('session_id')) {
      setCurrentView('success');
    } else if (window.location.pathname === '/subscription-cancelled') {
      setCurrentView('cancelled');
    }
  }, []);

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}>
        <div style={{ textAlign: 'center', color: 'white' }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid rgba(255,255,255,0.3)',
            borderTopColor: 'white',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px'
          }} />
          <p>Loading NeuroScan...</p>
        </div>
        <style>
          {`
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    );
  }

<<<<<<< HEAD
  // Wrap entire app with connection error handler
  return (
    <ConnectionErrorHandler>
      {/* Show success page */}
      {currentView === 'success' && (
        <SubscriptionSuccess onNavigateHome={handleNavigateToMain} />
      )}

      {/* Show cancelled page */}
      {currentView === 'cancelled' && (
        <SubscriptionCancelled onNavigateHome={handleNavigateToMain} />
      )}

      {/* Show pricing page */}
      {currentView === 'pricing' && (
        <PricingPage 
          currentPlan={user?.subscription}
          onBack={handleNavigateToMain}
        />
      )}

      {/* Not authenticated - show login */}
      {!user && currentView === 'main' && (
        <UniversalLogin onLogin={handleLogin} />
      )}

      {/* Route based on user type */}
      {user && currentView === 'main' && (
        <>
          {user.type === 'admin' && (
            <AdminPortal 
              user={user} 
              onLogout={handleLogout} 
              onNavigateToPricing={handleNavigateToPricing} 
            />
          )}

          {user.type === 'hospital' && (
            <HospitalPortal 
              user={user} 
              onLogout={handleLogout} 
              onNavigateToPricing={handleNavigateToPricing} 
            />
          )}

          {user.type === 'patient' && (
            <PatientPortal 
              patient={user} 
              onLogout={handleLogout} 
            />
          )}
        </>
      )}
    </ConnectionErrorHandler>
  );
=======
  // Show success page
  if (currentView === 'success') {
    return <SubscriptionSuccess onNavigateHome={handleNavigateToMain} />;
  }

  // Show cancelled page
  if (currentView === 'cancelled') {
    return <SubscriptionCancelled onNavigateHome={handleNavigateToMain} />;
  }

  // Show pricing page
  if (currentView === 'pricing') {
    return (
      <PricingPage 
        currentPlan={user?.subscription}
        onBack={handleNavigateToMain}
      />
    );
  }

  // Not authenticated - show login
  if (!user) {
    return <UniversalLogin onLogin={handleLogin} />;
  }

  // Route based on user type
  if (user.type === 'admin') {
    return <AdminPortal user={user} onLogout={handleLogout} onNavigateToPricing={handleNavigateToPricing} />;
  }

  if (user.type === 'hospital') {
    return <HospitalPortal user={user} onLogout={handleLogout} onNavigateToPricing={handleNavigateToPricing} />;
  }

  if (user.type === 'patient') {
    return <PatientPortal patient={user} onLogout={handleLogout} />;
  }

  // Fallback
  return <UniversalLogin onLogin={handleLogin} />;
>>>>>>> 25cee4587776c53dfb2b0f21018e1885f1f153c1
}