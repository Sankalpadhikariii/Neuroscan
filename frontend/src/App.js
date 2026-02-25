import React, { useEffect, useState } from 'react';
import UniversalLogin from './UniversalLogin';
import AdminPortal from './AdminPortal';
import HospitalPortal from './HospitalPortal';
import PatientPortal from './PatientPortal';
import LandingPage from './LandingPage';
import PricingPage from './PricingPage';
import SubscriptionSuccess from './SubscriptionSuccess';
import SubscriptionCancelled from './SubscriptionCancelled';
import ConnectionErrorHandler from './ConnectionErrorHandler';


const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [currentView, setCurrentView] = useState('landing');

  useEffect(() => {
    checkAuth();
  }, []);

  async function checkAuth() {
    // Check URL for Stripe redirect params FIRST
    const urlParams = new URLSearchParams(window.location.search);
    const hasSessionId = urlParams.get('session_id');
    const isCancelled = window.location.pathname === '/subscription-cancelled';

    try {
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
        // Don't override to 'main' if we need to show Stripe result pages
        if (hasSessionId) {
          setCurrentView('success');
        } else if (isCancelled) {
          setCurrentView('cancelled');
        } else {
          setCurrentView('main');
        }
      }
    } catch (err) {
      console.error('Auth check failed:', err);
      try {
        const res = await fetch(`${API_BASE}/me`, { credentials: 'include' });
        if (res.ok) {
          const data = await res.json();
          setUser(data.user);
          if (hasSessionId) {
            setCurrentView('success');
          } else if (isCancelled) {
            setCurrentView('cancelled');
          } else {
            setCurrentView('main');
          }
        }
      } catch (retryErr) {
        console.error('Auth check retry failed:', retryErr);
      }
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
      setCurrentView('landing');
      // Clean URL params
      window.history.replaceState({}, '', '/');
    } catch (err) {
      console.error('Logout failed:', err);
    }
  }

  function handleLogin(userData) {
    setUser(userData);
    setCurrentView('main');
  }

  function handleNavigateToPricing() {
    setCurrentView('pricing');
  }

  function handleNavigateToMain() {
    // Clean up session_id from URL when navigating home
    window.history.replaceState({}, '', '/');
    setCurrentView('main');
  }

  function handleLoginClick() {
    setCurrentView('login');
  }

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
          user={user}
          currentPlan={user?.subscription}
          onBack={handleNavigateToMain}
        />
      )}

      {/* Not authenticated - show Landing Page if not in other views */}
      {!user && currentView === 'landing' && (
        <LandingPage 
          onLogin={handleLogin} 
          onNavigateToPricing={handleNavigateToPricing}
        />
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
}
