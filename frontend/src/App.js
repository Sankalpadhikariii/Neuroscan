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
        background: '#f8fafc',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Glow Effects */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '300px',
          height: '300px',
          background: 'radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%)',
          borderRadius: '50%',
          filter: 'blur(20px)',
          animation: 'pulseGlow 3s infinite ease-in-out'
        }} />

        <div style={{ 
          textAlign: 'center', 
          position: 'relative',
          zIndex: 10,
          background: 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          padding: '40px',
          borderRadius: '24px',
          border: '1px solid rgba(255,255,255,0.6)',
          boxShadow: '0 20px 40px rgba(0,0,0,0.05)'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid rgba(59, 130, 246, 0.1)',
            borderTopColor: '#2563eb',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 20px'
          }} />
          <p style={{
            fontSize: '16px',
            fontWeight: '600',
            color: '#1e293b',
            margin: 0,
            letterSpacing: '-0.3px'
          }}>
            Loading NeuroScan...
          </p>
        </div>
        <style>
          {`
            @keyframes spin {
              to { transform: rotate(360deg); }
            }
            @keyframes pulseGlow {
              0%, 100% {
                transform: translate(-50%, -50%) scale(1);
                opacity: 0.8;
              }
              50% {
                transform: translate(-50%, -50%) scale(1.2);
                opacity: 1;
              }
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
