import React, { useState } from 'react';
import { Crown, Zap, ArrowRight, Loader } from 'lucide-react';

export default function UpgradeButton({ 
  currentPlan = 'free',
  targetPlan = 'premium',
  billingCycle = 'monthly',
  darkMode = false,
  size = 'medium',
  fullWidth = false,
  showIcon = true
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";

  const planConfig = {
    basic: { name: 'Basic', color: '#3b82f6', icon: Zap },
    premium: { name: 'Premium', color: '#8b5cf6', icon: Crown },
    enterprise: { name: 'Enterprise', color: '#f59e0b', icon: Crown }
  };

  const config = planConfig[targetPlan] || planConfig.premium;
  const Icon = config.icon;

  const sizes = {
    small: { padding: '8px 16px', fontSize: '14px', iconSize: 16 },
    medium: { padding: '12px 24px', fontSize: '16px', iconSize: 20 },
    large: { padding: '16px 32px', fontSize: '18px', iconSize: 24 }
  };

  const sizeConfig = sizes[size];

  async function handleUpgrade() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plan_id: targetPlan,
          billing_cycle: billingCycle
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to create checkout session');
      }

      if (data.url) {
        // Redirect to Stripe Checkout
        window.location.href = data.url;
      } else {
        throw new Error('No checkout URL received');
      }
    } catch (err) {
      console.error('Upgrade error:', err);
      setError(err.message);
      setLoading(false);
    }
  }

  return (
    <div style={{ width: fullWidth ? '100%' : 'auto' }}>
      <button
        onClick={handleUpgrade}
        disabled={loading}
        style={{
          width: fullWidth ? '100%' : 'auto',
          padding: sizeConfig.padding,
          background: loading 
            ? (darkMode ? '#475569' : '#e2e8f0')
            : `linear-gradient(135deg, ${config.color} 0%, ${config.color}dd 100%)`,
          color: 'white',
          border: 'none',
          borderRadius: '12px',
          fontSize: sizeConfig.fontSize,
          fontWeight: '600',
          cursor: loading ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          transition: 'all 0.3s ease',
          boxShadow: loading ? 'none' : `0 4px 12px ${config.color}33`,
          opacity: loading ? 0.7 : 1
        }}
        onMouseEnter={(e) => {
          if (!loading) {
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = `0 6px 20px ${config.color}44`;
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = `0 4px 12px ${config.color}33`;
        }}
      >
        {loading ? (
          <>
            <Loader size={sizeConfig.iconSize} style={{ animation: 'spin 1s linear infinite' }} />
            <span>Processing...</span>
          </>
        ) : (
          <>
            {showIcon && <Icon size={sizeConfig.iconSize} />}
            <span>Upgrade to {config.name}</span>
            <ArrowRight size={sizeConfig.iconSize} />
          </>
        )}
      </button>

      {error && (
        <p style={{
          marginTop: '8px',
          fontSize: '14px',
          color: '#ef4444',
          textAlign: 'center'
        }}>
          {error}
        </p>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}