// FeatureGate.js
// Component to restrict features based on subscription plan

import React, { useState, useEffect } from 'react';
import { Lock, Crown, ArrowRight, X } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export function FeatureGate({ feature, children, darkMode, fallback }) {
  const [hasAccess, setHasAccess] = useState(false);
  const [loading, setLoading] = useState(true);
  const [planInfo, setPlanInfo] = useState(null);

  useEffect(() => {
    checkAccess();
  }, [feature]);

  async function checkAccess() {
    try {
      const res = await fetch(`${API_BASE}/hospital/check-feature/${feature}`, {
        credentials: 'include'
      });

      if (res.ok) {
        const data = await res.json();
        setHasAccess(data.has_access);
        setPlanInfo(data);
      }
    } catch (err) {
      console.error('Error checking feature access:', err);
      setHasAccess(false);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <div style={{ opacity: 0.5 }}>{children}</div>;
  }

  if (hasAccess) {
    return children;
  }

  // Show fallback or upgrade prompt
  return fallback || <UpgradePrompt feature={feature} planInfo={planInfo} darkMode={darkMode} />;
}


export function UpgradePrompt({ feature, planInfo, darkMode, inline = false }) {
  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  const featureNames = {
    'video_call': 'Video Consultations',
    'ai_chat': 'AI-Powered Chat',
    'advanced_analytics': 'Advanced Analytics',
    'tumor_tracking': 'Tumor Progression Tracking'
  };

  const featureDescriptions = {
    'video_call': 'Connect with patients through secure video calls directly in the platform',
    'ai_chat': 'Get instant AI-powered responses to medical queries and patient questions',
    'advanced_analytics': 'Access detailed analytics and insights from your scan data',
    'tumor_tracking': 'Track tumor growth and changes over time with advanced visualization'
  };

  if (inline) {
    return (
      <div style={{
        padding: '12px 16px',
        background: darkMode ? '#0f172a' : '#fef3c7',
        border: `1px solid ${darkMode ? '#334155' : '#fbbf24'}`,
        borderRadius: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Lock size={16} color="#f59e0b" />
          <span style={{ fontSize: '13px', color: darkMode ? '#fbbf24' : '#78350f' }}>
            <strong>{featureNames[feature] || 'This feature'}</strong> requires {' '}
            {planInfo?.required_plans?.join(' or ')?.toUpperCase()} plan
          </span>
        </div>
        <button
          onClick={() => window.location.href = '/hospital/subscription'}
          style={{
            padding: '6px 12px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '12px',
            fontWeight: '600',
            cursor: 'pointer',
            whiteSpace: 'nowrap'
          }}
        >
          Upgrade
        </button>
      </div>
    );
  }

  return (
    <div style={{
      background: bgColor,
      border: `2px solid ${borderColor}`,
      borderRadius: '16px',
      padding: '32px',
      textAlign: 'center',
      boxShadow: darkMode 
        ? '0 8px 24px rgba(0,0,0,0.3)' 
        : '0 8px 24px rgba(0,0,0,0.08)'
    }}>
      <div style={{
        width: '80px',
        height: '80px',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        margin: '0 auto 20px',
        boxShadow: '0 8px 20px rgba(102, 126, 234, 0.3)'
      }}>
        <Lock size={40} color="white" />
      </div>

      <h3 style={{
        margin: '0 0 8px 0',
        fontSize: '24px',
        fontWeight: '700',
        color: textPrimary
      }}>
        {featureNames[feature] || 'Premium Feature'}
      </h3>

      <p style={{
        margin: '0 0 24px 0',
        fontSize: '15px',
        color: textSecondary,
        lineHeight: '1.6',
        maxWidth: '500px',
        marginLeft: 'auto',
        marginRight: 'auto'
      }}>
        {featureDescriptions[feature] || 'This feature is available on premium plans'}
      </p>

      <div style={{
        padding: '16px',
        background: darkMode ? '#0f172a' : '#f8fafc',
        borderRadius: '12px',
        marginBottom: '24px',
        display: 'inline-block'
      }}>
        <p style={{
          margin: '0 0 8px 0',
          fontSize: '13px',
          color: textSecondary,
          fontWeight: '500'
        }}>
          Available on:
        </p>
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
          {planInfo?.required_plans?.map(plan => (
            <span
              key={plan}
              style={{
                padding: '6px 16px',
                background: plan === 'premium' 
                  ? 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)'
                  : 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
                color: 'white',
                borderRadius: '20px',
                fontSize: '13px',
                fontWeight: '700',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
              }}
            >
              <Crown size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
              {plan}
            </span>
          ))}
        </div>
      </div>

      <button
        onClick={() => window.location.href = '/hospital/subscription'}
        style={{
          padding: '14px 32px',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          border: 'none',
          borderRadius: '12px',
          fontSize: '16px',
          fontWeight: '600',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          boxShadow: '0 4px 16px rgba(102, 126, 234, 0.4)',
          transition: 'all 0.2s'
        }}
        onMouseOver={(e) => {
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.5)';
        }}
        onMouseOut={(e) => {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = '0 4px 16px rgba(102, 126, 234, 0.4)';
        }}
      >
        // In FeatureGate.js, update the blocked message
<UpgradeButton 
  targetPlan="premium"
  darkMode={darkMode}
  size="medium"
/>
        <Crown size={20} />
        Upgrade to Unlock
        <ArrowRight size={18} />
      </button>

      <p style={{
        margin: '16px 0 0 0',
        fontSize: '13px',
        color: textSecondary
      }}>
        Currently on: <strong style={{ color: textPrimary, textTransform: 'uppercase' }}>
          {planInfo?.plan || 'FREE'} PLAN
        </strong>
      </p>
    </div>
  );
}


