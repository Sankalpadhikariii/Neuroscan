// SubscriptionLimitsBanner.js
// Shows scan limits at the top of hospital portal

import React, { useState, useEffect } from 'react';
import { Activity, Users, Scan, TrendingUp, AlertCircle, Crown } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function SubscriptionLimitsBanner({ darkMode }) {
  const [limits, setLimits] = useState(null);
  const [loading, setLoading] = useState(true);

  const bgColor = darkMode ? '#1e293b' : '#ffffff';
  const textPrimary = darkMode ? '#f1f5f9' : '#0f172a';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';

  useEffect(() => {
    loadLimits();
  }, []);

  async function loadLimits() {
    try {
      const res = await fetch(`${API_BASE}/hospital/subscription-limits`, {
        credentials: 'include'
      });

      if (res.ok) {
        const data = await res.json();
        setLimits(data);
      }
    } catch (err) {
      console.error('Error loading limits:', err);
    } finally {
      setLoading(false);
    }
  }

  if (loading || !limits) return null;

  const scansPercentage = limits.max_scans === 'unlimited' 
    ? 0 
    : (limits.scans_used / limits.max_scans) * 100;

  const getProgressColor = (percentage) => {
    if (percentage >= 90) return '#ef4444';
    if (percentage >= 75) return '#f59e0b';
    return '#10b981';
  };

  const getPlanBadgeColor = (planName) => {
    const plan = planName.toLowerCase();
    if (plan.includes('enterprise')) return 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)';
    if (plan.includes('premium')) return 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)';
    if (plan.includes('basic')) return 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)';
    return 'linear-gradient(135deg, #94a3b8 0%, #64748b 100%)'; // free
  };

  return (
    <div style={{
      background: bgColor,
      border: `1px solid ${borderColor}`,
      borderRadius: '16px',
      padding: '20px 24px',
      marginBottom: '24px',
      boxShadow: darkMode 
        ? '0 4px 12px rgba(0,0,0,0.2)' 
        : '0 4px 12px rgba(0,0,0,0.05)'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        {/* Plan Badge */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div style={{
            background: getPlanBadgeColor(limits.plan_name),
            padding: '8px 20px',
            borderRadius: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.15)'
          }}>
            <Crown size={18} color="white" />
            <span style={{
              color: 'white',
              fontSize: '14px',
              fontWeight: '700',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              {limits.plan_name}
            </span>
          </div>
        </div>

        {/* Limits Display */}
        <div style={{
          display: 'flex',
          gap: '32px',
          flexWrap: 'wrap'
        }}>
          {/* Scans Limit */}
          <div style={{ minWidth: '200px' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px'
            }}>
              <Activity size={16} color={textSecondary} />
              <span style={{
                fontSize: '13px',
                color: textSecondary,
                fontWeight: '500'
              }}>
                Monthly Scans
              </span>
            </div>

            {limits.max_scans === 'unlimited' ? (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                <span style={{
                  fontSize: '28px',
                  fontWeight: '700',
                  background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}>
                  ∞
                </span>
                <span style={{
                  fontSize: '14px',
                  color: textSecondary
                }}>
                  Unlimited
                </span>
              </div>
            ) : (
              <>
                <div style={{
                  display: 'flex',
                  alignItems: 'baseline',
                  gap: '4px',
                  marginBottom: '8px'
                }}>
                  <span style={{
                    fontSize: '24px',
                    fontWeight: '700',
                    color: textPrimary
                  }}>
                    {limits.scans_used}
                  </span>
                  <span style={{
                    fontSize: '16px',
                    color: textSecondary
                  }}>
                    / {limits.max_scans}
                  </span>
                </div>

                {/* Progress Bar */}
                <div style={{
                  width: '100%',
                  height: '6px',
                  background: darkMode ? '#0f172a' : '#f1f5f9',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${Math.min(scansPercentage, 100)}%`,
                    height: '100%',
                    background: getProgressColor(scansPercentage),
                    borderRadius: '3px',
                    transition: 'width 0.3s ease'
                  }} />
                </div>

                {scansPercentage >= 75 && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    marginTop: '6px'
                  }}>
                    <AlertCircle size={14} color={getProgressColor(scansPercentage)} />
                    <span style={{
                      fontSize: '11px',
                      color: getProgressColor(scansPercentage),
                      fontWeight: '500'
                    }}>
                      {scansPercentage >= 90 ? 'Limit almost reached!' : 'Approaching limit'}
                    </span>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Users Limit */}
          <div style={{ minWidth: '150px' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px'
            }}>
              <Users size={16} color={textSecondary} />
              <span style={{
                fontSize: '13px',
                color: textSecondary,
                fontWeight: '500'
              }}>
                Staff Users
              </span>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '4px'
            }}>
              <span style={{
                fontSize: '24px',
                fontWeight: '700',
                color: textPrimary
              }}>
                {limits.users_count}
              </span>
              <span style={{
                fontSize: '16px',
                color: textSecondary
              }}>
                / {limits.max_users === 'unlimited' ? '∞' : limits.max_users}
              </span>
            </div>
          </div>

          {/* Patients Limit */}
          <div style={{ minWidth: '150px' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px'
            }}>
              <TrendingUp size={16} color={textSecondary} />
              <span style={{
                fontSize: '13px',
                color: textSecondary,
                fontWeight: '500'
              }}>
                Total Patients
              </span>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '4px'
            }}>
              <span style={{
                fontSize: '24px',
                fontWeight: '700',
                color: textPrimary
              }}>
                {limits.patients_count}
              </span>
              <span style={{
                fontSize: '16px',
                color: textSecondary
              }}>
                / {limits.max_patients === 'unlimited' ? '∞' : limits.max_patients}
              </span>
            </div>
          </div>
        </div>

        {/* Upgrade Button (if not on highest plan) */}
        {limits.plan_name.toLowerCase() !== 'enterprise' && (
          <button
            onClick={() => window.location.href = '/hospital/subscription'}
            style={{
              padding: '10px 24px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 16px rgba(102, 126, 234, 0.4)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.3)';
            }}
          >
            <Crown size={16} />
            Upgrade Plan
          </button>
        )}
      </div>
    </div>
  );
}