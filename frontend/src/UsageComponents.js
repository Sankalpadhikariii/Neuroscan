// src/components/UsageComponents.js
import React, { useState } from 'react';
import { 
  AlertCircle, X, Crown, Zap, Clock, 
  CheckCircle, ArrowRight, CreditCard, Sparkles 
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

// ============================================
// USAGE INDICATOR (Sidebar Component)
// ============================================
export function UsageIndicator({ usage, onUpgrade }) {
  if (!usage || usage.is_unlimited) {
    return (
      <div style={{
        padding: '8px 16px',
        background: 'linear-gradient(135deg, #10b981, #059669)',
        borderRadius: '8px',
        color: 'white',
        fontSize: '13px',
        fontWeight: '600',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <Sparkles size={16} />
        Unlimited Scans
      </div>
    );
  }

  const percentage = usage.usage_percent || 0;
  const isWarning = percentage >= 70;
  const isDanger = percentage >= 90;

  const color = isDanger ? '#ef4444' : isWarning ? '#f59e0b' : '#10b981';
  const bgColor = isDanger ? '#fee2e2' : isWarning ? '#fef3c7' : '#dcfce7';

  return (
    <div style={{
      padding: '12px 16px',
      background: bgColor,
      border: `1px solid ${color}20`,
      borderRadius: '8px',
      cursor: isWarning ? 'pointer' : 'default'
    }}
    onClick={isWarning ? onUpgrade : undefined}
    >
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '8px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Zap size={16} color={color} />
          <span style={{ fontSize: '13px', fontWeight: '600', color: '#374151' }}>
            {usage.scans_used} / {usage.scans_limit} scans
          </span>
        </div>
        {isWarning && (
          <button style={{
            padding: '4px 12px',
            background: color,
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '12px',
            fontWeight: '600',
            cursor: 'pointer'
          }}>
            Upgrade
          </button>
        )}
      </div>

      {/* Progress Bar */}
      <div style={{
        width: '100%',
        height: '6px',
        background: '#e5e7eb',
        borderRadius: '3px',
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${Math.min(percentage, 100)}%`,
          height: '100%',
          background: `linear-gradient(90deg, ${color}, ${color}aa)`,
          transition: 'width 0.3s ease'
        }} />
      </div>

      {/* Warning Message */}
      {isWarning && usage.block_message && (
        <p style={{
          margin: '8px 0 0 0',
          fontSize: '11px',
          color: '#6b7280'
        }}>
          {usage.block_message}
        </p>
      )}

      {/* Days Until Reset */}
      {usage.days_until_reset >= 0 && (
        <p style={{
          margin: '4px 0 0 0',
          fontSize: '10px',
          color: '#9ca3af',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          <Clock size={10} />
          Resets in {usage.days_until_reset} days
        </p>
      )}
    </div>
  );
}

// ============================================
// WARNING BANNER (Top of Screen)
// ============================================
export function UsageWarningBanner({ usage, onUpgrade, onDismiss }) {
  if (!usage || usage.usage_percent < 80 || usage.is_blocked) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      background: 'linear-gradient(135deg, #f59e0b, #d97706)',
      color: 'white',
      padding: '12px 24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      zIndex: 999,
      boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <AlertCircle size={20} />
        <div>
          <p style={{ margin: 0, fontWeight: '600', fontSize: '14px' }}>
            You're running low on scans!
          </p>
          <p style={{ margin: '2px 0 0 0', fontSize: '12px', opacity: 0.9 }}>
            Only {usage.scans_remaining} scans remaining this month
          </p>
        </div>
      </div>
      
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <button
          onClick={onUpgrade}
          style={{
            padding: '8px 16px',
            background: 'white',
            color: '#d97706',
            border: 'none',
            borderRadius: '6px',
            fontWeight: '600',
            cursor: 'pointer',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <Crown size={14} />
          Upgrade Now
        </button>
        
        {onDismiss && (
          <button
            onClick={onDismiss}
            style={{
              padding: '4px',
              background: 'transparent',
              border: 'none',
              color: 'white',
              cursor: 'pointer'
            }}
          >
            <X size={18} />
          </button>
        )}
      </div>
    </div>
  );
}

// ============================================
// HARD BLOCK MODAL (Full Screen)
// ============================================
export function UpgradeRequiredModal({ usage, isOpen, onClose }) {
  if (!isOpen || !usage?.is_blocked) return null;

  const canClaimFreeScan = usage.is_free_tier && !usage.cooldown_active;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0, 0, 0, 0.75)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      padding: '20px',
      backdropFilter: 'blur(4px)'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '24px',
        maxWidth: '900px',
        width: '100%',
        maxHeight: '90vh',
        overflow: 'auto',
        boxShadow: '0 25px 50px rgba(0,0,0,0.3)'
      }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          padding: '40px 32px',
          color: 'white',
          textAlign: 'center',
          borderRadius: '24px 24px 0 0'
        }}>
          <div style={{
            width: '80px',
            height: '80px',
            background: 'rgba(255,255,255,0.2)',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 20px'
          }}>
            <Crown size={40} />
          </div>
          
          <h2 style={{
            fontSize: '32px',
            fontWeight: 'bold',
            margin: '0 0 12px 0'
          }}>
            {usage.is_free_tier ? "You've Used All Your Free Scans" : "Usage Limit Reached"}
          </h2>
          
          <p style={{
            fontSize: '18px',
            opacity: 0.95,
            margin: 0
          }}>
            Upgrade to continue scanning brain MRIs with NeuroScan
          </p>
        </div>

        {/* Body */}
        <div style={{ padding: '32px' }}>
          {/* Current Plan Info */}
          <div style={{
            background: '#f9fafb',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '24px',
            border: '1px solid #e5e7eb'
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div>
                <p style={{
                  margin: '0 0 4px 0',
                  fontSize: '13px',
                  color: '#6b7280',
                  fontWeight: '500'
                }}>
                  CURRENT PLAN
                </p>
                <p style={{
                  margin: 0,
                  fontSize: '20px',
                  fontWeight: 'bold',
                  color: '#111827'
                }}>
                  {usage.plan_name}
                </p>
              </div>
              
              <div style={{ textAlign: 'right' }}>
                <p style={{
                  margin: '0 0 4px 0',
                  fontSize: '13px',
                  color: '#6b7280'
                }}>
                  Usage This Month
                </p>
                <p style={{
                  margin: 0,
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#ef4444'
                }}>
                  {usage.scans_used} / {usage.scans_limit}
                </p>
              </div>
            </div>
          </div>

          {/* Cooldown Option (Free Tier Only) */}
          {canClaimFreeScan && (
            <div style={{
              background: '#dbeafe',
              border: '2px solid #3b82f6',
              borderRadius: '12px',
              padding: '20px',
              marginBottom: '24px'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '12px'
              }}>
                <Clock size={24} color="#3b82f6" />
                <h3 style={{
                  margin: 0,
                  fontSize: '18px',
                  fontWeight: '600',
                  color: '#1e40af'
                }}>
                  Free Scan Available!
                </h3>
              </div>
              <p style={{
                margin: '0 0 16px 0',
                fontSize: '14px',
                color: '#1e40af'
              }}>
                Your 24-hour cooldown has ended. Claim 1 free scan now!
              </p>
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${API_BASE}/hospital/claim-free-scan`, {
                      method: 'POST',
                      credentials: 'include'
                    });
                    if (res.ok) {
                      window.location.reload();
                    } else {
                      alert('Failed to claim free scan');
                    }
                  } catch (err) {
                    alert('Error claiming scan');
                  }
                }}
                style={{
                  padding: '12px 24px',
                  background: '#3b82f6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  fontSize: '14px'
                }}
              >
                Claim Free Scan
              </button>
            </div>
          )}

          {usage.cooldown_active && (
            <div style={{
              background: '#fef3c7',
              border: '1px solid #fbbf24',
              borderRadius: '12px',
              padding: '16px',
              marginBottom: '24px',
              textAlign: 'center'
            }}>
              <Clock size={24} color="#f59e0b" style={{ margin: '0 auto 8px' }} />
              <p style={{
                margin: 0,
                fontSize: '14px',
                color: '#78350f',
                fontWeight: '600'
              }}>
                Next free scan available in {usage.cooldown_ends ? 
                  new Date(usage.cooldown_ends).toLocaleTimeString() : '24 hours'}
              </p>
            </div>
          )}

          {/* Upgrade Options */}
          <h3 style={{
            fontSize: '20px',
            fontWeight: '600',
            margin: '0 0 20px 0',
            textAlign: 'center'
          }}>
            Choose Your Plan
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            marginBottom: '24px'
          }}>
            {(usage.upgrade_plans || []).map(plan => (
              <PlanCard key={plan.id} plan={plan} />
            ))}
          </div>

          {/* Benefits */}
          <div style={{
            padding: '20px',
            background: '#f9fafb',
            borderRadius: '12px',
            marginBottom: '20px'
          }}>
            <h4 style={{
              margin: '0 0 12px 0',
              fontSize: '14px',
              fontWeight: '600'
            }}>
              Why Upgrade?
            </h4>
            <div style={{ display: 'grid', gap: '8px' }}>
              {[
                'Unlimited MRI scans',
                'Advanced analytics',
                'GradCAM visualization',
                'Priority support'
              ].map((benefit, idx) => (
                <div key={idx} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '13px'
                }}>
                  <CheckCircle size={14} color="#10b981" />
                  {benefit}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: '20px 32px',
          borderTop: '1px solid #e5e7eb',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <p style={{
            margin: 0,
            fontSize: '12px',
            color: '#6b7280'
          }}>
            ✨ 30-day money-back guarantee
          </p>
          
          {onClose && (
            <button
              onClick={onClose}
              style={{
                padding: '8px 16px',
                background: '#e5e7eb',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '500',
                fontSize: '13px'
              }}
            >
              Maybe Later
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ============================================
// PLAN CARD
// ============================================
function PlanCard({ plan }) {
  async function handleUpgrade() {
    try {
      const res = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plan_id: plan.name,
          billing_cycle: 'monthly'
        })
      });
      
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        alert('Failed to start checkout');
      }
    } catch (err) {
      console.error('Checkout error:', err);
      alert('Failed to start checkout');
    }
  }

  return (
    <div style={{
      border: '1px solid #e5e7eb',
      borderRadius: '12px',
      padding: '20px',
      background: 'white'
    }}>
      <h4 style={{
        margin: '0 0 8px 0',
        fontSize: '16px',
        fontWeight: 'bold'
      }}>
        {plan.display_name}
      </h4>

      <div style={{ marginBottom: '12px' }}>
        <span style={{
          fontSize: '28px',
          fontWeight: 'bold'
        }}>
          ${plan.price_monthly}
        </span>
        <span style={{
          fontSize: '13px',
          color: '#6b7280'
        }}>
          /month
        </span>
      </div>

      <div style={{
        marginBottom: '16px',
        padding: '8px',
        background: '#f9fafb',
        borderRadius: '6px',
        textAlign: 'center',
        fontSize: '12px',
        fontWeight: '600'
      }}>
        {plan.max_scans_per_month === -1 ? 'Unlimited' : plan.max_scans_per_month} scans/month
      </div>

      <button
        onClick={handleUpgrade}
        style={{
          width: '100%',
          padding: '10px',
          background: '#667eea',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          fontWeight: '600',
          cursor: 'pointer',
          fontSize: '13px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '6px'
        }}
      >
        <CreditCard size={14} />
        Upgrade
        <ArrowRight size={14} />
      </button>
    </div>
  );
}