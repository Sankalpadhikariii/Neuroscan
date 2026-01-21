import React, { useState, useEffect } from 'react';
import {
  CreditCard, TrendingUp, Users, Activity, AlertCircle,
  Calendar, Zap, Crown, ArrowUpCircle, X, Check
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function SubscriptionDashboard({ darkMode = false }) {
  const [subscription, setSubscription] = useState(null);
  const [usage, setUsage] = useState(null);
  const [daysRemaining, setDaysRemaining] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [plans, setPlans] = useState([]);

  useEffect(() => {
    loadSubscriptionInfo();
    loadPlans();
  }, []);

  async function loadSubscriptionInfo() {
    try {
      const res = await fetch(`${API_BASE}/hospital/subscription`, {
        credentials: 'include'
      });
      if (res.ok) {
        const data = await res.json();
        setSubscription(data.subscription);
        setUsage(data.usage);
        setDaysRemaining(data.days_remaining);
      }
    } catch (err) {
      console.error('Failed to load subscription:', err);
    } finally {
      setLoading(false);
    }
  }

  async function loadPlans() {
    try {
      const res = await fetch(`${API_BASE}/api/subscription/plans`);
      const data = await res.json();
      setPlans(data.plans || []);
    } catch (err) {
      console.error('Failed to load plans:', err);
    }
  }

  async function handleUpgrade(planId, billingCycle) {
    try {
      // Call Stripe checkout session endpoint instead
      const res = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ 
          plan_id: planId, 
          billing_cycle: billingCycle 
        })
      });

      const data = await res.json();

      if (res.ok && data.url) {
        // Redirect to Stripe Checkout
        window.location.href = data.url;
      } else {
        alert(data.error || 'Failed to start checkout. Please try again.');
      }
    } catch (err) {
      console.error('Checkout error:', err);
      alert('Error starting checkout. Please try again.');
    }
  }

  const bg = darkMode ? '#1e293b' : 'white';
  const bgSecondary = darkMode ? '#0f172a' : '#f9fafb';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';
  const border = darkMode ? '#334155' : '#e5e7eb';

  if (loading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: textSecondary
      }}>
        Loading subscription...
      </div>
    );
  }

  if (!subscription) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        background: bg,
        borderRadius: '12px'
      }}>
        <AlertCircle size={48} color="#ef4444" style={{ margin: '0 auto 16px' }} />
        <h3 style={{ color: textPrimary, marginBottom: '8px' }}>No Active Subscription</h3>
        <p style={{ color: textSecondary, marginBottom: '20px' }}>
          Subscribe to a plan to start using NeuroScan
        </p>
        <button
          onClick={() => setShowUpgradeModal(true)}
          style={{
            padding: '12px 24px',
            background: '#667eea',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600'
          }}
        >
          View Plans
        </button>
      </div>
    );
  }

  const scansPercent = usage?.scans_limit === -1
    ? 0
    : (usage?.scans_used / usage?.scans_limit) * 100;

  const isNearLimit = scansPercent > 80;
  const isAtLimit = scansPercent >= 100;

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      {/* Warning Banner */}
      {(isNearLimit || subscription.is_trial) && (
        <div style={{
          padding: '16px 20px',
          background: isAtLimit ? '#fee2e2' : '#fef3c7',
          border: `1px solid ${isAtLimit ? '#fca5a5' : '#fbbf24'}`,
          borderRadius: '12px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <AlertCircle size={20} color={isAtLimit ? '#dc2626' : '#f59e0b'} />
          <div style={{ flex: 1 }}>
            {isAtLimit && (
              <p style={{ margin: 0, color: '#991b1b', fontWeight: '600' }}>
                Scan limit reached! Upgrade your plan to continue.
              </p>
            )}
            {!isAtLimit && isNearLimit && (
              <p style={{ margin: 0, color: '#78350f', fontWeight: '600' }}>
                You're approaching your scan limit. Consider upgrading soon.
              </p>
            )}
            {subscription.is_trial && !isNearLimit && (
              <p style={{ margin: 0, color: '#78350f', fontWeight: '600' }}>
                Free trial • {daysRemaining} days remaining
              </p>
            )}
          </div>
          <button
            onClick={() => setShowUpgradeModal(true)}
            style={{
              padding: '8px 16px',
              background: '#667eea',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            Upgrade Now
          </button>
        </div>
      )}

      {/* Current Plan Card */}
      <div style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        borderRadius: '16px',
        padding: '32px',
        color: 'white',
        marginBottom: '24px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: '-50px',
          right: '-50px',
          width: '200px',
          height: '200px',
          background: 'rgba(255,255,255,0.1)',
          borderRadius: '50%'
        }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: '24px'
          }}>
            <div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                background: 'rgba(255,255,255,0.2)',
                padding: '6px 16px',
                borderRadius: '20px',
                marginBottom: '12px',
                fontSize: '13px',
                fontWeight: '600'
              }}>
                {subscription.name === 'enterprise' ? <Crown size={16} /> : <Zap size={16} />}
                Current Plan
              </div>
              <h2 style={{
                fontSize: '36px',
                fontWeight: 'bold',
                margin: '0 0 8px 0'
              }}>
                {subscription.display_name}
              </h2>
              <p style={{
                fontSize: '18px',
                opacity: 0.9,
                margin: 0
              }}>
                ${subscription.price_monthly}/{subscription.billing_cycle === 'yearly' ? 'year' : 'month'}
              </p>
            </div>

            {subscription.name !== 'enterprise' && (
              <button
                onClick={() => setShowUpgradeModal(true)}
                style={{
                  padding: '10px 20px',
                  background: 'white',
                  color: '#667eea',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: '600',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '14px'
                }}
              >
                <ArrowUpCircle size={18} />
                Upgrade
              </button>
            )}
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '16px'
          }}>
            <div>
              <div style={{ fontSize: '13px', opacity: 0.8, marginBottom: '4px' }}>
                Status
              </div>
              <div style={{ fontSize: '16px', fontWeight: '600' }}>
                {subscription.status === 'active' ? '✓ Active' : subscription.status}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '13px', opacity: 0.8, marginBottom: '4px' }}>
                Billing Cycle
              </div>
              <div style={{ fontSize: '16px', fontWeight: '600', textTransform: 'capitalize' }}>
                {subscription.billing_cycle}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '13px', opacity: 0.8, marginBottom: '4px' }}>
                Renewal Date
              </div>
              <div style={{ fontSize: '16px', fontWeight: '600' }}>
                {new Date(subscription.current_period_end).toLocaleDateString()}
              </div>
            </div>
            <div>
            <div style={{ fontSize: '13px', opacity: 0.8, marginBottom: '4px' }}>
                Days Remaining
              </div>
              <div style={{ fontSize: '16px', fontWeight: '600' }}>
                {daysRemaining} days
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Usage Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '20px',
        marginBottom: '24px'
      }}>
        {/* Scans Usage */}
        <UsageCard
          icon={<Activity size={24} />}
          title="Scans Used"
          current={usage?.scans_used || 0}
          limit={usage?.scans_limit || 0}
          color="#667eea"
          darkMode={darkMode}
        />

        {/* Users */}
        <UsageCard
          icon={<Users size={24} />}
          title="Active Users"
          current={usage?.users_count || 0}
          limit={usage?.users_limit || 0}
          color="#10b981"
          darkMode={darkMode}
        />

        {/* Patients */}
        <UsageCard
          icon={<CreditCard size={24} />}
          title="Total Patients"
          current={usage?.patients_count || 0}
          limit={usage?.patients_limit || 0}
          color="#f59e0b"
          darkMode={darkMode}
        />
      </div>

      {/* Features List */}
      <div style={{
        background: bg,
        borderRadius: '12px',
        padding: '24px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{
          fontSize: '20px',
          fontWeight: '600',
          color: textPrimary,
          marginBottom: '16px'
        }}>
          Your Plan Features
        </h3>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
          gap: '12px'
        }}>
          {(subscription.features || []).map((feature, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px',
                background: bgSecondary,
                borderRadius: '8px'
              }}
            >
              <Check size={18} color="#10b981" />
              <span style={{
                color: textPrimary,
                fontSize: '14px',
                textTransform: 'capitalize'
              }}>
                {feature.replace(/_/g, ' ')}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Upgrade Modal */}
      {showUpgradeModal && (
        <UpgradeModal
          currentPlan={subscription}
          plans={plans}
          onUpgrade={handleUpgrade}
          onClose={() => setShowUpgradeModal(false)}
          darkMode={darkMode}
        />
      )}
    </div>
  );
}

// Usage Card Component
function UsageCard({ icon, title, current, limit, color, darkMode }) {
  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';

  const percentage = limit === -1 ? 0 : (current / limit) * 100;
  const isUnlimited = limit === -1;

  return (
    <div style={{
      background: bg,
      borderRadius: '12px',
      padding: '24px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '16px'
      }}>
        <div style={{
          width: '48px',
          height: '48px',
          borderRadius: '12px',
          background: `${color}20`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: color
        }}>
          {icon}
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{
            fontSize: '28px',
            fontWeight: 'bold',
            color: textPrimary
          }}>
            {current}
          </div>
          <div style={{
            fontSize: '13px',
            color: textSecondary
          }}>
            {isUnlimited ? 'Unlimited' : `of ${limit}`}
          </div>
        </div>
      </div>

      <div style={{ marginBottom: '8px' }}>
        <div style={{
          fontSize: '14px',
          fontWeight: '500',
          color: textPrimary,
          marginBottom: '8px'
        }}>
          {title}
        </div>
        {!isUnlimited && (
          <div style={{
            width: '100%',
            height: '8px',
            background: darkMode ? '#334155' : '#e5e7eb',
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${Math.min(percentage, 100)}%`,
              height: '100%',
              background: percentage > 90 ? '#ef4444' : percentage > 70 ? '#f59e0b' : color,
              transition: 'width 0.3s ease'
            }} />
          </div>
        )}
      </div>

      {!isUnlimited && percentage > 80 && (
        <div style={{
          fontSize: '12px',
          color: percentage >= 100 ? '#dc2626' : '#f59e0b',
          fontWeight: '600'
        }}>
          {percentage >= 100 ? 'Limit reached!' : 'Approaching limit'}
        </div>
      )}
    </div>
  );
}

// Upgrade Modal Component
function UpgradeModal({ currentPlan, plans, onUpgrade, onClose, darkMode }) {
  const [billingCycle, setBillingCycle] = useState('monthly');
  const bg = darkMode ? '#1e293b' : 'white';
  const textPrimary = darkMode ? '#f3f4f6' : '#111827';
  const textSecondary = darkMode ? '#94a3b8' : '#6b7280';

  const availablePlans = plans.filter(p =>
    p.price_monthly > currentPlan.price_monthly && p.name !== 'free'
  );

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div style={{
        background: bg,
        borderRadius: '16px',
        width: '100%',
        maxWidth: '900px',
        maxHeight: '90vh',
        overflowY: 'auto'
      }}>
        <div style={{
          padding: '24px',
          borderBottom: `1px solid ${darkMode ? '#334155' : '#e5e7eb'}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h2 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', color: textPrimary }}>
            Upgrade Your Plan
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '8px'
            }}
          >
            <X size={24} color={textSecondary} />
          </button>
        </div>

        <div style={{ padding: '24px' }}>
          {/* Billing Toggle */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '12px',
            marginBottom: '24px'
          }}>
            <button
              onClick={() => setBillingCycle('monthly')}
              style={{
                padding: '10px 24px',
                background: billingCycle === 'monthly' ? '#667eea' : 'transparent',
                color: billingCycle === 'monthly' ? 'white' : textPrimary,
                border: `1px solid ${darkMode ? '#334155' : '#e5e7eb'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600'
              }}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingCycle('yearly')}
              style={{
                padding: '10px 24px',
                background: billingCycle === 'yearly' ? '#667eea' : 'transparent',
                color: billingCycle === 'yearly' ? 'white' : textPrimary,
                border: `1px solid ${darkMode ? '#334155' : '#e5e7eb'}`,
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                position: 'relative'
              }}
            >
              Yearly
              <span style={{
                position: 'absolute',
                top: '-8px',
                right: '-8px',
                background: '#10b981',
                color: 'white',
                padding: '2px 8px',
                borderRadius: '12px',
                fontSize: '10px'
              }}>
                Save 17%
              </span>
            </button>
          </div>

          {/* Plans Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '20px'
          }}>
            {availablePlans.map(plan => {
              const price = billingCycle === 'yearly' ? plan.price_yearly : plan.price_monthly;

              return (
                <div
                  key={plan.id}
                  style={{
                    border: `2px solid ${plan.name === 'enterprise' ? '#fbbf24' : darkMode ? '#334155' : '#e5e7eb'}`,
                    borderRadius: '12px',
                    padding: '24px',
                    position: 'relative'
                  }}
                >
                  {plan.name === 'enterprise' && (
                    <div style={{
                      position: 'absolute',
                      top: '-12px',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      background: '#fbbf24',
                      color: 'white',
                      padding: '4px 12px',
                      borderRadius: '12px',
                      fontSize: '11px',
                      fontWeight: 'bold'
                    }}>
                      RECOMMENDED
                    </div>
                  )}

                  <h3 style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: textPrimary,
                    marginBottom: '8px'
                  }}>
                    {plan.display_name}
                  </h3>

                  <div style={{
                    fontSize: '32px',
                    fontWeight: 'bold',
                    color: textPrimary,
                    marginBottom: '16px'
                  }}>
                    ${price}
                    <span style={{ fontSize: '14px', fontWeight: 'normal', color: textSecondary }}>
                      /{billingCycle === 'yearly' ? 'year' : 'month'}
                    </span>
                  </div>

                  <div style={{ marginBottom: '20px', fontSize: '14px' }}>
                    <div style={{ color: textPrimary, marginBottom: '8px' }}>
                      <strong>{plan.max_scans_per_month === -1 ? 'Unlimited' : plan.max_scans_per_month}</strong> scans/month
                    </div>
                    <div style={{ color: textPrimary, marginBottom: '8px' }}>
                      <strong>{plan.max_users === -1 ? 'Unlimited' : plan.max_users}</strong> users
                    </div>
                    <div style={{ color: textPrimary }}>
                      <strong>{plan.max_patients === -1 ? 'Unlimited' : plan.max_patients}</strong> patients
                    </div>
                  </div>

                  <button
                    onClick={() => onUpgrade(plan.id, billingCycle)}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: plan.name === 'enterprise'
                        ? 'linear-gradient(135deg, #fbbf24, #f59e0b)'
                        : '#667eea',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontWeight: '600'
                    }}
                  >
                    Upgrade to {plan.display_name}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}