import React, { useState, useEffect } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import { Check, X, CreditCard, Shield, Zap } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

// Initialize Stripe
let stripePromise = null;

export default function PricingPage({ currentPlan, onSelectPlan }) {
  const [plans, setPlans] = useState([]);
  const [billingCycle, setBillingCycle] = useState('monthly');
  const [loading, setLoading] = useState(true);
  const [processingPlan, setProcessingPlan] = useState(null);

  useEffect(() => {
    loadPlans();
    initializeStripe();
  }, []);

  async function initializeStripe() {
    try {
      const res = await fetch(`${API_BASE}/stripe-config`, { credentials: 'include' });
      const { publishableKey } = await res.json();
      stripePromise = loadStripe(publishableKey);
    } catch (error) {
      console.error('Failed to initialize Stripe:', error);
    }
  }

  async function loadPlans() {
    try {
      const res = await fetch(`${API_BASE}/subscription-plans`, { credentials: 'include' });
      const data = await res.json();
      setPlans(data.plans || []);
    } catch (error) {
      console.error('Failed to load plans:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleSubscribe(plan) {
    if (!plan || plan.price_monthly === 0) return;
    
    setProcessingPlan(plan.id);
    
    try {
      // Create checkout session
     const res = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          plan_id: plan.id,
          billing_cycle: billingCycle
        })
      });

      const { sessionId, url } = await res.json();
      
      if (url) {
        // Redirect to Stripe Checkout
        window.location.href = url;
      }
    } catch (error) {
      console.error('Checkout error:', error);
      alert('Failed to start checkout. Please try again.');
    } finally {
      setProcessingPlan(null);
    }
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
          <p>Loading pricing plans...</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '60px 20px'
    }}>
      {/* Header */}
      <div style={{ maxWidth: '1200px', margin: '0 auto 60px', textAlign: 'center' }}>
        <h1 style={{
          fontSize: '48px',
          fontWeight: 'bold',
          color: 'white',
          marginBottom: '16px'
        }}>
          Choose Your Plan
        </h1>
        <p style={{
          fontSize: '20px',
          color: 'rgba(255, 255, 255, 0.9)',
          marginBottom: '32px'
        }}>
          Start with a free trial, upgrade as you grow
        </p>

        {/* Billing Toggle */}
        <div style={{
          display: 'inline-flex',
          background: 'rgba(255, 255, 255, 0.2)',
          borderRadius: '12px',
          padding: '6px'
        }}>
          <button
            onClick={() => setBillingCycle('monthly')}
            style={{
              padding: '12px 32px',
              borderRadius: '8px',
              border: 'none',
              background: billingCycle === 'monthly' ? 'white' : 'transparent',
              color: billingCycle === 'monthly' ? '#667eea' : 'white',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.3s'
            }}
          >
            Monthly
          </button>
          <button
            onClick={() => setBillingCycle('yearly')}
            style={{
              padding: '12px 32px',
              borderRadius: '8px',
              border: 'none',
              background: billingCycle === 'yearly' ? 'white' : 'transparent',
              color: billingCycle === 'yearly' ? '#667eea' : 'white',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.3s',
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
              fontSize: '10px',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: 'bold'
            }}>
              Save 17%
            </span>
          </button>
        </div>
      </div>

      {/* Pricing Cards */}
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '24px'
      }}>
        {plans.map((plan) => {
          const price = billingCycle === 'monthly' ? plan.price_monthly : plan.price_yearly / 12;
          const yearlyTotal = plan.price_yearly;
          const isCurrentPlan = currentPlan?.plan_id === plan.id;
          const isFree = plan.price_monthly === 0;
          const isProcessing = processingPlan === plan.id;
          
          // Parse features
          let features = [];
          try {
            features = JSON.parse(plan.features || '[]');
          } catch (e) {
            features = [];
          }

          return (
            <div
              key={plan.id}
              style={{
                background: 'white',
                borderRadius: '16px',
                padding: '32px',
                boxShadow: isCurrentPlan ? '0 20px 60px rgba(0,0,0,0.3)' : '0 10px 30px rgba(0,0,0,0.2)',
                transform: isCurrentPlan ? 'scale(1.05)' : 'scale(1)',
                transition: 'all 0.3s',
                position: 'relative',
                border: isCurrentPlan ? '3px solid #10b981' : '1px solid #e5e7eb'
              }}
            >
              {/* Current Plan Badge */}
              {isCurrentPlan && (
                <div style={{
                  position: 'absolute',
                  top: '-12px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  background: '#10b981',
                  color: 'white',
                  padding: '4px 16px',
                  borderRadius: '20px',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  CURRENT PLAN
                </div>
              )}

              {/* Plan Header */}
              <div style={{ marginBottom: '24px', textAlign: 'center' }}>
                <h3 style={{
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: '#111827',
                  marginBottom: '8px'
                }}>
                  {plan.display_name}
                </h3>
                <p style={{
                  fontSize: '14px',
                  color: '#6b7280',
                  marginBottom: '16px'
                }}>
                  {plan.description}
                </p>

                {/* Price */}
                <div style={{ marginBottom: '8px' }}>
                  <span style={{
                    fontSize: '48px',
                    fontWeight: 'bold',
                    color: '#111827'
                  }}>
                    ${price.toFixed(0)}
                  </span>
                  <span style={{
                    fontSize: '16px',
                    color: '#6b7280',
                    marginLeft: '4px'
                  }}>
                    /month
                  </span>
                </div>

                {billingCycle === 'yearly' && !isFree && (
                  <p style={{
                    fontSize: '14px',
                    color: '#6b7280'
                  }}>
                    ${yearlyTotal} billed annually
                  </p>
                )}
              </div>

              {/* Features */}
              <div style={{ marginBottom: '24px' }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '12px',
                  padding: '8px 12px',
                  background: '#f3f4f6',
                  borderRadius: '8px'
                }}>
                  <Zap size={16} style={{ color: '#f59e0b', marginRight: '8px' }} />
                  <span style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                    {plan.max_scans_per_month === -1 
                      ? 'Unlimited scans' 
                      : `${plan.max_scans_per_month} scans/month`}
                  </span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '12px',
                  padding: '8px 12px',
                  background: '#f3f4f6',
                  borderRadius: '8px'
                }}>
                  <Check size={16} style={{ color: '#10b981', marginRight: '8px' }} />
                  <span style={{ fontSize: '14px', color: '#374151' }}>
                    {plan.max_users === -1 ? 'Unlimited users' : `Up to ${plan.max_users} users`}
                  </span>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '12px',
                  padding: '8px 12px',
                  background: '#f3f4f6',
                  borderRadius: '8px'
                }}>
                  <Check size={16} style={{ color: '#10b981', marginRight: '8px' }} />
                  <span style={{ fontSize: '14px', color: '#374151' }}>
                    {plan.max_patients === -1 
                      ? 'Unlimited patient records' 
                      : `Up to ${plan.max_patients} patients`}
                  </span>
                </div>

                {/* Feature List */}
                {features.slice(0, 5).map((feature, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      marginBottom: '8px'
                    }}
                  >
                    <Check size={16} style={{ color: '#10b981', marginRight: '8px', flexShrink: 0 }} />
                    <span style={{ fontSize: '14px', color: '#6b7280' }}>
                      {formatFeatureName(feature)}
                    </span>
                  </div>
                ))}
              </div>

              {/* CTA Button */}
              <button
                onClick={() => handleSubscribe(plan)}
                disabled={isCurrentPlan || isProcessing}
                style={{
                  width: '100%',
                  padding: '14px',
                  borderRadius: '8px',
                  border: 'none',
                  background: isCurrentPlan 
                    ? '#d1d5db' 
                    : isFree 
                    ? '#6b7280' 
                    : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  fontWeight: '600',
                  fontSize: '16px',
                  cursor: isCurrentPlan || isProcessing ? 'not-allowed' : 'pointer',
                  opacity: isProcessing ? 0.7 : 1,
                  transition: 'all 0.3s',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '8px'
                }}
              >
                {isProcessing ? (
                  <>
                    <div style={{
                      width: '16px',
                      height: '16px',
                      border: '2px solid rgba(255,255,255,0.3)',
                      borderTopColor: 'white',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite'
                    }} />
                    Processing...
                  </>
                ) : isCurrentPlan ? (
                  'Current Plan'
                ) : isFree ? (
                  <>
                    <Shield size={18} />
                    Start Free Trial
                  </>
                ) : (
                  <>
                    <CreditCard size={18} />
                    Subscribe Now
                  </>
                )}
              </button>

              {/* Security Badge */}
              {!isFree && (
                <div style={{
                  marginTop: '16px',
                  textAlign: 'center',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  fontSize: '12px',
                  color: '#6b7280'
                }}>
                  <Shield size={14} />
                  <span>Secure payment by Stripe</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Trust Indicators */}
      <div style={{
        maxWidth: '800px',
        margin: '60px auto 0',
        textAlign: 'center',
        color: 'white'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '40px',
          flexWrap: 'wrap'
        }}>
          <div>
            <Shield size={32} style={{ marginBottom: '8px' }} />
            <p style={{ fontSize: '14px', fontWeight: '600' }}>Secure Payments</p>
          </div>
          <div>
            <Check size={32} style={{ marginBottom: '8px' }} />
            <p style={{ fontSize: '14px', fontWeight: '600' }}>Money-Back Guarantee</p>
          </div>
          <div>
            <Zap size={32} style={{ marginBottom: '8px' }} />
            <p style={{ fontSize: '14px', fontWeight: '600' }}>Cancel Anytime</p>
          </div>
        </div>
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

function formatFeatureName(feature) {
  const nameMap = {
    'basic_scan': 'MRI Scan Analysis',
    'advanced_analytics': 'Advanced Analytics',
    'pdf_reports': 'PDF Reports',
    'patient_portal': 'Patient Portal',
    'chat_support': 'Chat Support',
    'email_support': 'Email Support',
    'api_access': 'API Access',
    'priority_support': '24/7 Priority Support',
    'gradcam_visualization': 'GradCAM Heatmaps',
    'custom_branding': 'Custom Branding',
    'dedicated_support': 'Dedicated Account Manager',
    'sla_guarantee': '99.9% Uptime SLA',
    'custom_integration': 'Custom EHR Integration'
  };
  
  return nameMap[feature] || feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}
