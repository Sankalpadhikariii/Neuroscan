import React, { useState, useEffect } from 'react';
import { 
  Users, Building2, UserCircle, LogOut, Plus, 
  Search, Edit2, Trash2, Crown, Shield, 
  CreditCard, BarChart3, Activity, AlertCircle,
  CheckCircle, XCircle, Calendar, Mail, Phone
} from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export default function AdminPortal({ user, onLogout }) {
  const [view, setView] = useState('dashboard');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Data states
  const [stats, setStats] = useState(null);
  const [admins, setAdmins] = useState([]);
  const [hospitals, setHospitals] = useState([]);
  const [patients, setPatients] = useState([]);
  const [subscriptionPlans, setSubscriptionPlans] = useState([]);
  
  // Modal states
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [userTypeToAdd, setUserTypeToAdd] = useState('admin');
  
  // Search and filters
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  useEffect(() => {
    loadDashboardData();
    loadSubscriptionPlans();
  }, []);

  useEffect(() => {
    if (view === 'admins') loadAdmins();
    else if (view === 'hospitals') loadHospitals();
    else if (view === 'patients') loadPatients();
  }, [view]);

  // Auto-dismiss messages
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  async function loadDashboardData() {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/admin/dashboard`, {
        credentials: 'include'
      });
      const data = await res.json();
      if (res.ok) {
        setStats(data.stats);
      } else {
        setError(data.error || 'Failed to load dashboard');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  }

  async function loadSubscriptionPlans() {
    try {
      const res = await fetch(`${API_BASE}/admin/subscription-plans`, {
        credentials: 'include'
      });
      const data = await res.json();
      if (res.ok) {
        setSubscriptionPlans(data.plans || []);
      }
    } catch (err) {
      console.error('Failed to load plans:', err);
    }
  }

  async function loadAdmins() {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/admin/users/admins`, {
        credentials: 'include'
      });
      const data = await res.json();
      if (res.ok) {
        setAdmins(data.admins || []);
      } else {
        setError(data.error || 'Failed to load admins');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  }

  async function loadHospitals() {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/admin/users/hospitals`, {
        credentials: 'include'
      });
      const data = await res.json();
      if (res.ok) {
        setHospitals(data.hospitals || []);
      } else {
        setError(data.error || 'Failed to load hospitals');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  }

  async function loadPatients() {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/admin/users/patients`, {
        credentials: 'include'
      });
      const data = await res.json();
      if (res.ok) {
        setPatients(data.patients || []);
      } else {
        setError(data.error || 'Failed to load patients');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  }

  function openAddModal(userType) {
    setUserTypeToAdd(userType);
    setShowAddModal(true);
  }

  function openEditModal(user, userType) {
    setSelectedUser({ ...user, userType });
    setShowEditModal(true);
  }

  function openDeleteModal(user, userType) {
    setSelectedUser({ ...user, userType });
    setShowDeleteModal(true);
  }

  const filteredData = (data) => {
    if (!searchQuery && statusFilter === 'all') return data;
    
    return data.filter(item => {
      const matchesSearch = searchQuery === '' || 
        Object.values(item).some(val => 
          val && val.toString().toLowerCase().includes(searchQuery.toLowerCase())
        );
      
      const matchesStatus = statusFilter === 'all' || 
        item.status === statusFilter ||
        (item.subscription_status && item.subscription_status === statusFilter);
      
      return matchesSearch && matchesStatus;
    });
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    }}>
      {/* Header */}
      <header style={{
        background: 'rgba(255, 255, 255, 0.95)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(0,0,0,0.1)',
        padding: '20px 40px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <div style={{
          maxWidth: '1400px',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '48px',
              height: '48px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Shield size={28} color="white" />
            </div>
            <div>
              <h1 style={{
                margin: 0,
                fontSize: '24px',
                fontWeight: 'bold',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}>
                Admin Portal
              </h1>
              <p style={{ margin: 0, fontSize: '14px', color: '#6b7280' }}>
                System Management & User Control
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{ textAlign: 'right' }}>
              <p style={{ margin: 0, fontSize: '14px', fontWeight: '600', color: '#111827' }}>
                {user?.username || user?.email}
              </p>
              <p style={{ margin: 0, fontSize: '12px', color: '#6b7280' }}>
                {user?.role === 'superadmin' ? 'Super Administrator' : 'Administrator'}
              </p>
            </div>
            <button
              onClick={onLogout}
              style={{
                padding: '10px 20px',
                background: '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => e.target.style.background = '#dc2626'}
              onMouseLeave={(e) => e.target.style.background = '#ef4444'}
            >
              <LogOut size={18} />
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <div style={{
        background: 'white',
        borderBottom: '1px solid rgba(0,0,0,0.1)',
        padding: '0 40px'
      }}>
        <div style={{
          maxWidth: '1400px',
          margin: '0 auto',
          display: 'flex',
          gap: '8px'
        }}>
          {[
            { id: 'dashboard', icon: BarChart3, label: 'Dashboard' },
            { id: 'admins', icon: Shield, label: 'Admins' },
            { id: 'hospitals', icon: Building2, label: 'Hospitals' },
            { id: 'patients', icon: Users, label: 'Patients' }
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = view === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setView(tab.id)}
                style={{
                  padding: '16px 24px',
                  background: isActive ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'transparent',
                  color: isActive ? 'white' : '#6b7280',
                  border: 'none',
                  borderBottom: isActive ? '3px solid transparent' : 'none',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '14px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  transition: 'all 0.2s',
                  borderRadius: '8px 8px 0 0'
                }}
              >
                <Icon size={18} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Content */}
      <div style={{
        maxWidth: '1400px',
        margin: '0 auto',
        padding: '40px'
      }}>
        {/* Alerts */}
        {error && (
          <div style={{
            background: '#fee2e2',
            border: '1px solid #fecaca',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'start',
            gap: '12px'
          }}>
            <AlertCircle size={20} color="#dc2626" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div style={{ flex: 1 }}>
              <p style={{ margin: 0, color: '#991b1b', fontWeight: '600' }}>Error</p>
              <p style={{ margin: '4px 0 0 0', color: '#7f1d1d', fontSize: '14px' }}>{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}
            >
              <XCircle size={20} color="#dc2626" />
            </button>
          </div>
        )}

        {success && (
          <div style={{
            background: '#d1fae5',
            border: '1px solid #a7f3d0',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'start',
            gap: '12px'
          }}>
            <CheckCircle size={20} color="#059669" style={{ flexShrink: 0, marginTop: '2px' }} />
            <div style={{ flex: 1 }}>
              <p style={{ margin: 0, color: '#065f46', fontWeight: '600' }}>Success</p>
              <p style={{ margin: '4px 0 0 0', color: '#047857', fontSize: '14px' }}>{success}</p>
            </div>
            <button
              onClick={() => setSuccess(null)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '4px' }}
            >
              <XCircle size={20} color="#059669" />
            </button>
          </div>
        )}

        {/* Dashboard View */}
        {view === 'dashboard' && (
          <DashboardView stats={stats} loading={loading} />
        )}

        {/* Admins View */}
        {view === 'admins' && (
          <UsersView
            title="System Administrators"
            icon={Shield}
            users={filteredData(admins)}
            userType="admin"
            loading={loading}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            onAdd={() => openAddModal('admin')}
            onEdit={(user) => openEditModal(user, 'admin')}
            onDelete={(user) => openDeleteModal(user, 'admin')}
          />
        )}

        {/* Hospitals View */}
        {view === 'hospitals' && (
          <UsersView
            title="Hospital Accounts"
            icon={Building2}
            users={filteredData(hospitals)}
            userType="hospital"
            loading={loading}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            statusFilter={statusFilter}
            setStatusFilter={setStatusFilter}
            onAdd={() => openAddModal('hospital')}
            onEdit={(user) => openEditModal(user, 'hospital')}
            onDelete={(user) => openDeleteModal(user, 'hospital')}
            showSubscription={true}
          />
        )}

        {/* Patients View */}
        {view === 'patients' && (
          <UsersView
            title="Patient Records"
            icon={Users}
            users={filteredData(patients)}
            userType="patient"
            loading={loading}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            onAdd={() => openAddModal('patient')}
            onEdit={(user) => openEditModal(user, 'patient')}
            onDelete={(user) => openDeleteModal(user, 'patient')}
          />
        )}
      </div>

      {/* Modals */}
      {showAddModal && (
        <AddUserModal
          userType={userTypeToAdd}
          subscriptionPlans={subscriptionPlans}
          hospitals={hospitals}
          onClose={() => setShowAddModal(false)}
          onSuccess={(message) => {
            setSuccess(message);
            setShowAddModal(false);
            if (userTypeToAdd === 'admin') loadAdmins();
            else if (userTypeToAdd === 'hospital') loadHospitals();
            else if (userTypeToAdd === 'patient') loadPatients();
            loadDashboardData();
          }}
          onError={setError}
        />
      )}

      {showEditModal && selectedUser && (
        <EditUserModal
          user={selectedUser}
          subscriptionPlans={subscriptionPlans}
          hospitals={hospitals}
          onClose={() => {
            setShowEditModal(false);
            setSelectedUser(null);
          }}
          onSuccess={(message) => {
            setSuccess(message);
            setShowEditModal(false);
            setSelectedUser(null);
            if (selectedUser.userType === 'admin') loadAdmins();
            else if (selectedUser.userType === 'hospital') loadHospitals();
            else if (selectedUser.userType === 'patient') loadPatients();
            loadDashboardData();
          }}
          onError={setError}
        />
      )}

      {showDeleteModal && selectedUser && (
        <DeleteUserModal
          user={selectedUser}
          onClose={() => {
            setShowDeleteModal(false);
            setSelectedUser(null);
          }}
          onSuccess={(message) => {
            setSuccess(message);
            setShowDeleteModal(false);
            setSelectedUser(null);
            if (selectedUser.userType === 'admin') loadAdmins();
            else if (selectedUser.userType === 'hospital') loadHospitals();
            else if (selectedUser.userType === 'patient') loadPatients();
            loadDashboardData();
          }}
          onError={setError}
        />
      )}
    </div>
  );
}

// Dashboard View Component
function DashboardView({ stats, loading }) {
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '60px' }}>
        <Activity size={48} color="#667eea" style={{ animation: 'pulse 2s infinite' }} />
        <p style={{ marginTop: '16px', color: 'white', fontSize: '18px' }}>Loading dashboard...</p>
      </div>
    );
  }

  const statCards = [
    {
      title: 'Total Admins',
      value: stats?.total_admins || 0,
      icon: Shield,
      color: '#8b5cf6',
      bgColor: '#f3e8ff'
    },
    {
      title: 'Total Hospitals',
      value: stats?.total_hospitals || 0,
      icon: Building2,
      color: '#3b82f6',
      bgColor: '#dbeafe'
    },
    {
      title: 'Total Patients',
      value: stats?.total_patients || 0,
      icon: Users,
      color: '#10b981',
      bgColor: '#d1fae5'
    },
    {
      title: 'Active Subscriptions',
      value: stats?.active_subscriptions || 0,
      icon: CreditCard,
      color: '#f59e0b',
      bgColor: '#fef3c7'
    }
  ];

  return (
    <div>
      <h2 style={{
        fontSize: '28px',
        fontWeight: 'bold',
        color: 'white',
        marginBottom: '24px'
      }}>
        System Overview
      </h2>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '24px',
        marginBottom: '40px'
      }}>
        {statCards.map((card, index) => {
          const Icon = card.icon;
          return (
            <div
              key={index}
              style={{
                background: 'white',
                borderRadius: '16px',
                padding: '24px',
                boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                transition: 'transform 0.2s',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                <div>
                  <p style={{
                    margin: 0,
                    fontSize: '14px',
                    color: '#6b7280',
                    fontWeight: '600'
                  }}>
                    {card.title}
                  </p>
                  <p style={{
                    margin: '8px 0 0 0',
                    fontSize: '32px',
                    fontWeight: 'bold',
                    color: '#111827'
                  }}>
                    {card.value}
                  </p>
                </div>
                <div style={{
                  width: '48px',
                  height: '48px',
                  background: card.bgColor,
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Icon size={24} color={card.color} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Recent Activity */}
      <div style={{
        background: 'white',
        borderRadius: '16px',
        padding: '24px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <h3 style={{
          margin: '0 0 20px 0',
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#111827'
        }}>
          System Statistics
        </h3>
        
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '16px'
        }}>
          <StatItem label="Total Scans" value={stats?.total_scans || 0} />
          <StatItem label="Scans Today" value={stats?.scans_today || 0} />
          <StatItem label="Scans This Month" value={stats?.scans_this_month || 0} />
          <StatItem label="Active Users" value={stats?.active_users || 0} />
        </div>
      </div>
    </div>
  );
}

function StatItem({ label, value }) {
  return (
    <div style={{
      padding: '16px',
      background: '#f9fafb',
      borderRadius: '8px',
      borderLeft: '4px solid #667eea'
    }}>
      <p style={{ margin: 0, fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>
        {label}
      </p>
      <p style={{ margin: '4px 0 0 0', fontSize: '24px', fontWeight: 'bold', color: '#111827' }}>
        {value}
      </p>
    </div>
  );
}

// Users View Component
function UsersView({
  title,
  icon: Icon,
  users,
  userType,
  loading,
  searchQuery,
  setSearchQuery,
  statusFilter,
  setStatusFilter,
  onAdd,
  onEdit,
  onDelete,
  showSubscription = false
}) {
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '60px' }}>
        <Activity size={48} color="#667eea" style={{ animation: 'pulse 2s infinite' }} />
        <p style={{ marginTop: '16px', color: 'white', fontSize: '18px' }}>Loading {title.toLowerCase()}...</p>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '48px',
            height: '48px',
            background: 'white',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Icon size={24} color="#667eea" />
          </div>
          <div>
            <h2 style={{
              margin: 0,
              fontSize: '28px',
              fontWeight: 'bold',
              color: 'white'
            }}>
              {title}
            </h2>
            <p style={{ margin: '4px 0 0 0', color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
              {users.length} total records
            </p>
          </div>
        </div>

        <button
          onClick={onAdd}
          style={{
            padding: '12px 24px',
            background: 'white',
            color: '#667eea',
            border: 'none',
            borderRadius: '8px',
            fontWeight: '600',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            e.target.style.background = '#667eea';
            e.target.style.color = 'white';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = 'white';
            e.target.style.color = '#667eea';
          }}
        >
          <Plus size={18} />
          Add {userType === 'admin' ? 'Admin' : userType === 'hospital' ? 'Hospital' : 'Patient'}
        </button>
      </div>

      {/* Search and Filters */}
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '20px',
        marginBottom: '20px',
        display: 'flex',
        gap: '12px',
        flexWrap: 'wrap'
      }}>
        <div style={{ flex: '1 1 300px', position: 'relative' }}>
          <Search
            size={20}
            color="#9ca3af"
            style={{
              position: 'absolute',
              left: '12px',
              top: '50%',
              transform: 'translateY(-50%)'
            }}
          />
          <input
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: '100%',
              padding: '12px 12px 12px 44px',
              border: '2px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '14px',
              outline: 'none',
              transition: 'border-color 0.2s'
            }}
            onFocus={(e) => e.target.style.borderColor = '#667eea'}
            onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
          />
        </div>

        {showSubscription && (
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{
              padding: '12px 16px',
              border: '2px solid #e5e7eb',
              borderRadius: '8px',
              fontSize: '14px',
              outline: 'none',
              cursor: 'pointer',
              background: 'white'
            }}
          >
            <option value="all">All Status</option>
            <option value="active">Active</option>
            <option value="trial">Trial</option>
            <option value="expired">Expired</option>
            <option value="cancelled">Cancelled</option>
          </select>
        )}
      </div>

      {/* Users Table */}
      <div style={{
        background: 'white',
        borderRadius: '16px',
        overflow: 'hidden',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        {users.length === 0 ? (
          <div style={{
            padding: '60px',
            textAlign: 'center'
          }}>
            <Icon size={48} color="#d1d5db" />
            <p style={{
              margin: '16px 0 0 0',
              color: '#6b7280',
              fontSize: '16px'
            }}>
              No {title.toLowerCase()} found
            </p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse'
            }}>
              <thead>
                <tr style={{ background: '#f9fafb', borderBottom: '2px solid #e5e7eb' }}>
                  <th style={tableHeaderStyle}>ID</th>
                  {userType === 'admin' && <th style={tableHeaderStyle}>Username</th>}
                  {userType === 'hospital' && <th style={tableHeaderStyle}>Hospital Name</th>}
                  {userType === 'patient' && <th style={tableHeaderStyle}>Patient Name</th>}
                  <th style={tableHeaderStyle}>Email</th>
                  {userType === 'hospital' && <th style={tableHeaderStyle}>Hospital Code</th>}
                  {userType === 'patient' && <th style={tableHeaderStyle}>Patient Code</th>}
                  {showSubscription && <th style={tableHeaderStyle}>Subscription</th>}
                  <th style={tableHeaderStyle}>Created</th>
                  <th style={tableHeaderStyle}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user, index) => (
                  <tr
                    key={user.id}
                    style={{
                      borderBottom: '1px solid #f3f4f6',
                      background: index % 2 === 0 ? 'white' : '#fafbfc'
                    }}
                  >
                    <td style={tableCellStyle}>#{user.id}</td>
                    <td style={tableCellStyle}>
                      {user.username || user.hospital_name || user.full_name}
                    </td>
                    <td style={tableCellStyle}>{user.email}</td>
                    {(userType === 'hospital' || userType === 'patient') && (
                      <td style={tableCellStyle}>
                        <code style={{
                          background: '#f3f4f6',
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontFamily: 'monospace'
                        }}>
                          {user.hospital_code || user.patient_code}
                        </code>
                      </td>
                    )}
                    {showSubscription && (
                      <td style={tableCellStyle}>
                        <span style={{
                          padding: '4px 12px',
                          borderRadius: '12px',
                          fontSize: '12px',
                          fontWeight: '600',
                          background: user.subscription_status === 'active' ? '#d1fae5' :
                                     user.subscription_status === 'trial' ? '#fef3c7' :
                                     '#fee2e2',
                          color: user.subscription_status === 'active' ? '#065f46' :
                                 user.subscription_status === 'trial' ? '#92400e' :
                                 '#991b1b'
                        }}>
                          {user.subscription_plan || 'Free'} ({user.subscription_status || 'N/A'})
                        </span>
                      </td>
                    )}
                    <td style={tableCellStyle}>
                      {new Date(user.created_at).toLocaleDateString()}
                    </td>
                    <td style={tableCellStyle}>
                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          onClick={() => onEdit(user)}
                          style={{
                            padding: '6px 12px',
                            background: '#3b82f6',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}
                        >
                          <Edit2 size={14} />
                          Edit
                        </button>
                        <button
                          onClick={() => onDelete(user)}
                          style={{
                            padding: '6px 12px',
                            background: '#ef4444',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px'
                          }}
                        >
                          <Trash2 size={14} />
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

const tableHeaderStyle = {
  padding: '16px',
  textAlign: 'left',
  fontSize: '12px',
  fontWeight: '700',
  color: '#6b7280',
  textTransform: 'uppercase',
  letterSpacing: '0.5px'
};

const tableCellStyle = {
  padding: '16px',
  fontSize: '14px',
  color: '#374151'
};

// Add User Modal Component (continued in next part)
function AddUserModal({ userType, subscriptionPlans, hospitals, onClose, onSuccess, onError }) {
  const [formData, setFormData] = useState({
    // Common fields
    email: '',
    password: '',
    
    // Admin fields
    username: '',
    role: 'admin',
    
    // Hospital fields
    hospital_name: '',
    hospital_code: '',
    contact_person: '',
    phone: '',
    address: '',
    subscription_plan: 'free',
    
    // Patient fields
    full_name: '',
    patient_code: '',
    access_code: '',
    hospital_id: '',
    date_of_birth: '',
    gender: '',
    blood_group: '',
    emergency_contact: '',
    medical_history: ''
  });

  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/admin/users/${userType}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(formData)
      });

      const data = await res.json();

      if (res.ok) {
        onSuccess(data.message || `${userType} created successfully`);
      } else {
        onError(data.error || 'Failed to create user');
      }
    } catch (err) {
      onError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      zIndex: 1000
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        maxWidth: '600px',
        width: '100%',
        maxHeight: '90vh',
        overflow: 'auto',
        boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)'
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '24px',
          borderBottom: '1px solid #e5e7eb',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h3 style={{
            margin: 0,
            fontSize: '20px',
            fontWeight: 'bold',
            color: '#111827'
          }}>
            Add New {userType === 'admin' ? 'Administrator' : userType === 'hospital' ? 'Hospital' : 'Patient'}
          </h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '4px'
            }}
          >
            <XCircle size={24} color="#9ca3af" />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} style={{ padding: '24px' }}>
          {userType === 'admin' && (
            <>
              <FormField
                label="Username"
                type="text"
                required
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              />
              <FormField
                label="Email"
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
              <FormField
                label="Password"
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
              <FormSelect
                label="Role"
                required
                value={formData.role}
                onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                options={[
                  { value: 'admin', label: 'Admin' },
                  { value: 'superadmin', label: 'Super Admin' }
                ]}
              />
            </>
          )}

          {userType === 'hospital' && (
            <>
              <FormField
                label="Hospital Name"
                type="text"
                required
                value={formData.hospital_name}
                onChange={(e) => setFormData({ ...formData, hospital_name: e.target.value })}
              />
              <FormField
                label="Hospital Code"
                type="text"
                required
                placeholder="e.g., HOSP001"
                value={formData.hospital_code}
                onChange={(e) => setFormData({ ...formData, hospital_code: e.target.value.toUpperCase() })}
              />
              <FormField
                label="Email"
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
              <FormField
                label="Password"
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
              <FormField
                label="Contact Person"
                type="text"
                value={formData.contact_person}
                onChange={(e) => setFormData({ ...formData, contact_person: e.target.value })}
              />
              <FormField
                label="Phone"
                type="tel"
                value={formData.phone}
                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
              />
              <FormField
                label="Address"
                type="text"
                value={formData.address}
                onChange={(e) => setFormData({ ...formData, address: e.target.value })}
              />
              <FormSelect
                label="Subscription Plan"
                required
                value={formData.subscription_plan}
                onChange={(e) => setFormData({ ...formData, subscription_plan: e.target.value })}
                options={subscriptionPlans.map(plan => ({
                  value: plan.name,
                  label: `${plan.display_name} - $${plan.price_monthly}/month`
                }))}
              />
            </>
          )}

          {userType === 'patient' && (
            <>
              <FormField
                label="Full Name"
                type="text"
                required
                value={formData.full_name}
                onChange={(e) => setFormData({ ...formData, full_name: e.target.value })}
              />
              <FormField
                label="Patient Code"
                type="text"
                required
                placeholder="e.g., PAT001"
                value={formData.patient_code}
                onChange={(e) => setFormData({ ...formData, patient_code: e.target.value.toUpperCase() })}
              />
              <FormField
                label="Access Code"
                type="text"
                required
                placeholder="6-digit code"
                value={formData.access_code}
                onChange={(e) => setFormData({ ...formData, access_code: e.target.value })}
              />
              <FormField
                label="Email"
                type="email"
                required
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
              <FormSelect
                label="Hospital"
                required
                value={formData.hospital_id}
                onChange={(e) => setFormData({ ...formData, hospital_id: e.target.value })}
                options={[
                  { value: '', label: 'Select Hospital' },
                  ...hospitals.map(h => ({
                    value: h.id,
                    label: `${h.hospital_name} (${h.hospital_code})`
                  }))
                ]}
              />
              <FormField
                label="Date of Birth"
                type="date"
                value={formData.date_of_birth}
                onChange={(e) => setFormData({ ...formData, date_of_birth: e.target.value })}
              />
              <FormSelect
                label="Gender"
                value={formData.gender}
                onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                options={[
                  { value: '', label: 'Select Gender' },
                  { value: 'male', label: 'Male' },
                  { value: 'female', label: 'Female' },
                  { value: 'other', label: 'Other' }
                ]}
              />
              <FormField
                label="Phone"
                type="tel"
                value={formData.phone}
                onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
              />
              <FormField
                label="Emergency Contact"
                type="tel"
                value={formData.emergency_contact}
                onChange={(e) => setFormData({ ...formData, emergency_contact: e.target.value })}
              />
            </>
          )}

          {/* Submit Button */}
          <div style={{
            marginTop: '24px',
            display: 'flex',
            gap: '12px',
            justifyContent: 'flex-end'
          }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '12px 24px',
                background: '#f3f4f6',
                color: '#374151',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '12px 24px',
                background: loading ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              {loading ? 'Creating...' : 'Create User'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// Edit and Delete modals would follow similar patterns...
// (Due to length constraints, I'll create simplified versions)

function EditUserModal({ user, subscriptionPlans, hospitals, onClose, onSuccess, onError }) {
  const [formData, setFormData] = useState(user);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/admin/users/${user.userType}/${user.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(formData)
      });

      const data = await res.json();

      if (res.ok) {
        onSuccess(data.message || 'User updated successfully');
      } else {
        onError(data.error || 'Failed to update user');
      }
    } catch (err) {
      onError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      zIndex: 1000
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        maxWidth: '600px',
        width: '100%',
        maxHeight: '90vh',
        overflow: 'auto'
      }}>
        <div style={{
          padding: '24px',
          borderBottom: '1px solid #e5e7eb',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h3 style={{ margin: 0, fontSize: '20px', fontWeight: 'bold' }}>
            Edit User
          </h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>
            <XCircle size={24} color="#9ca3af" />
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ padding: '24px' }}>
          {user.userType === 'hospital' && subscriptionPlans.length > 0 && (
            <FormSelect
              label="Subscription Plan"
              value={formData.subscription_plan || 'free'}
              onChange={(e) => setFormData({ ...formData, subscription_plan: e.target.value })}
              options={subscriptionPlans.map(plan => ({
                value: plan.name,
                label: `${plan.display_name} - $${plan.price_monthly}/month`
              }))}
            />
          )}

          <FormField
            label="Email"
            type="email"
            value={formData.email || ''}
            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          />

          <div style={{ marginTop: '24px', display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '12px 24px',
                background: '#f3f4f6',
                color: '#374151',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '12px 24px',
                background: loading ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontWeight: '600',
                cursor: loading ? 'not-allowed' : 'pointer'
              }}
            >
              {loading ? 'Updating...' : 'Update User'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function DeleteUserModal({ user, onClose, onSuccess, onError }) {
  const [loading, setLoading] = useState(false);

  const handleDelete = async () => {
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/admin/users/${user.userType}/${user.id}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      const data = await res.json();

      if (res.ok) {
        onSuccess(data.message || 'User deleted successfully');
      } else {
        onError(data.error || 'Failed to delete user');
      }
    } catch (err) {
      onError('Failed to connect to server');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      zIndex: 1000
    }}>
      <div style={{
        background: 'white',
        borderRadius: '16px',
        maxWidth: '500px',
        width: '100%',
        padding: '24px'
      }}>
        <div style={{
          width: '64px',
          height: '64px',
          background: '#fee2e2',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 20px'
        }}>
          <AlertCircle size={32} color="#dc2626" />
        </div>

        <h3 style={{
          margin: '0 0 12px 0',
          fontSize: '20px',
          fontWeight: 'bold',
          textAlign: 'center'
        }}>
          Delete User?
        </h3>

        <p style={{
          margin: '0 0 24px 0',
          color: '#6b7280',
          textAlign: 'center',
          fontSize: '14px'
        }}>
          Are you sure you want to delete this user? This action cannot be undone.
        </p>

        <div style={{
          background: '#f9fafb',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '24px'
        }}>
          <p style={{ margin: '0 0 8px 0', fontSize: '12px', color: '#6b7280', fontWeight: '600' }}>
            User Details:
          </p>
          <p style={{ margin: 0, fontSize: '14px', color: '#111827' }}>
            <strong>{user.username || user.hospital_name || user.full_name}</strong>
          </p>
          <p style={{ margin: '4px 0 0 0', fontSize: '14px', color: '#6b7280' }}>
            {user.email}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={onClose}
            style={{
              flex: 1,
              padding: '12px',
              background: '#f3f4f6',
              color: '#374151',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={loading}
            style={{
              flex: 1,
              padding: '12px',
              background: loading ? '#9ca3af' : '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer'
            }}
          >
            {loading ? 'Deleting...' : 'Delete User'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Form Components
function FormField({ label, type = 'text', required = false, value, onChange, placeholder }) {
  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{
        display: 'block',
        marginBottom: '6px',
        fontSize: '14px',
        fontWeight: '600',
        color: '#374151'
      }}>
        {label} {required && <span style={{ color: '#ef4444' }}>*</span>}
      </label>
      <input
        type={type}
        required={required}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        style={{
          width: '100%',
          padding: '10px 12px',
          border: '2px solid #e5e7eb',
          borderRadius: '8px',
          fontSize: '14px',
          outline: 'none',
          transition: 'border-color 0.2s'
        }}
        onFocus={(e) => e.target.style.borderColor = '#667eea'}
        onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
      />
    </div>
  );
}

function FormSelect({ label, required = false, value, onChange, options }) {
  return (
    <div style={{ marginBottom: '16px' }}>
      <label style={{
        display: 'block',
        marginBottom: '6px',
        fontSize: '14px',
        fontWeight: '600',
        color: '#374151'
      }}>
        {label} {required && <span style={{ color: '#ef4444' }}>*</span>}
      </label>
      <select
        required={required}
        value={value}
        onChange={onChange}
        style={{
          width: '100%',
          padding: '10px 12px',
          border: '2px solid #e5e7eb',
          borderRadius: '8px',
          fontSize: '14px',
          outline: 'none',
          cursor: 'pointer',
          background: 'white'
        }}
      >
        {options.map(opt => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}