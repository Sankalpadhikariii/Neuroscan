import React, { useState, useMemo } from 'react';
import { ChevronLeft, ChevronRight, Clock, User, Calendar as CalendarIcon } from 'lucide-react';

const AppointmentCalendar = ({ appointments, darkMode }) => {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(new Date());

  const daysInMonth = (year, month) => new Date(year, month + 1, 0).getDate();
  const firstDayOfMonth = (year, month) => new Date(year, month, 1).getDay();

  const monthNames = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ];

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();

  const handlePrevMonth = () => {
    setCurrentDate(new Date(year, month - 1, 1));
  };

  const handleNextMonth = () => {
    setCurrentDate(new Date(year, month + 1, 1));
  };

  const isToday = (day) => {
    const today = new Date();
    return today.getDate() === day && today.getMonth() === month && today.getFullYear() === year;
  };

  const isSelected = (day) => {
    return selectedDate.getDate() === day && selectedDate.getMonth() === month && selectedDate.getFullYear() === year;
  };

  const getAppointmentsForDate = (date) => {
    if (!appointments) return [];
    return appointments.filter(app => {
      const appDate = new Date(app.appointment_date);
      return appDate.getDate() === date.getDate() &&
             appDate.getMonth() === date.getMonth() &&
             appDate.getFullYear() === date.getFullYear();
    });
  };

  const selectedAppointments = useMemo(() => {
    return getAppointmentsForDate(selectedDate);
  }, [selectedDate, appointments]);

  const textColor = darkMode ? '#f1f5f9' : '#1e293b';
  const textSecondary = darkMode ? '#94a3b8' : '#64748b';
  const borderColor = darkMode ? '#334155' : '#e2e8f0';
  const cardBg = darkMode ? '#1e293b' : '#ffffff';

  return (
    <div style={{
      background: cardBg,
      borderRadius: '24px',
      padding: '24px',
      border: `1px solid ${borderColor}`,
      boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      gap: '20px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: textColor, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <CalendarIcon size={20} color="#2563eb" />
          Schedule
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '14px', fontWeight: '600', color: textSecondary }}>
            {monthNames[month]} {year}
          </span>
          <div style={{ display: 'flex', gap: '4px' }}>
            <button onClick={handlePrevMonth} style={{ padding: '4px', background: 'transparent', border: 'none', cursor: 'pointer', color: textSecondary }}>
              <ChevronLeft size={20} />
            </button>
            <button onClick={handleNextMonth} style={{ padding: '4px', background: 'transparent', border: 'none', cursor: 'pointer', color: textSecondary }}>
              <ChevronRight size={20} />
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '8px', textAlign: 'center' }}>
        {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map(day => (
          <div key={day} style={{ fontSize: '12px', fontWeight: '700', color: textSecondary, paddingBottom: '8px' }}>
            {day}
          </div>
        ))}
        {Array.from({ length: firstDayOfMonth(year, month) }).map((_, i) => (
          <div key={`empty-${i}`} />
        ))}
        {Array.from({ length: daysInMonth(year, month) }).map((_, i) => {
          const day = i + 1;
          const dateObj = new Date(year, month, day);
          const apps = getAppointmentsForDate(dateObj);
          
          return (
            <div 
              key={day}
              onClick={() => setSelectedDate(dateObj)}
              style={{
                aspectRatio: '1',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                borderRadius: '12px',
                fontSize: '14px',
                fontWeight: isSelected(day) ? '700' : '500',
                background: isSelected(day) ? '#2563eb' : (isToday(day) ? (darkMode ? '#334155' : '#f1f5f9') : 'transparent'),
                color: isSelected(day) ? 'white' : textColor,
                position: 'relative',
                transition: 'all 0.2s'
              }}
            >
              {day}
              {apps.length > 0 && (
                <div style={{ display: 'flex', gap: '2px', position: 'absolute', bottom: '6px' }}>
                  {apps.slice(0, 3).map((_, idx) => (
                    <div key={idx} style={{ 
                      width: '4px', 
                      height: '4px', 
                      borderRadius: '50%', 
                      background: isSelected(day) ? 'rgba(255,255,255,0.8)' : '#2563eb' 
                    }} />
                  ))}
                  {apps.length > 3 && <div style={{ fontSize: '6px', color: isSelected(day) ? 'white' : '#2563eb' }}>+</div>}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div style={{ 
        marginTop: '10px', 
        paddingTop: '20px', 
        borderTop: `1px solid ${borderColor}`,
        flex: 1,
        overflowY: 'auto',
        minHeight: '120px'
      }}>
        <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', fontWeight: '700', color: textColor }}>
          {selectedDate.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}
        </h4>
        {selectedAppointments.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {selectedAppointments.map((app, idx) => (
              <div key={idx} style={{ 
                padding: '12px', 
                background: darkMode ? '#0f172a' : '#f8fafc',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px'
              }}>
                <div style={{ 
                  width: '32px', 
                  height: '32px', 
                  borderRadius: '10px', 
                  background: '#2563eb22', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: '#2563eb'
                }}>
                  <User size={16} />
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{ margin: 0, fontSize: '13px', fontWeight: '700', color: textColor }}>
                    {app.patient_name || app.doctor_name || "Patient"}
                  </p>
                  <p style={{ margin: 0, fontSize: '11px', color: textSecondary, display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Clock size={10} /> {app.appointment_time}
                  </p>
                </div>
                <div style={{
                  fontSize: '10px',
                  fontWeight: '700',
                  padding: '4px 8px',
                  borderRadius: '6px',
                  background: app.status === 'scheduled' ? '#dcfce7' : '#f1f5f9',
                  color: app.status === 'scheduled' ? '#166534' : textSecondary,
                  textTransform: 'capitalize'
                }}>
                  {app.status}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ margin: 0, fontSize: '13px', color: textSecondary, textAlign: 'center', padding: '20px 0' }}>
            No appointments scheduled
          </p>
        )}
      </div>
    </div>
  );
};

export default AppointmentCalendar;
