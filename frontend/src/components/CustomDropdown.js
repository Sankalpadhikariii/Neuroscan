import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check } from 'lucide-react';

const CustomDropdown = ({ 
  options = [], 
  value, 
  onChange, 
  placeholder = "Select option", 
  label,
  darkMode = false,
  fullWidth = true,
  style = {},
  variant = 'default' // 'default' or 'glass'
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedOption = options.find(opt => String(opt.value) === String(value));

  const handleToggle = () => setIsOpen(!isOpen);

  const handleSelect = (optionValue) => {
    onChange({ target: { name: dropdownRef.current?.getAttribute('name') || '', value: optionValue } });
    setIsOpen(false);
  };

  // Styles based on dark mode and variant
  const containerStyle = {
    position: 'relative',
    width: fullWidth ? '100%' : 'auto',
    ...style
  };

  const triggerStyle = {
    width: '100%',
    padding: '14px 20px',
    borderRadius: '12px',
    border: darkMode 
      ? '1px solid rgba(255,255,255,0.1)' 
      : '1px solid #e2e8f0',
    background: darkMode 
      ? (variant === 'glass' ? 'rgba(255,255,255,0.05)' : '#1e293b')
      : (variant === 'glass' ? 'rgba(255,255,255,0.8)' : '#ffffff'),
    backdropFilter: variant === 'glass' ? 'blur(10px)' : 'none',
    color: darkMode ? '#ffffff' : '#1e293b',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    cursor: 'pointer',
    fontSize: '15px',
    fontWeight: '500',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    boxShadow: isOpen ? '0 0 0 2px rgba(37, 99, 235, 0.2)' : 'none',
  };

  const menuStyle = {
    position: 'absolute',
    top: 'calc(100% + 8px)',
    left: 0,
    right: 0,
    background: darkMode ? '#1e293b' : '#ffffff',
    border: darkMode ? '1px solid #334155' : '1px solid #e2e8f0',
    borderRadius: '12px',
    boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
    zIndex: 1000,
    maxHeight: '250px',
    overflowY: 'auto',
    display: isOpen ? 'block' : 'none',
    padding: '6px',
    animation: 'dropdownRotate 0.2s ease-out'
  };

  const optionItemStyle = (isSelected, isHovered) => ({
    padding: '10px 14px',
    borderRadius: '8px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: isSelected 
      ? (darkMode ? 'rgba(37, 99, 235, 0.2)' : '#eff6ff')
      : 'transparent',
    color: isSelected 
      ? '#2563eb' 
      : (darkMode ? '#e2e8f0' : '#475569'),
    fontSize: '14px',
    fontWeight: isSelected ? '600' : '500',
    transition: 'background-color 0.15s ease',
    marginBottom: '2px'
  });

  return (
    <div ref={dropdownRef} style={containerStyle}>
      {label && (
        <label style={{ 
          display: 'block', 
          marginBottom: '8px', 
          fontSize: '14px', 
          fontWeight: '600', 
          color: darkMode ? '#94a3b8' : '#64748b' 
        }}>
          {label}
        </label>
      )}
      
      <div 
        style={triggerStyle} 
        onClick={handleToggle}
        onMouseEnter={(e) => {
          if (!isOpen) e.currentTarget.style.borderColor = '#2563eb';
        }}
        onMouseLeave={(e) => {
          if (!isOpen) e.currentTarget.style.borderColor = darkMode ? 'rgba(255,255,255,0.1)' : '#e2e8f0';
        }}
      >
        <span style={{ opacity: selectedOption ? 1 : 0.6 }}>
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <ChevronDown 
          size={18} 
          style={{ 
            transform: isOpen ? 'rotate(180deg)' : 'rotate(0)', 
            transition: 'transform 0.2s',
            color: darkMode ? '#94a3b8' : '#64748b'
          }} 
        />
      </div>

      <div className="custom-dropdown-menu" style={menuStyle}>
        {options.length > 0 ? options.map((option) => {
          const isSelected = String(option.value) === String(value);
          return (
            <div
              key={option.value}
              onClick={() => handleSelect(option.value)}
              style={optionItemStyle(isSelected)}
              onMouseEnter={(e) => {
                if (!isSelected) e.currentTarget.style.backgroundColor = darkMode ? '#334155' : '#f8fafc';
              }}
              onMouseLeave={(e) => {
                if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              <span>{option.label}</span>
              {isSelected && <Check size={16} />}
            </div>
          );
        }) : (
          <div style={{ padding: '10px 14px', color: '#94a3b8', fontSize: '14px', textAlign: 'center' }}>
            No options available
          </div>
        )}
      </div>

      <style>{`
        @keyframes dropdownRotate {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .custom-dropdown-menu::-webkit-scrollbar {
          width: 5px;
        }
        .custom-dropdown-menu::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-dropdown-menu::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 10px;
        }
        ${darkMode ? '.custom-dropdown-menu::-webkit-scrollbar-thumb { background: #475569; }' : ''}
      `}</style>
    </div>
  );
};

export default CustomDropdown;
