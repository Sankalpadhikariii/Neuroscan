import React, { useState, useEffect, useRef } from 'react';
import { 
  Brain, 
  Shield, 
  Zap, 
  Check, 
  X, 
  ArrowRight, 
  Activity, 
  Microscope, 
  Building2, 
  User, 
  Loader, 
  ChevronDown,
  Globe,
  Heart,
  Users,
  Award,
  Stethoscope,
  ChevronLeft,
  FileText,
  ChevronRight,
  MousePointer2
} from 'lucide-react';
import CustomDropdown from './components/CustomDropdown';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const LandingPage = ({ onLogin, onNavigateToPricing }) => {
  // State for layout and animations
  const [scrolled, setScrolled] = useState(false);
  const [activeTab, setActiveTab] = useState('home');
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isVisible, setIsVisible] = useState({});
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1024);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Refs for scrolling and visibility
  const heroRef = useRef(null);
  const aboutRef = useRef(null);
  const techRef = useRef(null);
  const pricingRef = useRef(null);
  const loginRef = useRef(null);

  // Carousel Removal - Simplified Hero
  // Hero is now static with a high-impact medical image

  // Login State
  const [loginType, setLoginType] = useState('hospital'); 
  const [credentials, setCredentials] = useState({
    username: '',
    password: '',
    hospitalId: '',
    patientCode: '',
    accessCode: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hospitals, setHospitals] = useState([]);
  const [hospitalsLoading, setHospitalsLoading] = useState(false);
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);

  // Responsive & Scroll Listeners
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    const handleScroll = () => setScrolled(window.scrollY > 50);

    window.addEventListener('resize', handleResize);
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Intersection Observer for Animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setIsVisible(prev => ({ ...prev, [entry.target.id]: true }));
          }
        });
      },
      { threshold: 0.1 }
    );

    const sections = [heroRef, aboutRef, techRef, pricingRef, loginRef];
    sections.forEach(ref => {
      if (ref.current) observer.observe(ref.current);
    });

    return () => observer.disconnect();
  }, []);

  // Scroll Spy for Active Navbar State
  useEffect(() => {
    const spyObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            setActiveTab(id === 'hero' ? 'home' : id);
          }
        });
      },
      { threshold: 0.5 }
    );

    const sections = [heroRef, aboutRef, pricingRef];
    sections.forEach(ref => {
      if (ref.current) spyObserver.observe(ref.current);
    });

    return () => spyObserver.disconnect();
  }, []);

  // Fetch hospitals for patient login
  useEffect(() => {
    if (loginType === 'patient') {
      fetchHospitals();
    }
  }, [loginType]);

  const fetchHospitals = async () => {
    try {
      setHospitalsLoading(true);
      const res = await fetch(`${API_BASE}/public/hospitals`);
      if (res.ok) {
        const data = await res.json();
        setHospitals(data.hospitals || []);
      }
    } catch (err) {
      console.error('Error fetching hospitals:', err);
    } finally {
      setHospitalsLoading(false);
    }
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      let endpoint = '';
      let body = {};

      if (loginType === 'admin') {
        endpoint = '/admin/login';
        body = { username: credentials.username, password: credentials.password };
      } else if (loginType === 'hospital') {
        endpoint = '/hospital/login';
        body = { username: credentials.username, password: credentials.password };
      } else if (loginType === 'patient') {
        endpoint = '/patient/verify';
        body = {
          hospital_id: credentials.hospitalId,
          patient_code: credentials.patientCode,
          access_code: credentials.accessCode
        };
      }

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Login failed');
      }

      const data = await res.json();
      onLogin(data.user || data.patient);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    setCredentials({ ...credentials, [e.target.name]: e.target.value });
  };

  const scrollTo = (ref, name) => {
    ref.current?.scrollIntoView({ behavior: 'smooth' });
    setActiveTab(name);
  };

  return (
    <div style={{ 
      fontFamily: "'Inter', sans-serif",
      color: "#1e293b",
      overflowX: "hidden",
      scrollBehavior: "smooth"
    }}>
      {/* Refined AI Background Infrastructure - Permanent & Site-Wide */}
      <div style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "#ffffff",
        zIndex: -20,
        pointerEvents: "none"
      }} />
      <div className="ai-grid" style={{ 
        position: "fixed", 
        top: 0, left: 0, right: 0, bottom: 0,
        maskImage: "linear-gradient(to bottom, black 0%, transparent 100%)",
        WebkitMaskImage: "linear-gradient(to bottom, black 0%, transparent 100%)",
        opacity: 0.4,
        zIndex: -15
      }} />
      
      {/* Persistent Glow Blobs - Now Fixed to Viewport for Site-Wide Vibe */}
      <div style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, overflow: "hidden", zIndex: -10, pointerEvents: "none" }}>
        <div className="glow-blob" style={{ 
          width: "1000px", height: "1000px", background: "rgba(37, 99, 235, 0.4)", top: "-300px", right: "-200px", animationDelay: "0s" 
        }} />
        <div className="glow-blob" style={{ 
          width: "800px", height: "800px", background: "rgba(79, 70, 229, 0.35)", top: "20%", left: "-250px", animationDelay: "-7s" 
        }} />
        <div className="glow-blob" style={{ 
          width: "700px", height: "700px", background: "rgba(139, 92, 246, 0.3)", bottom: "10%", right: "-100px", animationDelay: "-14s" 
        }} />
      </div>
<style>{`
        .mobile-menu { 
          display: none;
          position: fixed; top: 0; right: 0; bottom: 0; left: 0; 
          background: white; z-index: 2000; 
          flex-direction: column; align-items: center; justify-content: center;
          gap: 32px; transform: translateX(100%); transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .mobile-menu.open { display: flex; transform: translateX(0); }
        .software-preview-img { width: 100%; transition: transform 0.5s; }
        
        @keyframes blobMove {
          0% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0, 0) scale(1); }
        }

        .glow-blob {
          position: absolute;
          filter: blur(120px);
          opacity: 0.4;
          border-radius: 50%;
          animation: blobMove 30s infinite alternate cubic-bezier(0.4, 0, 0.2, 1);
        }

        .ai-grid {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-image: 
            linear-gradient(rgba(37, 99, 235, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(37, 99, 235, 0.05) 1px, transparent 1px);
          background-size: 60px 60px;
        }

        @media (max-width: 1024px) {
          .mobile-hide { display: none !important; }
          .hero-title { font-size: 38px !important; letter-spacing: -1.5px !important; }
          .hero-desc { font-size: 16px !important; }
          .section-padding { padding: 60px 0 !important; }
          .responsive-grid-about { 
            display: flex !important; 
            flex-direction: column !important; 
            gap: 60px !important; 
            align-items: stretch !important;
          }
          .responsive-grid-pricing { grid-template-columns: 1fr !important; gap: 24px !important; }
          .nav-container { padding: 0 24px !important; width: 97% !important; height: 72px !important; top: 10px !important; }
          .mobile-menu { display: flex; }
          .software-preview-container { padding: 0 !important; width: 100% !important; max-width: none !important; margin-top: -20px; }
          .software-preview-img { width: 125% !important; transform: translateX(-10%); }
          .glow-blob { opacity: 0.3 !important; filter: blur(60px) !important; }
          .responsive-capabilities-grid { grid-template-columns: 1fr !important; gap: 12px !important; }
          .main-wrapper { padding: 0 16px !important; }
          .footer-wrapper { padding: 0 20px !important; }
          .footer-content-row { flex-direction: column !important; gap: 60px !important; }
          .footer-links-row { flex-direction: column !important; gap: 40px !important; }
          .footer-bottom-row { flex-direction: column !important; gap: 24px !important; align-items: center !important; text-align: center !important; }
        }
      `}</style>

      {/* Navbar - Modern Floating Pop-out */}
      <nav 
        className="nav-container"
        style={{
          position: "fixed",
          top: scrolled ? "20px" : "0",
          left: "50%",
          transform: "translateX(-50%)",
          width: scrolled ? "94%" : "100%",
          maxWidth: "1440px",
          height: scrolled ? "80px" : "100px",
          background: scrolled ? "rgba(255, 255, 255, 0.85)" : "transparent",
          backdropFilter: scrolled ? "blur(20px)" : "none",
          borderRadius: scrolled ? "40px" : "0",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "0 60px",
          zIndex: 1000,
          transition: "all 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
          boxShadow: scrolled ? "0 20px 40px rgba(0,0,0,0.06)" : "none",
          border: scrolled ? "1px solid rgba(255, 255, 255, 0.3)" : "none"
        }}
      >
        <div 
          onClick={() => scrollTo(heroRef, 'home')}
          style={{ display: "flex", alignItems: "center", gap: "12px", cursor: "pointer" }}
        >
          <div style={{ 
            background: "linear-gradient(135deg, #2563eb 0%, #1e40af 100%)",
            width: isMobile ? "36px" : "44px",
            height: isMobile ? "36px" : "44px",
            borderRadius: "12px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "white",
            boxShadow: "0 8px 16px rgba(37, 99, 235, 0.2)"
          }}>
            <Brain size={isMobile ? 20 : 26} />
          </div>
          <span style={{ fontSize: isMobile ? "20px" : "26px", fontWeight: "900", letterSpacing: "-1px", color: "#0f172a" }}>
            NeuroScan
          </span>
        </div>

        {/* Desktop Menu */}
        <div className="mobile-hide" style={{ display: "flex", alignItems: "center", gap: "40px" }}>
          {['home', 'about', 'pricing'].map((tab) => (
            <button 
              key={tab}
              onClick={() => {
                if (tab === 'pricing' && onNavigateToPricing) {
                  onNavigateToPricing();
                } else {
                  scrollTo(tab === 'home' ? heroRef : tab === 'about' ? aboutRef : pricingRef, tab);
                }
              }}
              style={{ 
                background: activeTab === tab ? "rgba(37, 99, 235, 0.1)" : "none", 
                border: "none", 
                padding: "10px 24px",
                borderRadius: "30px",
                color: activeTab === tab ? "#2563eb" : "#64748b", 
                fontWeight: "700", 
                fontSize: "17px",
                cursor: "pointer", 
                transition: "all 0.3s",
                position: "relative",
                textTransform: "capitalize"
              }}
            >
              {tab}
            </button>
          ))}
          <button 
            onClick={() => scrollTo(loginRef, 'login')}
            style={{
              padding: "14px 36px",
              background: "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "50px",
              fontWeight: "700",
              fontSize: "16px",
              cursor: "pointer",
              boxShadow: "0 10px 20px rgba(37, 99, 235, 0.2)"
            }}
          >
            Get Started
          </button>
        </div>

        {/* Mobile Menu Toggle */}
        {isMobile && (
          <button 
            onClick={() => setMobileMenuOpen(true)}
            style={{ background: "none", border: "none", color: "#0f172a", cursor: "pointer" }}
          >
            <Users size={24} />
          </button>
        )}
      </nav>

      {/* Mobile Sidebar/Drawer Overlay */}
      {isMobile && (
        <div className={`mobile-menu ${mobileMenuOpen ? 'open' : ''}`}>
          <button 
            onClick={() => setMobileMenuOpen(false)}
            style={{ position: "absolute", top: "30px", right: "30px", background: "none", border: "none", color: "#0f172a" }}
          >
            <X size={32} />
          </button>
          {['home', 'about', 'pricing', 'login'].map((tab) => (
            <button 
              key={tab}
              onClick={() => {
                setMobileMenuOpen(false);
                scrollTo(tab === 'home' ? heroRef : tab === 'about' ? aboutRef : tab === 'pricing' ? pricingRef : loginRef, tab);
              }}
              style={{ background: "none", border: "none", fontSize: "32px", fontWeight: "800", color: "#0f172a", textTransform: "capitalize" }}
            >
              {tab}
            </button>
          ))}
        </div>
      )}


      {/* Main Content Centered Wrapper */}
      <main className="main-wrapper" style={{ maxWidth: "1440px", margin: "0 auto", padding: "0 40px" }}>
        
        {/* Redesigned Centered Hero Section */}
        <section id="hero" ref={heroRef} style={{ 
          minHeight: "85vh", 
          paddingTop: isMobile ? "100px" : "140px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          position: "relative",
          zIndex: 1
        }}>
          {/* Hero Content */}
          <div style={{ 
            maxWidth: "900px",
            opacity: isVisible.hero ? 1 : 0, 
            transform: isVisible.hero ? "translateY(0)" : "translateY(30px)",
            transition: "all 1.2s cubic-bezier(0.2, 0.8, 0.2, 1)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center"
          }}>
            <div style={{ 
              color: "#2563eb",
              fontSize: "13px",
              fontWeight: "800",
              textTransform: "uppercase",
              letterSpacing: "3px",
              marginBottom: "8px",
              opacity: 0.8
            }}>
              ★ The Gold Standard in Neuro-Diagnosis
            </div>
            <h1 className="hero-title" style={{ 
              fontSize: "84px", 
              fontWeight: "900", 
              lineHeight: "0.95", 
              color: "#0f172a", 
              marginBottom: "16px", 
              letterSpacing: "-4px" 
            }}>
              AI for Advanced <br />
              <span style={{ color: "#2563eb" }}>Brain Tumor</span> Analysis
            </h1>
            <p className="hero-desc" style={{ 
              fontSize: "22px", 
              color: "#64748b", 
              maxWidth: "700px", 
              marginBottom: "48px",
              lineHeight: "1.6"
            }}>
              Empowering clinics with state-of-the-art deep learning technology. Quantifying tumor progression with unmatched accuracy and clinical confidence.
            </p>
            <div style={{ display: "flex", gap: "20px", marginBottom: "80px" }}>
              <button 
                onClick={() => scrollTo(loginRef, 'login')}
                style={{
                  padding: "18px 48px",
                  background: "#2563eb",
                  color: "white",
                  border: "none",
                  borderRadius: "50px",
                  fontWeight: "700",
                  fontSize: "17px",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  boxShadow: "0 20px 40px rgba(37, 99, 235, 0.25)",
                  transition: "all 0.3s"
                }}
                onMouseEnter={e => e.target.style.transform = "translateY(-4px)"}
                onMouseLeave={e => e.target.style.transform = "translateY(0)"}
              >
                Try Demo <ArrowRight size={20} />
              </button>
              <button 
                onClick={() => scrollTo(aboutRef, 'about')}
                style={{
                  padding: "18px 48px",
                  background: "white",
                  color: "#0f172a",
                  border: "2px solid #f1f5f9",
                  borderRadius: "50px",
                  fontWeight: "700",
                  fontSize: "17px",
                  cursor: "pointer",
                  transition: "all 0.3s"
                }}
                onMouseEnter={e => e.target.style.background = "#f8fafc"}
                onMouseLeave={e => e.target.style.background = "white"}
              >
                Learn More
              </button>
            </div>
          </div>

          {/* Software Preview Section */}
          <div className="software-preview-container" style={{ 
            width: "100%",
            maxWidth: "1200px",
            opacity: isVisible.hero ? 1 : 0, 
            transform: isVisible.hero ? "translateY(0)" : "translateY(60px)",
            transition: "all 1.4s cubic-bezier(0.2, 0.8, 0.2, 1) 0.3s",
            padding: "0 20px"
          }}>
            <div style={{ 
              position: "relative",
              borderRadius: isMobile ? "12px" : "24px",
              overflow: "hidden",
              boxShadow: "0 50px 100px -20px rgba(0,0,0,0.25), 0 30px 60px -30px rgba(0,0,0,0.3)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              background: "#1e293b"
            }}>
              <img 
                src="/software_preview.png" 
                alt="NeuroScan Software Preview"
                className="software-preview-img"
                style={{
                  width: "100%",
                  height: "auto",
                  display: "block",
                }}
              />
            </div>
          </div>
        </section>

        {/* Redesigned Premium About Us Section - Simplified & Modern */}
        <section id="about" className="section-padding" ref={aboutRef} style={{ padding: "100px 0" }}>
          <div className="responsive-grid-about" style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1.2fr", 
            gap: "80px", 
            alignItems: "start",
            opacity: isVisible.about ? 1 : 0,
            transform: isVisible.about ? "translateY(0)" : "translateY(40px)",
            transition: "all 1s"
          }}>
            <div style={{ 
              position: isMobile ? "relative" : "sticky", 
              top: isMobile ? "0" : "140px", 
              width: "100%",
              marginBottom: isMobile ? "10px" : "0"
            }}>
              <h2 style={{ fontSize: "16px", fontWeight: "800", color: "#2563eb", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "20px" }}>About NeuroScan</h2>
              <h3 style={{ fontSize: "52px", fontWeight: "900", color: "#0f172a", marginBottom: "32px", lineHeight: "1.1", letterSpacing: "-2px" }}>Establishing Global Clinical Confidence</h3>
              <p style={{ fontSize: "19px", color: "#64748b", lineHeight: "1.7", marginBottom: "40px" }}>
                We bridge the gap between complex artificial intelligence research and daily clinical application. Our mission is to provide life-saving insights through precision neuro-imaging.
              </p>
              <div style={{ display: "flex", gap: "16px", marginBottom: "40px" }}>
                <div style={{ 
                  background: "#f1f5f9", 
                  padding: "20px", 
                  borderRadius: "20px",
                  flex: 1
                }}>
                  <div style={{ color: "#2563eb", fontWeight: "900", fontSize: "32px", marginBottom: "4px" }}>99.2%</div>
                  <div style={{ fontSize: "13px", fontWeight: "700", color: "#64748b", textTransform: "uppercase" }}>Analysis Accuracy</div>
                </div>
                <div style={{ 
                  background: "#f1f5f9", 
                  padding: "20px", 
                  borderRadius: "20px",
                  flex: 1
                }}>
                  <div style={{ color: "#2563eb", fontWeight: "900", fontSize: "32px", marginBottom: "4px" }}>2.4s</div>
                  <div style={{ fontSize: "13px", fontWeight: "700", color: "#64748b", textTransform: "uppercase" }}>Scan Processing</div>
                </div>
              </div>

              {/* Functional Feature Tile to fill space and highlight capabilities */}
              <div style={{ 
                background: "linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)",
                padding: "32px",
                borderRadius: "32px",
                border: "1px solid #f1f5f9",
                boxShadow: "0 20px 40px rgba(0,0,0,0.02)",
                display: "flex",
                flexDirection: "column",
                gap: "20px"
              }}>
                <h4 style={{ fontSize: "14px", fontWeight: "800", color: "#2563eb", textTransform: "uppercase", letterSpacing: "1px" }}>Core Capabilities</h4>
                <div className="responsive-capabilities-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                  {[
                    { label: "AI Segmentation", icon: <Brain size={16} /> },
                    { label: "MRI Multi-modal", icon: <Activity size={16} /> },
                    { label: "Real-time Sync", icon: <Zap size={16} /> },
                    { label: "Clinical PDF", icon: <FileText size={16} /> },
                    { label: "Encrypted Data", icon: <Shield size={16} /> },
                    { label: "Specialist Chat", icon: <Users size={16} /> }
                  ].map((feat, i) => (
                    <div key={i} style={{ 
                      display: "flex", 
                      alignItems: "center", 
                      gap: "10px", 
                      background: "white", 
                      padding: "12px 16px", 
                      borderRadius: "16px",
                      border: "1px solid #f1f5f9",
                      fontSize: "14px",
                      fontWeight: "600",
                      color: "#334155"
                    }}>
                      <div style={{ color: "#2563eb" }}>{feat.icon}</div>
                      {feat.label}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "24px" }}>
              {[
                { 
                  title: "Clinical-Grade Precision", 
                  desc: "Utilizing custom CNN architectures trained on massive clinical datasets for reliable tumor boundary detection.",
                  icon: <Activity size={24} /> 
                },
                { 
                  title: "Seamless Integration", 
                  desc: "Directly connects with existing clinical workflows and radiologist dashboards for instant reporting.",
                  icon: <Zap size={24} /> 
                },
                { 
                  title: "Privacy First Architecture", 
                  desc: "Enterprise-grade data encryption and HIPAA-compliant processing ensure patient confidentiality at all levels.",
                  icon: <Shield size={24} /> 
                },
                { 
                  title: "Real-time Collaboration", 
                  desc: "Integrated chat and conferencing tools facilitate instant consultation between neuro-specialists.",
                  icon: <Users size={24} /> 
                }
              ].map((item, idx) => (
                <div key={idx} style={{ 
                  background: "white", 
                  padding: "32px", 
                  borderRadius: "24px", 
                  border: "1px solid #f1f5f9",
                  boxShadow: "0 10px 30px rgba(0,0,0,0.02)",
                  display: "flex",
                  gap: "24px",
                  transition: "all 0.3s"
                }} onMouseEnter={e => {
                  e.currentTarget.style.transform = "translateX(10px)";
                  e.currentTarget.style.borderColor = "#2563eb";
                }} onMouseLeave={e => {
                  e.currentTarget.style.transform = "translateX(0)";
                  e.currentTarget.style.borderColor = "#f1f5f9";
                }}>
                  <div style={{ 
                    width: "56px", 
                    height: "56px", 
                    borderRadius: "16px", 
                    background: "rgba(37,99,235,0.08)", 
                    color: "#2563eb",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0
                  }}>
                    {item.icon}
                  </div>
                  <div>
                    <h4 style={{ fontSize: "20px", fontWeight: "800", marginBottom: "8px", color: "#0f172a" }}>{item.title}</h4>
                    <p style={{ color: "#64748b", lineHeight: "1.6", fontSize: "15px", margin: 0 }}>{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Pricing Table */}
        <section id="pricing" className="section-padding" ref={pricingRef} style={{ padding: "100px 0" }}>
          <div style={{ textAlign: "center", marginBottom: "80px" }}>
            <h2 style={{ fontSize: "16px", fontWeight: "800", color: "#2563eb", textTransform: "uppercase", letterSpacing: "2px", marginBottom: "8px" }}>Simple Pricing</h2>
            <h3 style={{ fontSize: isMobile ? "36px" : "52px", fontWeight: "900", color: "#0f172a", marginBottom: "24px", letterSpacing: "-2px" }}>Scalable Clinical Solutions</h3>
          </div>

          <div className="responsive-grid-pricing" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "40px" }}>
            {[
              { name: "Free", price: "0", scans: "10", features: ["10 MRI Scans", "AI Analysis", "Basic Reports"], popular: false },
              { name: "Basic", price: "500", scans: "100", features: ["100 MRI Scans", "AI Analysis", "PDF Reports", "Chat System"], popular: true },
              { name: "Premium", price: "1000", scans: "1000", features: ["1000 MRI Scans", "Unlimited Features", "Priority Support", "API Access"], popular: false }
            ].map((plan, i) => (
              <div key={i} style={{ 
                background: "white", 
                padding: "60px 40px", 
                borderRadius: "40px", 
                border: plan.popular ? "2px solid #2563eb" : "1px solid #f1f5f9",
                boxShadow: plan.popular ? "0 30px 60px rgba(37, 99, 235, 0.15)" : "0 20px 40px rgba(0,0,0,0.03)",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                transition: "all 0.4s",
                opacity: isVisible.pricing ? 1 : 0,
                transform: isVisible.pricing ? "translateY(0)" : "translateY(60px)",
                transitionDelay: `${i * 0.2}s`
              }}>
                {plan.popular && (
                  <div style={{ position: "absolute", top: "24px", right: "40px", background: "#2563eb", color: "white", padding: "6px 16px", borderRadius: "20px", fontSize: "12px", fontWeight: "800" }}>MOST POPULAR</div>
                )}
                <h4 style={{ fontSize: "22px", fontWeight: "800", marginBottom: "12px" }}>{plan.name}</h4>
                <div style={{ marginBottom: "40px" }}>
                  <span style={{ fontSize: "56px", fontWeight: "900", letterSpacing: "-3px" }}>Nrs {plan.price}</span>
                  <span style={{ color: "#64748b", fontWeight: "600" }}>/month</span>
                </div>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "20px", marginBottom: "48px" }}>
                  {plan.features.map((f, fi) => (
                    <div key={fi} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <div style={{ background: "rgba(37, 99, 235, 0.1)", width: "24px", height: "24px", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", color: "#2563eb" }}>
                        <Check size={14} strokeWidth={3} />
                      </div>
                      <span style={{ fontWeight: "600", color: "#475569" }}>{f}</span>
                    </div>
                  ))}
                </div>
                <button 
                  onClick={() => {
                    setShowLoginPrompt(true);
                  }}
                  style={{
                    padding: "18px",
                    borderRadius: "50px",
                    border: plan.popular ? "none" : "2px solid #f1f5f9",
                    background: plan.popular ? "#2563eb" : "white",
                    color: plan.popular ? "white" : "#0f172a",
                    fontWeight: "800",
                    cursor: "pointer",
                    fontSize: "16px",
                    transition: "all 0.3s"
                  }}
                  onMouseEnter={e => {
                    e.target.style.background = plan.popular ? "#1d4ed8" : "#f1f5f9";
                    e.target.style.transform = "translateY(-4px)";
                  }}
                  onMouseLeave={e => {
                    e.target.style.background = plan.popular ? "#2563eb" : "white";
                    e.target.style.transform = "translateY(0)";
                  }}
                >
                  Get Started
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Login Portal Section - Modernized */}
        <section id="login" className="section-padding" ref={loginRef} style={{ padding: "100px 0 160px" }}>
          <div className="responsive-grid-about" style={{ 
            background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)", 
            borderRadius: isMobile ? "30px" : "60px",
            padding: isMobile ? "40px 20px" : "80px",
            display: "flex",
            flexDirection: isMobile ? "column" : "row",
            gap: isMobile ? "40px" : "80px",
            alignItems: "center",
            boxShadow: "0 60px 120px -20px rgba(0,0,0,0.4)",
            opacity: isVisible.login ? 1 : 0,
            transform: isVisible.login ? "translateY(0)" : "translateY(80px)",
            transition: "all 1s ease-out"
          }}>
            {/* Login Text Side */}
            <div style={{ flex: 1 }}>
              <h2 style={{ fontSize: "56px", fontWeight: "900", color: "white", marginBottom: "32px", letterSpacing: "-3px" }}>Access Your Secure Portal</h2>
              <p style={{ color: "#94a3b8", fontSize: "20px", lineHeight: "1.7", marginBottom: "48px" }}>
                Select your clinical role to access the NeuroScan management dashboard. Experience enterprise-grade security and AI performance.
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
                {[
                  { icon: <Shield size={20} />, text: "HIPAA Compliant" },
                  { icon: <Zap size={20} />, text: "PCI-DSS Level 1" },
                  { icon: <Stethoscope size={20} />, text: "GCP Certified" }
                ].map((item, i) => (
                  <div key={i} style={{ 
                    background: "rgba(255,255,255,0.05)", 
                    padding: "12px 24px", 
                    borderRadius: "50px", 
                    color: "white", 
                    display: "flex", 
                    alignItems: "center", 
                    gap: "10px",
                    border: "1px solid rgba(255,255,255,0.1)",
                    fontSize: "14px",
                    fontWeight: "600"
                  }}>
                    {item.icon} {item.text}
                  </div>
                ))}
              </div>
            </div>

            {/* Login Form Side - Premium Redesign */}
            <div style={{ flex: 1 }}>
              <div style={{ 
                background: "rgba(255, 255, 255, 0.03)", 
                backdropFilter: "blur(40px)",
                padding: "60px",
                borderRadius: "40px",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                boxShadow: "inset 0 0 40px rgba(255,255,255,0.05)"
              }}>
                {/* Role Switcher */}
                <div style={{ 
                  display: "flex", 
                  background: "rgba(255,255,255,0.05)", 
                  padding: "8px", 
                  borderRadius: "50px", 
                  marginBottom: "40px",
                  border: "1px solid rgba(255,255,255,0.05)"
                }}>
                  {['hospital', 'patient', 'admin'].map(type => (
                    <button
                      key={type}
                      onClick={() => setLoginType(type)}
                      style={{
                        flex: 1,
                        padding: "12px",
                        borderRadius: "50px",
                        border: "none",
                        background: loginType === type ? "white" : "transparent",
                        color: loginType === type ? "#0f172a" : "#94a3b8",
                        fontWeight: "800",
                        fontSize: "14px",
                        textTransform: "capitalize",
                        cursor: "pointer",
                        transition: "all 0.3s"
                      }}
                    >
                      {type}
                    </button>
                  ))}
                </div>

                <form onSubmit={handleLoginSubmit}>
                  {loginType !== 'patient' ? (
                    <>
                      <div style={{ marginBottom: "24px" }}>
                        <label style={{ color: "#94a3b8", fontSize: "14px", fontWeight: "700", marginBottom: "10px", display: "block" }}>Username</label>
                        <input 
                          type="text" 
                          name="username" 
                          value={credentials.username} 
                          onChange={handleInputChange}
                          autoComplete="off"
                          style={{
                            width: "100%", padding: "18px 24px", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.1)",
                            background: "rgba(255,255,255,0.05)", color: "white", outline: "none", fontSize: "16px"
                          }} 
                        />
                      </div>
                      <div style={{ marginBottom: "40px" }}>
                        <label style={{ color: "#94a3b8", fontSize: "14px", fontWeight: "700", marginBottom: "10px", display: "block" }}>Password</label>
                        <input 
                          type="password" 
                          name="password" 
                          value={credentials.password} 
                          onChange={handleInputChange}
                          style={{
                            width: "100%", padding: "18px 24px", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.1)",
                            background: "rgba(255,255,255,0.05)", color: "white", outline: "none", fontSize: "16px"
                          }} 
                        />
                      </div>
                    </>
                  ) : (
                    <>
                      <div style={{ marginBottom: "20px" }}>
                        <CustomDropdown
                          label="Hospital"
                          placeholder="Select Hospital"
                          value={credentials.hospitalId}
                          onChange={(e) => setCredentials({ ...credentials, hospitalId: e.target.value })}
                          options={hospitals.map(h => ({ value: h.id, label: h.name }))}
                          darkMode={true}
                          variant="glass"
                        />
                      </div>
                      <div style={{ marginBottom: "20px" }}>
                        <input 
                          type="text" 
                          name="patientCode" 
                          placeholder="Patient Code"
                          value={credentials.patientCode} 
                          onChange={handleInputChange}
                          style={{
                            width: "100%", padding: "18px 24px", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.1)",
                            background: "rgba(255,255,255,0.05)", color: "white", outline: "none"
                          }}
                        />
                      </div>
                      <div style={{ marginBottom: "40px" }}>
                        <input 
                          type="text" 
                          name="accessCode" 
                          placeholder="Access Code"
                          value={credentials.accessCode} 
                          onChange={handleInputChange}
                          style={{
                            width: "100%", padding: "18px 24px", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.1)",
                            background: "rgba(255,255,255,0.05)", color: "white", outline: "none"
                          }}
                        />
                      </div>
                    </>
                  )}

                  {error && <div style={{ marginBottom: "24px", color: "#f87171", background: "rgba(248,113,113,0.1)", padding: "16px", borderRadius: "12px", fontSize: "14px", fontWeight: "600" }}>{error}</div>}

                  <button
                    type="submit"
                    disabled={loading}
                    style={{
                      width: "100%", padding: "20px", borderRadius: "50px", border: "none",
                      background: "linear-gradient(135deg, #2563eb 0%, #1e40af 100%)",
                      color: "white", fontWeight: "800", fontSize: "17px", cursor: "pointer",
                      display: "flex", alignItems: "center", justifyContent: "center", gap: "12px",
                      boxShadow: "0 20px 40px rgba(37, 99, 235, 0.4)",
                      transition: "all 0.3s"
                    }}
                    onMouseEnter={e => e.target.style.transform = "scale(1.02)"}
                    onMouseLeave={e => e.target.style.transform = "scale(1)"}
                  >
                    {loading ? <Loader size={20} className="spin" /> : 'Log In to System'}
                  </button>
                </form>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer Area Code */}
      <footer style={{ background: "#0f172a", color: "white", padding: "100px 0 60px" }}>
        <div className="footer-wrapper" style={{ maxWidth: "1440px", margin: "0 auto", padding: "0 60px" }}>
          <div className="footer-content-row" style={{ display: "flex", justifyContent: "space-between", marginBottom: "80px", alignItems: "flex-start" }}>
            <div style={{ maxWidth: "400px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "32px" }}>
                <Brain size={40} color="#2563eb" />
                <span style={{ fontSize: "32px", fontWeight: "900", letterSpacing: "-1px" }}>NeuroScan</span>
              </div>
              <p style={{ color: "#94a3b8", fontSize: "16px", lineHeight: "1.8", marginBottom: "40px" }}>
                Redefining neurological healthcare through artificial intelligence and deep learning. Empowering clinicians worldwide.
              </p>
              <div style={{ display: "flex", gap: "20px" }}>
                {[Globe, Users, Award].map((Icon, i) => (
                  <div key={i} style={{ width: "48px", height: "48px", borderRadius: "50%", background: "rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "center", color: "#94a3b8", cursor: "pointer", transition: "all 0.3s" }}>
                    <Icon size={20} />
                  </div>
                ))}
              </div>
            </div>
            
            <div className="footer-links-row" style={{ display: "flex", gap: "120px" }}>
              <div>
                <h5 style={{ fontSize: "18px", fontWeight: "800", marginBottom: "32px" }}>Platform</h5>
                <div style={{ display: "flex", flexDirection: "column", gap: "16px", color: "#94a3b8", fontSize: "15px" }}>
                  <span>Diagnostics</span>
                  <span>System AI</span>
                  <span>Security & Compliance</span>
                  <span>API Research</span>
                </div>
              </div>
              <div>
                <h5 style={{ fontSize: "18px", fontWeight: "800", marginBottom: "32px" }}>Company</h5>
                <div style={{ display: "flex", flexDirection: "column", gap: "16px", color: "#94a3b8", fontSize: "15px" }}>
                  <span>Our Mission</span>
                  <span>Clinical Trials</span>
                  <span>Privacy Policy</span>
                  <span>Contact Care</span>
                </div>
              </div>
            </div>
          </div>
          <div className="footer-bottom-row" style={{ borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "40px", display: "flex", justifyContent: "space-between", color: "#64748b", fontSize: "14px" }}>
            <span>© 2026 NeuroScan AI Solutions. All rights reserved.</span>
            <div style={{ display: "flex", gap: "32px" }}>
              <span>Terms of Service</span>
              <span>Cookie Policy</span>
            </div>
          </div>
        </div>
      </footer>

      <style>
        {`
          .spin { animation: spin 1s linear infinite; }
          @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
          
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          
          @keyframes modalScaleIn {
            0% { opacity: 0; transform: translate(-50%, -40%) scale(0.95); }
            100% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
          }
          
          /* Custom scrollbar for better medical feel */
          ::-webkit-scrollbar { width: 10px; }
          ::-webkit-scrollbar-track { background: #f8fafc; }
          ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; border: 3px solid #f8fafc; }
          ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        `}
      </style>

      {/* Premium Login Prompt Modal */}
      {showLoginPrompt && (
        <>
          <div 
            onClick={() => setShowLoginPrompt(false)}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(15, 23, 42, 0.4)",
              backdropFilter: "blur(8px)",
              zIndex: 9999,
              animation: "fadeIn 0.4s ease-out"
            }}
          />
          <div style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            width: "90%",
            maxWidth: "480px",
            background: "rgba(255, 255, 255, 0.8)",
            backdropFilter: "blur(24px) saturate(180%)",
            padding: "48px 40px",
            borderRadius: "32px",
            boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.15), inset 0 0 0 1px rgba(255, 255, 255, 0.5)",
            zIndex: 10000,
            textAlign: "center",
            animation: "modalScaleIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)"
          }}>
            {/* Close Button */}
            <button
              onClick={() => setShowLoginPrompt(false)}
              style={{
                position: "absolute",
                top: "24px",
                right: "24px",
                background: "rgba(0,0,0,0.05)",
                border: "none",
                width: "36px",
                height: "36px",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#64748b",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
              onMouseEnter={e => e.target.style.background = "rgba(0,0,0,0.1)"}
              onMouseLeave={e => e.target.style.background = "rgba(0,0,0,0.05)"}
            >
              <X size={20} />
            </button>

            <div style={{
              width: "80px",
              height: "80px",
              background: "linear-gradient(135deg, #2563eb 0%, #1e40af 100%)",
              borderRadius: "24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 32px",
              boxShadow: "0 10px 20px rgba(37, 99, 235, 0.2)"
            }}>
              <User size={40} color="white" />
            </div>

            <h3 style={{ fontSize: "28px", fontWeight: "900", color: "#0f172a", marginBottom: "16px", letterSpacing: "-1px" }}>
              Ready to Upgrade?
            </h3>
            
            <p style={{ fontSize: "17px", color: "#64748b", marginBottom: "36px", lineHeight: "1.6" }}>
              Join the future of neuro-diagnosis. Log in as a hospital to subscribe to our clinical-grade AI plans.
            </p>

            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <button
                onClick={() => {
                  setShowLoginPrompt(false);
                  scrollTo(loginRef, 'login');
                }}
                style={{
                  padding: "18px",
                  background: "#2563eb",
                  color: "white",
                  border: "none",
                  borderRadius: "50px",
                  fontWeight: "800",
                  fontSize: "17px",
                  cursor: "pointer",
                  boxShadow: "0 10px 20px rgba(37, 99, 235, 0.25)",
                  transition: "all 0.3s",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "10px"
                }}
                onMouseEnter={e => e.target.style.transform = "translateY(-4px)"}
                onMouseLeave={e => e.target.style.transform = "translateY(0)"}
              >
                Go to Portal <ArrowRight size={20} />
              </button>
              
              <button
                onClick={() => setShowLoginPrompt(false)}
                style={{
                  padding: "16px",
                  background: "transparent",
                  color: "#64748b",
                  border: "none",
                  fontWeight: "700",
                  fontSize: "15px",
                  cursor: "pointer"
                }}
              >
                Maybe later
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default LandingPage;
