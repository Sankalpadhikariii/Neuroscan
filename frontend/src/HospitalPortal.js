import React, { useState, useEffect, useRef } from "react";
import { loadStripe } from '@stripe/stripe-js';
import {
  Upload,
  Brain,
  LogOut,
  Users,
  FileText,
  Plus,
  Loader,
  FileDown,
  AlertCircle,
  BarChart3,
  Settings,
  CheckCircle,
  X,
  MessageCircle,
  Download,
  CreditCard,
  Zap,
  Send,
  Paperclip,
  Bell,
  TrendingUp,
  Eye,
  AlertTriangle,
  Image as ImageIcon,
  Video,
  Phone,
  LayoutDashboard,
  Search,
  ChevronDown,
  ChevronUp,
  Calendar,
  User,
  Mail,
  Shield,
  Check,
  ArrowRight,
} from "lucide-react";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
  AreaChart,
  Area
} from 'recharts';
import { io } from "socket.io-client";
import EnhancedChat from "./EnhancedChat";
import NotificationCentre from "./NotificationCentre";
import GradCamvisualization from "./GradCamvisualization";
import TumourProgressionTracker from "./TumourProgressionTracker";
import AppointmentCalendar from "./AppointmentCalendar";
import CustomDropdown from "./components/CustomDropdown";

let stripePromise = null;

const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";
const socket = io(API_BASE, { withCredentials: true });

export default function HospitalPortalEnhanced({ user, onLogout, onNavigateToPricing }) {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("hospitalTheme") === "dark",
  );

  const [view, setView] = useState("dashboard");
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [dashboardStats, setDashboardStats] = useState(null);
  const [chartFilter, setChartFilter] = useState("daily");

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [usage, setUsage] = useState(null);

  // Inline patient form states for New Scan tab
  const [newPatientName, setNewPatientName] = useState("");
  const [newPatientEmail, setNewPatientEmail] = useState("");
  const [newPatientPhone, setNewPatientPhone] = useState("");
  const [newPatientDob, setNewPatientDob] = useState("");
  const [newPatientGender, setNewPatientGender] = useState("");
  const [newPatientAddress, setNewPatientAddress] = useState("");
  const [newPatientEmergencyContact, setNewPatientEmergencyContact] = useState("");
  const [newPatientEmergencyPhone, setNewPatientEmergencyPhone] = useState("");

  // Scan mode toggle: "existing" or "new"
  const [scanMode, setScanMode] = useState("existing");
  const [existingPatientSearch, setExistingPatientSearch] = useState("");
  const [showExistingDropdown, setShowExistingDropdown] = useState(false);

  // Patients tab states
  const [expandedPatientId, setExpandedPatientId] = useState(null);
  const [patientSearch, setPatientSearch] = useState("");
  const [expandedPatientScans, setExpandedPatientScans] = useState([]);

  // Subscription management states
  const [subscriptionPlans, setSubscriptionPlans] = useState([]);
  const [loadingPlans, setLoadingPlans] = useState(false);
  const [billingCycle, setBillingCycle] = useState("monthly");
  const [processingPlan, setProcessingPlan] = useState(null);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [upgradeMessage, setUpgradeMessage] = useState("");

  // New states for enhanced features
  const [showChat, setShowChat] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);
  const [gradcamData, setGradcamData] = useState(null);
  const [validationWarning, setValidationWarning] = useState(null);
  const [patientScans, setPatientScans] = useState([]);
  const [appointments, setAppointments] = useState([]);

  // Video call states (removed)
  // const [showVideoCall, setShowVideoCall] = useState(false);
  // const [callType, setCallType] = useState("video");

  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    loadPatients();
    loadUsageStatus();
    loadNotifications();
    loadDashboardStats();
    loadHospitalAppointments();
    setupSocketListeners();

    return () => {
      socket.off("notification");
      socket.off("new_message");
      socket.off("patient_update");
    };
  }, []);

  useEffect(() => {
    if (selectedPatient) {
      loadPatientScans(selectedPatient.id);
    }
  }, [selectedPatient]);

  // Auto-scroll to results when prediction is set
  useEffect(() => {
    if (prediction && resultsRef.current) {
      setTimeout(() => {
        resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }, [prediction]);

  function setupSocketListeners() {
    socket.on("notification", (notification) => {
      setNotifications((prev) => [notification, ...prev]);
      setUnreadCount((prev) => prev + 1);
      showToast(notification.message, "info");
    });

    socket.on("new_message", (message) => {
      if (message.sender_id !== user.id) {
        setUnreadCount((prev) => prev + 1);
        showToast("New message received", "info");
      }
    });

    socket.on("patient_update", (data) => {
      loadPatients();
      showToast("Patient information updated", "success");
    });
  }

  async function loadPatients() {
    try {
      const res = await fetch(`${API_BASE}/hospital/patients`, {
        credentials: "include",
      });
      const data = await res.json();
      setPatients(data.patients || []);
      
      // Update selected patient info if one is active
      if (selectedPatient) {
        const updated = data.patients.find(p => p.id === selectedPatient.id);
        if (updated) setSelectedPatient(updated);
      }
    } catch (err) {
      console.error("Error loading patients:", err);
    }
  }

  async function loadUsageStatus() {
    try {
      const res = await fetch(`${API_BASE}/hospital/usage-status`, {
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        setUsage(data);
      }
    } catch (err) {
      console.error("Error loading usage:", err);
    }
  }

  async function loadNotifications() {
    try {
      const res = await fetch(`${API_BASE}/notifications`, {
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.notifications.filter((n) => !n.read).length);
      }
    } catch (err) {
      console.error("Error loading notifications:", err);
    }
  }

  async function loadDashboardStats() {
    try {
      const res = await fetch(`${API_BASE}/hospital/dashboard-stats`, {
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        setDashboardStats(data);
      }
    } catch (err) {
      console.error("Error loading dashboard stats:", err);
    }
  }

  async function loadHospitalAppointments() {
    try {
      const res = await fetch(`${API_BASE}/hospital/appointments`, {
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        setAppointments(data.appointments || []);
      }
    } catch (err) {
      console.error("Error loading hospital appointments:", err);
    }
  }

  async function loadPatientScans(patientId) {
    try {
      const res = await fetch(
        `${API_BASE}/hospital/patient-scans/${patientId}`,
        {
          credentials: "include",
        },
      );
      if (res.ok) {
        const data = await res.json();
        setPatientScans(data.scans || []);
      }
    } catch (err) {
      console.error("Error loading patient scans:", err);
    }
  }

  function handlePatientAdded(patient) {
    if (!patient) return;

    setPatients((prev) => [
      patient,
      ...prev.filter((p) => p.id !== patient.id),
    ]);
    setSelectedPatient(patient);
    setView("scan");
    loadPatientScans(patient.id);
    showToast("Patient added successfully", "success");
  }

  function validateMRIImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const reader = new FileReader();

      reader.onload = (e) => {
        img.onload = () => {
          // Basic MRI validation checks
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData.data;

          // Check if image is grayscale (typical for MRI)
          let isGrayscale = true;
          let hasContrast = false;
          let minIntensity = 255;
          let maxIntensity = 0;

          for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // Check if RGB values are similar (grayscale characteristic)
            if (
              Math.abs(r - g) > 10 ||
              Math.abs(g - b) > 10 ||
              Math.abs(r - b) > 10
            ) {
              isGrayscale = false;
            }

            minIntensity = Math.min(minIntensity, r);
            maxIntensity = Math.max(maxIntensity, r);
          }

          hasContrast = maxIntensity - minIntensity > 50;

          const warnings = [];
          if (!isGrayscale) {
            warnings.push(
              "Image appears to be in color. MRI scans are typically grayscale.",
            );
          }
          if (!hasContrast) {
            warnings.push(
              "Image has low contrast. This may not be a valid medical scan.",
            );
          }
          if (img.width < 128 || img.height < 128) {
            warnings.push("Image resolution is too low for accurate analysis.");
          }

          resolve({
            isValid: warnings.length === 0,
            warnings: warnings,
            confidence:
              warnings.length === 0
                ? "high"
                : warnings.length === 1
                  ? "medium"
                  : "low",
          });
        };

        img.onerror = () => {
          reject(new Error("Failed to load image"));
        };

        img.src = e.target.result;
      };

      reader.onerror = () => {
        reject(new Error("Failed to read file"));
      };

      reader.readAsDataURL(file);
    });
  }

  async function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    setPrediction(null);
    setError(null);
    setValidationWarning(null);
    setGradcamData(null);

    // Validate file type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type)) {
      setError("Invalid file type. Please upload a JPEG or PNG image.");
      setValidationWarning({
        type: "error",
        message: "Only JPEG and PNG images are accepted for MRI analysis.",
      });
      return;
    }

    setSelectedFile(file);
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);

    // Validate if it's an MRI image
    try {
      const validation = await validateMRIImage(file);

      if (!validation.isValid) {
        setValidationWarning({
          type: "warning",
          message: "Image validation warnings detected:",
          warnings: validation.warnings,
          confidence: validation.confidence,
        });

        // Send notification
        const notif = {
          type: "warning",
          message: `Uploaded image may not be a valid MRI scan: ${validation.warnings.join(", ")}`,
          timestamp: new Date().toISOString(),
        };
        setNotifications((prev) => [notif, ...prev]);
        setUnreadCount((prev) => prev + 1);
      } else {
        showToast("Valid MRI image detected", "success");
      }
    } catch (err) {
      console.error("Validation error:", err);
      setValidationWarning({
        type: "warning",
        message: "Could not validate image format. Proceed with caution.",
        warnings: ["Image validation failed"],
      });
    }
  }

  async function performAnalysis() {
    if (!selectedFile) {
      setError("Please select an image file");
      return;
    }

    if (!selectedPatient?.id) {
      setError("Please select a patient before analyzing");
      return;
    }

    // Check scan limit
    if (usage && usage.scans_used >= usage.scan_limit) {
      setError(`Scan limit reached (${usage.scans_used}/${usage.scan_limit}). Please upgrade your plan to continue.`);
      return;
    }

    if (validationWarning && validationWarning.type === "error") {
      setError("Cannot analyze: Invalid image format");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("patient_id", selectedPatient.id.toString());
    formData.append("generate_gradcam", "true");

    try {
      const res = await fetch(`${API_BASE}/hospital/predict`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        // Handle scan limit reached (403)
        if (res.status === 403 && errorData.upgrade_required) {
          setUpgradeMessage(errorData.message || `You've reached your monthly scan limit. Upgrade your plan to continue scanning.`);
          setShowUpgradeModal(true);
          setLoading(false);
          return;
        }
        throw new Error(errorData.message || errorData.error || `Server error`);
      }

      const data = await res.json();

      if (data.error) {
        setError(data.message || data.error);
        setPrediction(null);
      } else {
        setPrediction(data);

        // Load Grad-CAM if available
        if (data.gradcam_available) {
          loadGradCAM(data.scan_id);
        }

        // Refresh patient scans to show progression
        await loadPatientScans(selectedPatient.id);
        
        // Refresh patients list to update status indicators
        await loadPatients();

        // Send notification
        const notif = {
          type: data.is_tumor ? "alert" : "success",
          message: `New scan analyzed for ${selectedPatient.full_name}: ${data.prediction}`,
          timestamp: new Date().toISOString(),
          scan_id: data.scan_id,
        };
        socket.emit("send_notification", {
          recipient_id: selectedPatient.id,
          notification: notif,
        });

        showToast("Analysis completed successfully", "success");
      }

      await loadUsageStatus();
      await loadDashboardStats();
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err.message || "Failed to perform analysis.");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  }

  async function loadGradCAM(scanId) {
    try {
      const res = await fetch(`${API_BASE}/gradcam/${scanId}`, {
        credentials: "include",
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setGradcamData(url);
      }
    } catch (err) {
      console.error("Failed to load Grad-CAM:", err);
    }
  }

  async function downloadPDF(scanId) {
    try {
      const res = await fetch(`${API_BASE}/generate-report/${scanId}`, {
        credentials: "include",
      });

      if (!res.ok) throw new Error("Failed to generate report");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `NeuroScan_Report_${scanId}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      showToast("Report downloaded successfully", "success");
    } catch (err) {
      console.error("PDF download failed:", err);
      setError("Failed to download PDF report");
    }
  }

  function resetInlineForm() {
    setNewPatientName("");
    setNewPatientEmail("");
    setNewPatientPhone("");
    setNewPatientDob("");
    setNewPatientGender("");
    setNewPatientAddress("");
    setNewPatientEmergencyContact("");
    setNewPatientEmergencyPhone("");
  }

  async function handleAnalyzeClick() {
    if (!selectedFile) {
      setError("Please select an image file first");
      return;
    }

    if (!newPatientName.trim() || !newPatientEmail.trim()) {
      setError("Patient name and email are required");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const payload = {
        full_name: newPatientName.trim(),
        email: newPatientEmail.trim(),
        phone: newPatientPhone.trim(),
        date_of_birth: newPatientDob,
        gender: newPatientGender,
        address: newPatientAddress.trim(),
        emergency_contact: newPatientEmergencyContact.trim(),
        emergency_phone: newPatientEmergencyPhone.trim(),
      };

      const res = await fetch(`${API_BASE}/hospital/patients`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to create patient");
      }

      const data = await res.json();
      const patient = data.patient;

      if (patient && patient.id) {
        setSelectedPatient(patient);
        await loadPatients();
        await performAnalysisWithPatient(patient.id);
        resetInlineForm();
        showToast("Patient created & scan analyzed!", "success");
      }
    } catch (err) {
      console.error("Patient creation / analysis failed:", err);
      setError(err.message || "Unable to create patient for scan.");
      setLoading(false);
    }
  }

  async function loadExpandedPatientScans(patientId) {
    try {
      const res = await fetch(
        `${API_BASE}/hospital/patient-scans/${patientId}`,
        { credentials: "include" }
      );
      if (res.ok) {
        const data = await res.json();
        setExpandedPatientScans(data.scans || []);
      }
    } catch (err) {
      console.error("Error loading expanded patient scans:", err);
    }
  }

  // ──── Subscription Management Helpers ────
  function getPlanDisplayName(planType) {
    const names = { free: "Free Plan", basic: "Basic Plan", premium: "Premium Plan", enterprise: "Enterprise Plan" };
    return names[planType] || "Free Plan";
  }

  async function loadSubscriptionPlans() {
    setLoadingPlans(true);
    try {
      const res = await fetch(`${API_BASE}/api/subscription/plans`, { credentials: "include" });
      const data = await res.json();
      setSubscriptionPlans(data.plans || []);

      // Initialize Stripe if not done
      if (!stripePromise) {
        try {
          const cfgRes = await fetch(`${API_BASE}/api/stripe/config`, { credentials: "include" });
          const cfg = await cfgRes.json();
          if (cfg.publishableKey) stripePromise = loadStripe(cfg.publishableKey);
        } catch (e) { console.error("Stripe init error:", e); }
      }
    } catch (err) {
      console.error("Failed to load plans:", err);
    } finally {
      setLoadingPlans(false);
    }
  }

  async function handleUpgrade(plan) {
    if (!plan || plan.price_monthly === 0) return;
    setProcessingPlan(plan.id);
    try {
      const res = await fetch(`${API_BASE}/api/stripe/create-checkout-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ plan_id: plan.id, billing_cycle: billingCycle }),
      });
      const { url } = await res.json();
      if (url) window.location.href = url;
    } catch (err) {
      console.error("Checkout error:", err);
      showToast("Failed to start checkout. Please try again.", "error");
    } finally {
      setProcessingPlan(null);
    }
  }

  async function performAnalysisWithPatient(patientId) {
    if (!selectedFile || !patientId) return;

    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("patient_id", patientId.toString());
    formData.append("generate_gradcam", "true");

    try {
      const res = await fetch(`${API_BASE}/hospital/predict`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        // Handle scan limit reached (403)
        if (res.status === 403 && errorData.upgrade_required) {
          setUpgradeMessage(errorData.message || `You've reached your monthly scan limit. Upgrade your plan to continue scanning.`);
          setShowUpgradeModal(true);
          setLoading(false);
          return;
        }
        throw new Error(errorData.message || errorData.error);
      }

      const data = await res.json();

      if (data.error) {
        setError(data.message || data.error);
      } else {
        setPrediction(data);

        if (data.gradcam_available) {
          loadGradCAM(data.scan_id);
        }

        await loadPatientScans(patientId);
        await loadPatients();
      }

      await loadUsageStatus();
      await loadDashboardStats();
    } catch (err) {
      console.error("Analysis failed:", err);
      setError(err.message || "Failed to perform analysis.");
    } finally {
      setLoading(false);
    }
  }

  function showToast(message, type = "info") {
    // Simple toast notification (you can replace with a proper toast library)
    const toast = document.createElement("div");
    toast.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      padding: 16px 24px;
      background: ${type === "success" ? "#10b981" : type === "error" ? "#ef4444" : "#2563eb"};
      color: white;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      z-index: 10000;
      animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = "slideOut 0.3s ease";
      setTimeout(() => document.body.removeChild(toast), 300);
    }, 3000);
  }

  // function startVideoCall() {
  //   if (!selectedPatient) {
  //     showToast("Please select a patient first", "error");
  //     return;
  //   }
  //   setCallType("video");
  //   setShowVideoCall(true);
  // }

  // function startAudioCall() {
  //   if (!selectedPatient) {
  //     showToast("Please select a patient first", "error");
  //     return;
  //   }
  //   setCallType("audio");
  //   setShowVideoCall(true);
  // }

  const bgColor = darkMode ? "#0f172a" : "#f8fafc";
  const textPrimary = darkMode ? "#f1f5f9" : "#0f172a";
  const textSecondary = darkMode ? "#94a3b8" : "#64748b";

  const DashboardTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: darkMode ? '#1e293b' : '#ffffff',
          padding: '12px 16px',
          borderRadius: '12px',
          border: `1px solid ${darkMode ? '#334155' : '#e2e8f0'}`,
          boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)',
          color: textPrimary
        }}>
          <p style={{ margin: '0 0 8px 0', fontSize: '11px', fontWeight: '600', color: textSecondary, textTransform: 'uppercase' }}>
            {label}
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {payload.map((entry, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: entry.color }} />
                <span style={{ fontSize: '14px', fontWeight: '600', color: textPrimary }}>
                  {entry.name}: {entry.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={darkMode ? "dark" : ""}>
    <div style={{ display: "flex", height: "100vh", width: "100vw", overflow: "hidden", background: "#1e293b", padding: "20px 20px 0 0" }}>
        {/* Sidebar */}
        <aside
          style={{
            width: 280,
            background: 'linear-gradient(180deg, #1e293b 0%, #334155 100%)',
            padding: "24px 20px",
            display: "flex",
            flexDirection: "column",
            position: 'relative',
            overflow: 'hidden',
            zIndex: 10
          }}
        >
          {/* Decorative Glow Orbs */}
          <div style={{
            position: 'absolute',
            top: '-20%',
            left: '-30%',
            width: '200px',
            height: '200px',
            background: 'radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
            borderRadius: '50%',
            pointerEvents: 'none'
          }} />
          <div style={{
            position: 'absolute',
            bottom: '10%',
            right: '-20%',
            width: '150px',
            height: '150px',
            background: 'radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 70%)',
            borderRadius: '50%',
            pointerEvents: 'none'
          }} />

          {/* Logo Section */}
          <div style={{ marginBottom: "32px", position: 'relative', zIndex: 1 }}>
            <h1
              style={{
                fontSize: "26px",
                fontWeight: "800",
                color: "#60a5fa",
                display: "flex",
                alignItems: "center",
                gap: "10px",
                textShadow: '0 0 30px rgba(96, 165, 250, 0.4)'
              }}
            >
              <Brain size={30} /> NeuroScan
            </h1>
            <p
              style={{
                fontSize: "13px",
                color: "rgba(148, 163, 184, 0.9)",
                marginTop: "6px",
                fontWeight: '500',
                letterSpacing: '0.5px'
              }}
            >
              Hospital Portal
            </p>
          </div>


          
          <nav style={{ flex: 1, overflowY: "auto", marginBottom: "16px" }}>
            <NavItem
              icon={<LayoutDashboard size={20} />}
              label="Dashboard"
              active={view === "dashboard"}
              onClick={() => setView("dashboard")}
            />
            <NavItem
              icon={<Upload size={20} />}
              label="New Scan"
              active={view === "scan"}
              onClick={() => setView("scan")}
            />
            <NavItem
              icon={<Users size={20} />}
              label="Patients"
              active={view === "patients"}
              onClick={() => setView("patients")}
            />
            <NavItem
              icon={<MessageCircle size={20} />}
              label="Chat"
              active={view === "chat"}
              onClick={() => setView("chat")}
              badge={unreadCount > 0 ? unreadCount : null}
            />
          </nav>

          {/* Current Plan Button */}
          <button
            onClick={() => { setView("subscription"); loadSubscriptionPlans(); }}
            style={{
              width: "100%",
              padding: "12px 14px",
              display: "flex",
              alignItems: "center",
              gap: "10px",
              background: view === "subscription" ? "rgba(59, 130, 246, 0.2)" : "rgba(59, 130, 246, 0.08)",
              border: `1px solid ${view === "subscription" ? "rgba(59, 130, 246, 0.5)" : "rgba(59, 130, 246, 0.2)"}`,
              borderRadius: "12px",
              cursor: "pointer",
              marginBottom: "12px",
              color: "white",
              transition: "all 0.25s ease",
              position: 'relative',
              zIndex: 1,
              textAlign: "left"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(59, 130, 246, 0.2)";
              e.currentTarget.style.borderColor = "rgba(59, 130, 246, 0.5)";
            }}
            onMouseLeave={(e) => {
              if (view !== "subscription") {
                e.currentTarget.style.background = "rgba(59, 130, 246, 0.08)";
                e.currentTarget.style.borderColor = "rgba(59, 130, 246, 0.2)";
              }
            }}
          >
            <CreditCard size={18} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: "13px", fontWeight: "600" }}>Current Plan</div>
              <div style={{ fontSize: "11px", opacity: 0.8 }}>
                {usage ? getPlanDisplayName(usage.plan_type) : "Loading..."}
              </div>
            </div>
          </button>

          <button
            onClick={onLogout}
            style={{
              width: "100%",
              padding: "12px 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-start",
              gap: "10px",
              background: "transparent",
              color: "rgba(239, 68, 68, 0.9)",
              border: "1px solid rgba(239, 68, 68, 0.2)",
              borderRadius: "12px",
              cursor: "pointer",
              fontWeight: "500",
              fontSize: "14px",
              transition: "all 0.25s ease",
              position: 'relative',
              zIndex: 1
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(239, 68, 68, 0.1)";
              e.currentTarget.style.borderColor = "rgba(239, 68, 68, 0.4)";
              e.currentTarget.style.color = "#fca5a5";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.borderColor = "rgba(239, 68, 68, 0.2)";
              e.currentTarget.style.color = "rgba(239, 68, 68, 0.9)";
            }}
          >
            <LogOut size={18} />
            Logout
          </button>
        </aside>

        {/* Main Content */}
        <main style={{ 
          flex: 1, 
          padding: "40px", 
          overflowY: "auto",
          overflowX: "hidden",
          background: "#f8fafc",
          borderRadius: "32px 0 0 0",
          position: "relative",
          zIndex: 5,
          boxShadow: "-5px 5px 30px rgba(0, 0, 0, 0.05)"
        }}>
          {/* Header with Notifications */}
          <div style={{ 
            display: "flex", 
            justifyContent: "space-between", 
            alignItems: "center",
            marginBottom: "32px"
          }}>
            <div>
              <h2 style={{ fontSize: "28px", fontWeight: "700", color: textPrimary, margin: 0 }}>
                {view.charAt(0).toUpperCase() + view.slice(1)}
              </h2>
              <p style={{ margin: "4px 0 0 0", color: "#475569", fontSize: "14px", fontWeight: "500" }}>
                Welcome back, {user.full_name}
              </p>
            </div>
            
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              {/* Upgrade Button - Only show for Free or Basic tier */}
              {usage && (usage.plan_type === "free" || usage.plan_type === "basic") && (
                <button
                  onClick={() => { setView("subscription"); loadSubscriptionPlans(); }}
                  style={{
                    padding: "12px 20px",
                    background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                    border: "none",
                    borderRadius: "12px",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    color: "white",
                    fontWeight: "700",
                    fontSize: "14px",
                    boxShadow: "0 4px 12px rgba(245, 158, 11, 0.3)",
                    transition: "all 0.2s ease"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-2px)";
                    e.currentTarget.style.boxShadow = "0 6px 16px rgba(245, 158, 11, 0.4)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "0 4px 12px rgba(245, 158, 11, 0.3)";
                  }}
                >
                  <Zap size={18} />
                  Upgrade Plan
                </button>
              )}
            
              <button
                onClick={() => setShowNotifications(!showNotifications)}
                style={{
                  padding: "12px",
                  background: darkMode ? "#1e293b" : "#ffffff",
                  border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                  borderRadius: "12px",
                  cursor: "pointer",
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  color: textPrimary,
                  boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
                }}
              >
                <Bell size={20} />
                <span style={{ fontWeight: "600", fontSize: "14px" }}>Notifications</span>
                {unreadCount > 0 && (
                  <span style={{
                    background: "#ef4444",
                    color: "white",
                    borderRadius: "50%",
                    minWidth: "20px",
                    height: "20px",
                    padding: "0 6px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "11px",
                    fontWeight: "bold",
                  }}>
                    {unreadCount}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Dashboard View */}
          {view === "dashboard" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
              {/* Stats Cards */}
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", 
                gap: "20px" 
              }}>
                <DashboardCard 
                  title="Total Patients" 
                  value={dashboardStats?.total_patients || 0} 
                  icon={<Users color="#2563eb" size={24} />}
                  darkMode={darkMode}
                  bgGradient="linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)"
                />
                <DashboardCard 
                  title="Total Scans" 
                  value={dashboardStats?.total_scans || 0} 
                  icon={<Brain color="#0891b2" size={24} />}
                  darkMode={darkMode}
                  bgGradient="linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%)"
                />
                <DashboardCard 
                  title="Tumor Positive" 
                  value={dashboardStats?.tumor_patients || 0} 
                  icon={<AlertCircle color="#e11d48" size={24} />}
                  darkMode={darkMode}
                  subtitle="Unique patients"
                  bgGradient="linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%)"
                />
                <DashboardCard 
                  title="Tumor Negative" 
                  value={dashboardStats?.normal_patients || 0} 
                  icon={<CheckCircle color="#059669" size={24} />}
                  darkMode={darkMode}
                  subtitle="Unique patients"
                  bgGradient="linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
                />
              </div>

              {/* Scan Usage Bar */}
              {usage && usage.scan_limit !== -1 && (
                <div style={{
                  background: darkMode ? "#1e293b" : "white",
                  padding: "20px 24px",
                  borderRadius: "16px",
                  border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                  boxShadow: "0 2px 8px rgba(0,0,0,0.04)"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Zap size={16} color={usage.usage_percent >= 80 ? "#ef4444" : "#3b82f6"} />
                      <span style={{ fontSize: "14px", fontWeight: "600", color: textPrimary }}>Monthly Scan Usage</span>
                      <span style={{
                        padding: "2px 8px",
                        borderRadius: "6px",
                        fontSize: "11px",
                        fontWeight: "700",
                        background: usage.is_free_tier ? "rgba(148,163,184,0.15)" : "rgba(59,130,246,0.1)",
                        color: usage.is_free_tier ? "#64748b" : "#3b82f6"
                      }}>{usage.plan_name || usage.plan_type}</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                      <span style={{ fontSize: "13px", fontWeight: "600", color: usage.usage_percent >= 80 ? "#ef4444" : textSecondary }}>
                        {usage.scans_used}/{usage.scan_limit} scans
                      </span>
                      {usage.usage_percent >= 80 && (
                        <button
                          onClick={() => { setView("subscription"); loadSubscriptionPlans(); }}
                          style={{
                            padding: "4px 12px",
                            background: "linear-gradient(135deg, #f59e0b, #d97706)",
                            color: "white",
                            border: "none",
                            borderRadius: "6px",
                            fontSize: "11px",
                            fontWeight: "700",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            gap: "4px"
                          }}
                        >
                          <Zap size={12} /> Upgrade
                        </button>
                      )}
                    </div>
                  </div>
                  <div style={{
                    width: "100%",
                    height: "8px",
                    background: darkMode ? "#0f172a" : "#f1f5f9",
                    borderRadius: "4px",
                    overflow: "hidden"
                  }}>
                    <div style={{
                      width: `${Math.min(usage.usage_percent || 0, 100)}%`,
                      height: "100%",
                      borderRadius: "4px",
                      background: usage.usage_percent >= 90 ? "linear-gradient(90deg, #ef4444, #dc2626)"
                        : usage.usage_percent >= 80 ? "linear-gradient(90deg, #f59e0b, #d97706)"
                        : "linear-gradient(90deg, #3b82f6, #8b5cf6)",
                      transition: "width 0.5s ease"
                    }} />
                  </div>
                  {usage.is_blocked && (
                    <div style={{
                      marginTop: "10px",
                      padding: "8px 12px",
                      background: "rgba(239,68,68,0.08)",
                      borderRadius: "8px",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px"
                    }}>
                      <AlertTriangle size={14} color="#ef4444" />
                      <span style={{ fontSize: "13px", color: "#ef4444", fontWeight: "600" }}>
                        {usage.block_message || "Scan limit reached. Upgrade to continue."}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Charts & Calendar Area */}
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "1.6fr 1fr", 
                gap: "24px",
                alignItems: "stretch"
              }}>
                <div style={{ 
                  background: darkMode ? "#1e293b" : "white",
                  padding: "24px",
                  borderRadius: "20px",
                  border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                  boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)",
                  backdropFilter: "blur(10px)",
                  display: "flex",
                  flexDirection: "column"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
                    <h3 style={{ margin: 0, fontSize: "18px", fontWeight: "600", color: textPrimary }}>
                      Patient Analysis Trends
                    </h3>
                    <div style={{ display: "flex", background: darkMode ? "#0f172a" : "#f1f5f9", padding: "4px", borderRadius: "8px" }}>
                      <button 
                        onClick={() => setChartFilter("daily")}
                        style={{
                          padding: "6px 12px",
                          background: chartFilter === "daily" ? (darkMode ? "#334155" : "white") : "transparent",
                          border: "none",
                          borderRadius: "6px",
                          cursor: "pointer",
                          fontSize: "13px",
                          fontWeight: "600",
                          color: chartFilter === "daily" ? "#2563eb" : textSecondary,
                          boxShadow: chartFilter === "daily" ? "0 2px 4px rgba(0,0,0,0.05)" : "none"
                        }}
                      >
                        Daily
                      </button>
                      <button 
                        onClick={() => setChartFilter("weekly")}
                        style={{
                          padding: "6px 12px",
                          background: chartFilter === "weekly" ? (darkMode ? "#334155" : "white") : "transparent",
                          border: "none",
                          borderRadius: "6px",
                          cursor: "pointer",
                          fontSize: "13px",
                          fontWeight: "600",
                          color: chartFilter === "weekly" ? "#2563eb" : textSecondary,
                          boxShadow: chartFilter === "weekly" ? "0 2px 4px rgba(0,0,0,0.05)" : "none"
                        }}
                      >
                        Weekly
                      </button>
                    </div>
                  </div>

                  <div style={{ height: "350px", width: "100%", flex: 1 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={chartFilter === "daily" ? (dashboardStats?.daily_stats?.length > 1 ? dashboardStats.daily_stats : [
                          { name: 'Mon', infected: 3, normal: 8 },
                          { name: 'Tue', infected: 5, normal: 12 },
                          { name: 'Wed', infected: 2, normal: 15 },
                          { name: 'Thu', infected: 4, normal: 10 },
                          { name: 'Fri', infected: 6, normal: 18 },
                          { name: 'Sat', infected: 3, normal: 14 },
                          { name: 'Sun', infected: 1, normal: 9 },
                        ]) : dashboardStats?.weekly_stats}
                        margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                      >
                        <defs>
                          <linearGradient id="colorInfected" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="colorNormal" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={darkMode ? "#334155" : "#f1f5f9"} opacity={0.5} />
                        <XAxis 
                          dataKey="name" 
                          stroke={textSecondary} 
                          fontSize={11} 
                          tickLine={false}
                          axisLine={false}
                          dy={10}
                          tickFormatter={(value) => chartFilter === "daily" ? value.split('-').slice(1).join('/') : value}
                        />
                        <YAxis stroke={textSecondary} fontSize={11} tickLine={false} axisLine={false} dx={-10} />
                        <RechartsTooltip content={<DashboardTooltip />} />
                        <Legend 
                          verticalAlign="top" 
                          align="right" 
                          height={36} 
                          iconType="circle"
                          wrapperStyle={{ fontSize: '12px', fontWeight: '500', color: textSecondary, paddingBottom: '20px' }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="infected" 
                          name="Tumor Detected"
                          stroke="#ef4444" 
                          strokeWidth={3}
                          fillOpacity={1} 
                          fill="url(#colorInfected)" 
                          activeDot={{ r: 6, strokeWidth: 0, fill: '#ef4444' }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="normal" 
                          name="No Tumor"
                          stroke="#10b981" 
                          strokeWidth={3}
                          fillOpacity={1} 
                          fill="url(#colorNormal)" 
                          activeDot={{ r: 6, strokeWidth: 0, fill: '#10b981' }}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Calendar Section */}
                <AppointmentCalendar appointments={appointments} darkMode={darkMode} />
              </div>

              {/* Bottom Row */}
              <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "24px" }}>
                {/* Recent Patients */}
                <div style={{ 
                  background: darkMode ? "#1e293b" : "white",
                  padding: "24px",
                  borderRadius: "16px",
                  border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
                    <h3 style={{ margin: 0, fontSize: "18px", fontWeight: "600", color: textPrimary }}>
                      Recent Patients
                    </h3>
                    <button 
                      onClick={() => setView("patients")}
                      style={{ background: "none", border: "none", color: "#2563eb", fontSize: "14px", fontWeight: "600", cursor: "pointer" }}
                    >
                      View All
                    </button>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                    {patients.slice(0, 5).map(p => (
                      <div key={p.id} style={{ 
                        display: "flex", 
                        alignItems: "center", 
                        gap: "12px", 
                        padding: "12px",
                        borderRadius: "12px",
                        background: darkMode ? "#0f172a" : "#f8fafc",
                        border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`
                      }}>
                        <div style={{ 
                          width: "40px", height: "40px", borderRadius: "10px", background: "#e0e7ff",
                          display: "flex", alignItems: "center", justifyContent: "center", color: "#2563eb", fontWeight: "bold"
                        }}>
                          {p.full_name?.charAt(0)}
                        </div>
                        <div style={{ flex: 1 }}>
                          <p style={{ margin: 0, fontWeight: "600", color: textPrimary, fontSize: "14px" }}>{p.full_name}</p>
                          <p style={{ margin: 0, fontSize: "12px", color: textSecondary }}>{p.email}</p>
                        </div>
                        <button 
                          onClick={() => { setSelectedPatient(p); setScanMode("existing"); setView("scan"); }}
                          style={{ padding: "6px 12px", background: "#2563eb", color: "white", border: "none", borderRadius: "6px", fontSize: "12px", cursor: "pointer" }}
                        >
                          Analyze
                        </button>
                      </div>
                    ))}
                    {patients.length === 0 && <p style={{ textAlign: "center", color: textSecondary }}>No patients found</p>}
                  </div>
                </div>

                {/* Quick Actions */}
                <div style={{ 
                  background: darkMode ? "#1e293b" : "white",
                  padding: "24px",
                  borderRadius: "16px",
                  border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`
                }}>
                  <h3 style={{ margin: "0 0 20px 0", fontSize: "18px", fontWeight: "600", color: textPrimary }}>
                    Quick Actions
                  </h3>
                  <div style={{ display: "grid", gap: "12px" }}>
                    <ActionButton 
                      icon={<Plus size={20} />} 
                      label="Add New Patient" 
                      onClick={() => setView("scan")} 
                      color="#10b981"
                      darkMode={darkMode}
                    />
                    <ActionButton 
                      icon={<Upload size={20} />} 
                      label="Upload MRI Scan" 
                      onClick={() => setView("scan")} 
                      color="#2563eb"
                      darkMode={darkMode}
                    />
                    <ActionButton 
                      icon={<MessageCircle size={20} />} 
                      label="Open Messages" 
                      onClick={() => setView("chat")} 
                      color="#8b5cf6"
                      darkMode={darkMode}
                    />
                    <ActionButton 
                      icon={<FileText size={20} />} 
                      label="Generate Reports" 
                      onClick={() => setView("patients")} 
                      color="#f59e0b"
                      darkMode={darkMode}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}



          {/* Scan View */}
          {view === "scan" && (
            <div>
              <h2
                style={{
                  fontSize: "36px",
                  fontWeight: "800",
                  marginBottom: "28px",
                  background: darkMode ? "linear-gradient(135deg, #60a5fa, #a78bfa)" : "linear-gradient(135deg, #2563eb, #6d28d9)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  display: "inline-block"
                }}
              >
                Brain Tumor Analysis
              </h2>

              {/* Patient Selection Section */}
              <div style={{
                marginBottom: "28px",
                padding: "28px",
                background: darkMode ? "rgba(30, 41, 59, 0.6)" : "rgba(255, 255, 255, 0.7)",
                backdropFilter: "blur(12px)",
                borderRadius: "20px",
                border: `1px solid ${darkMode ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)"}`,
                boxShadow: darkMode ? "0 8px 32px rgba(0,0,0,0.3)" : "0 8px 32px rgba(0,0,0,0.05)",
                overflow: "visible",
                position: "relative",
                zIndex: 10,
              }}>
                <h3 style={{ margin: "0 0 4px 0", fontSize: "18px", fontWeight: "700", color: textPrimary, display: "flex", alignItems: "center", gap: "8px" }}>
                  <User size={20} color="#2563eb" />
                  Patient Details
                </h3>
                <p style={{ margin: "0 0 16px 0", fontSize: "13px", color: textSecondary }}>
                  Select an existing patient or add a new one, then upload the MRI scan
                </p>

                {/* Mode Toggle: Existing / New Patient */}
                <div style={{ display: "flex", gap: "0", marginBottom: "20px", background: darkMode ? "#0f172a" : "#f1f5f9", borderRadius: "12px", padding: "4px" }}>
                  <button
                    onClick={() => { setScanMode("existing"); setError(null); }}
                    style={{
                      flex: 1, padding: "10px 16px",
                      background: scanMode === "existing" ? (darkMode ? "#334155" : "white") : "transparent",
                      border: "none", borderRadius: "10px", cursor: "pointer",
                      fontSize: "14px", fontWeight: "600",
                      color: scanMode === "existing" ? "#2563eb" : textSecondary,
                      display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
                      boxShadow: scanMode === "existing" ? "0 2px 8px rgba(0,0,0,0.08)" : "none",
                      transition: "all 0.2s ease"
                    }}
                  >
                    <Users size={16} />
                    Existing Patient
                  </button>
                  <button
                    onClick={() => { setScanMode("new"); setSelectedPatient(null); setError(null); }}
                    style={{
                      flex: 1, padding: "10px 16px",
                      background: scanMode === "new" ? (darkMode ? "#334155" : "white") : "transparent",
                      border: "none", borderRadius: "10px", cursor: "pointer",
                      fontSize: "14px", fontWeight: "600",
                      color: scanMode === "new" ? "#2563eb" : textSecondary,
                      display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
                      boxShadow: scanMode === "new" ? "0 2px 8px rgba(0,0,0,0.08)" : "none",
                      transition: "all 0.2s ease"
                    }}
                  >
                    <Plus size={16} />
                    New Patient
                  </button>
                </div>

                {/* ── Existing Patient Mode ── */}
                {scanMode === "existing" && (
                  <div>
                    {/* Selected patient chip */}
                    {selectedPatient ? (
                      <div style={{
                        display: "flex", alignItems: "center", gap: "12px",
                        padding: "14px 18px",
                        background: darkMode ? "rgba(37, 99, 235, 0.15)" : "rgba(37, 99, 235, 0.08)",
                        border: "1px solid rgba(37, 99, 235, 0.3)",
                        borderRadius: "14px",
                      }}>
                        <div style={{
                          width: "40px", height: "40px", borderRadius: "10px",
                          background: "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          color: "#2563eb", fontWeight: "700", fontSize: "16px"
                        }}>
                          {(selectedPatient.full_name || "?").charAt(0).toUpperCase()}
                        </div>
                        <div style={{ flex: 1 }}>
                          <p style={{ margin: 0, fontWeight: "700", color: textPrimary, fontSize: "15px" }}>
                            {selectedPatient.full_name}
                          </p>
                          <p style={{ margin: "2px 0 0 0", fontSize: "13px", color: textSecondary }}>
                            {selectedPatient.email || "No email"}{selectedPatient.phone ? ` • ${selectedPatient.phone}` : ""}
                          </p>
                        </div>
                        <button
                          onClick={() => { setSelectedPatient(null); setExistingPatientSearch(""); setPatientScans([]); }}
                          style={{
                            padding: "6px 12px", background: "rgba(239, 68, 68, 0.15)",
                            border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "8px",
                            cursor: "pointer", color: "#ef4444", fontSize: "13px", fontWeight: "600",
                            display: "flex", alignItems: "center", gap: "4px"
                          }}
                        >
                          <X size={14} /> Change
                        </button>
                      </div>
                    ) : (
                      /* Patient search input + dropdown */
                      <div style={{ position: "relative" }}>
                        <Search size={18} style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)", color: textSecondary, zIndex: 2 }} />
                        <input
                          type="text"
                          placeholder="Search patients by name or email..."
                          value={existingPatientSearch}
                          onChange={(e) => { setExistingPatientSearch(e.target.value); setShowExistingDropdown(true); }}
                          onFocus={() => setShowExistingDropdown(true)}
                          onBlur={() => setTimeout(() => setShowExistingDropdown(false), 200)}
                          style={{
                            width: "100%", padding: "12px 14px 12px 42px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary, outline: "none",
                            transition: "all 0.2s ease",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                        />
                        {/* Dropdown results */}
                        {showExistingDropdown && (
                          <div style={{
                            position: "absolute", top: "100%", left: 0, right: 0,
                            marginTop: "4px",
                            background: darkMode ? "#1e293b" : "white",
                            border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                            borderRadius: "12px",
                            boxShadow: "0 12px 32px rgba(0,0,0,0.15)",
                            maxHeight: "240px", overflowY: "auto",
                            zIndex: 50,
                          }}>
                            {patients
                              .filter(p => {
                                if (!existingPatientSearch.trim()) return true;
                                const q = existingPatientSearch.toLowerCase();
                                return (p.full_name || "").toLowerCase().includes(q) || (p.email || "").toLowerCase().includes(q);
                              })
                              .slice(0, 8)
                              .map((p) => (
                                <div
                                  key={p.id}
                                  onClick={() => {
                                    setSelectedPatient(p);
                                    setExistingPatientSearch("");
                                    setShowExistingDropdown(false);
                                    loadPatientScans(p.id);
                                  }}
                                  style={{
                                    padding: "12px 16px",
                                    display: "flex", alignItems: "center", gap: "10px",
                                    cursor: "pointer",
                                    borderBottom: `1px solid ${darkMode ? "#334155" : "#f1f5f9"}`,
                                    transition: "background 0.15s"
                                  }}
                                  onMouseEnter={(e) => e.currentTarget.style.background = darkMode ? "#334155" : "#f0f9ff"}
                                  onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                                >
                                  <div style={{
                                    width: "32px", height: "32px", borderRadius: "8px",
                                    background: "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    color: "#2563eb", fontWeight: "700", fontSize: "13px"
                                  }}>
                                    {(p.full_name || "?").charAt(0).toUpperCase()}
                                  </div>
                                  <div style={{ flex: 1 }}>
                                    <p style={{ margin: 0, fontWeight: "600", color: textPrimary, fontSize: "14px" }}>{p.full_name}</p>
                                    <p style={{ margin: 0, fontSize: "12px", color: textSecondary }}>{p.email || "No email"}</p>
                                  </div>
                                </div>
                              ))}
                            {patients.filter(p => {
                              if (!existingPatientSearch.trim()) return true;
                              const q = existingPatientSearch.toLowerCase();
                              return (p.full_name || "").toLowerCase().includes(q) || (p.email || "").toLowerCase().includes(q);
                            }).length === 0 && (
                              <div style={{ padding: "20px", textAlign: "center", color: textSecondary, fontSize: "14px" }}>
                                No patients found. Try a different search or add a new patient.
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* ── New Patient Mode ── */}
                {scanMode === "new" && (
                  <div>
                    {/* Row 1: Name, Email, Phone */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px", marginBottom: "16px" }}>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Full Name <span style={{ color: "#ef4444" }}>*</span>
                        </label>
                        <input
                          type="text"
                          value={newPatientName}
                          onChange={(e) => setNewPatientName(e.target.value)}
                          placeholder="e.g., John Doe"
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Email Address <span style={{ color: "#ef4444" }}>*</span>
                        </label>
                        <input
                          type="email"
                          value={newPatientEmail}
                          onChange={(e) => setNewPatientEmail(e.target.value)}
                          placeholder="patient@example.com"
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Phone Number
                        </label>
                        <input
                          type="tel"
                          value={newPatientPhone}
                          onChange={(e) => setNewPatientPhone(e.target.value)}
                          placeholder="+1234567890"
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                    </div>

                    {/* Row 2: Address, DOB, Gender */}
                    <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: "16px", marginBottom: "16px" }}>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Address
                        </label>
                        <input
                          type="text"
                          value={newPatientAddress}
                          onChange={(e) => setNewPatientAddress(e.target.value)}
                          placeholder="123 Main St, City"
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Date of Birth
                        </label>
                        <input
                          type="date"
                          value={newPatientDob}
                          onChange={(e) => setNewPatientDob(e.target.value)}
                          max={new Date().toISOString().split("T")[0]}
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)",
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Gender
                        </label>
                        <select
                          value={newPatientGender}
                          onChange={(e) => setNewPatientGender(e.target.value)}
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: newPatientGender ? textPrimary : textSecondary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)",
                            cursor: "pointer",
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        >
                          <option value="">Select</option>
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                          <option value="Other">Other</option>
                        </select>
                      </div>
                    </div>

                    {/* Row 3: Emergency Contact */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Emergency Contact Name
                        </label>
                        <input
                          type="text"
                          value={newPatientEmergencyContact}
                          onChange={(e) => setNewPatientEmergencyContact(e.target.value)}
                          placeholder="e.g., Jane Doe"
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                      <div>
                        <label style={{ display: "block", fontSize: "13px", fontWeight: "600", color: textSecondary, marginBottom: "6px" }}>
                          Emergency Contact Phone
                        </label>
                        <input
                          type="tel"
                          value={newPatientEmergencyPhone}
                          onChange={(e) => setNewPatientEmergencyPhone(e.target.value)}
                          placeholder="+1234567890"
                          style={{
                            width: "100%", padding: "12px 14px",
                            border: `1px solid ${darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"}`,
                            borderRadius: "12px", fontSize: "14px",
                            boxSizing: "border-box",
                            background: darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)",
                            color: textPrimary,
                            transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                            outline: "none",
                            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                          }}
                          onFocus={(e) => { e.target.style.borderColor = "#3b82f6"; e.target.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.15), inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.6)" : "#ffffff"; }}
                          onBlur={(e) => { e.target.style.borderColor = darkMode ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.1)"; e.target.style.boxShadow = "inset 0 2px 4px rgba(0,0,0,0.02)"; e.target.style.background = darkMode ? "rgba(15,23,42,0.4)" : "rgba(248,250,252,0.6)"; }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Validation Warning */}
              {validationWarning && (
                <div
                  style={{
                    padding: "16px",
                    background:
                      validationWarning.type === "error"
                        ? "#fee2e2"
                        : "#fef3c7",
                    border: `1px solid ${validationWarning.type === "error" ? "#ef4444" : "#f59e0b"}`,
                    borderRadius: "8px",
                    marginBottom: "24px",
                    display: "flex",
                    gap: "12px",
                  }}
                >
                  <AlertTriangle
                    size={24}
                    color={
                      validationWarning.type === "error" ? "#dc2626" : "#d97706"
                    }
                  />
                  <div style={{ flex: 1 }}>
                    <p
                      style={{
                        margin: "0 0 8px 0",
                        fontWeight: "600",
                        color:
                          validationWarning.type === "error"
                            ? "#dc2626"
                            : "#92400e",
                      }}
                    >
                      {validationWarning.message}
                    </p>
                    {validationWarning.warnings && (
                      <ul
                        style={{
                          margin: 0,
                          paddingLeft: "20px",
                          color:
                            validationWarning.type === "error"
                              ? "#dc2626"
                              : "#92400e",
                        }}
                      >
                        {validationWarning.warnings.map((w, i) => (
                          <li key={i} style={{ fontSize: "14px" }}>
                            {w}
                          </li>
                        ))}
                      </ul>
                    )}
                    <p
                      style={{
                        margin: "8px 0 0 0",
                        fontSize: "13px",
                        fontStyle: "italic",
                        color:
                          validationWarning.type === "error"
                            ? "#dc2626"
                            : "#92400e",
                      }}
                    >
                      Validation confidence:{" "}
                      {validationWarning.confidence || "unknown"}
                    </p>
                  </div>
                </div>
              )}

              {/* Upload Area */}
              <div
                onClick={() => !selectedFile && fileInputRef.current?.click()}
                style={{
                  border: selectedFile ? `2px solid #10b981` : `2px dashed ${darkMode ? "#475569" : "#cbd5e1"}`,
                  borderRadius: "24px",
                  padding: selectedFile ? "24px" : "60px 40px",
                  textAlign: "center",
                  cursor: selectedFile ? "default" : "pointer",
                  background: selectedFile 
                    ? (darkMode ? "linear-gradient(135deg, rgba(6,78,59,0.8), rgba(6,95,70,0.8))" : "linear-gradient(135deg, rgba(209,250,229,0.8), rgba(167,243,208,0.8))")
                    : (darkMode ? "rgba(30, 41, 59, 0.4)" : "rgba(255, 255, 255, 0.5)"),
                  backdropFilter: "blur(12px)",
                  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                  marginBottom: "24px",
                  position: "relative",
                  boxShadow: selectedFile ? "0 10px 25px rgba(16,185,129,0.2)" : (darkMode ? "0 8px 32px rgba(0,0,0,0.2)" : "0 8px 32px rgba(0,0,0,0.03)")
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.borderColor = "#2563eb";
                  e.currentTarget.style.background = darkMode
                    ? "#334155"
                    : "#f1f5f9";
                }}
                onDragLeave={(e) => {
                  e.currentTarget.style.borderColor = darkMode
                    ? "#475569"
                    : "#cbd5e1";
                  e.currentTarget.style.background = darkMode
                    ? "#1e293b"
                    : "white";
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.borderColor = darkMode
                    ? "#475569"
                    : "#cbd5e1";
                  e.currentTarget.style.background = darkMode
                    ? "#1e293b"
                    : "white";
                  const file = e.dataTransfer.files[0];
                  if (file) {
                    handleFile({ target: { files: [file] } });
                  }
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFile}
                  style={{ display: "none" }}
                />
                {selectedFile ? (
                  // File selected state with thumbnail
                  <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
                    {preview && (
                      <img 
                        src={preview} 
                        alt="MRI thumbnail" 
                        style={{ 
                          width: "80px", 
                          height: "80px", 
                          borderRadius: "12px", 
                          objectFit: "cover",
                          border: "2px solid rgba(255, 255, 255, 0.5)",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
                        }} 
                      />
                    )}
                    <div style={{ flex: 1, textAlign: "left" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
                        <CheckCircle size={24} color="#10b981" />
                        <p style={{ margin: 0, fontSize: "16px", fontWeight: "700", color: darkMode ? "#d1fae5" : "#065f46" }}>
                          File Ready: {selectedFile.name}
                        </p>
                      </div>
                      <p style={{ margin: 0, fontSize: "13px", color: darkMode ? "#a7f3d0" : "#047857" }}>
                        {(selectedFile.size / 1024).toFixed(2)} KB • Click analyze button below
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedFile(null);
                        setPreview(null);
                      }}
                      style={{
                        padding: "8px",
                        background: "rgba(239, 68, 68, 0.2)",
                        border: "1px solid rgba(239, 68, 68, 0.4)",
                        borderRadius: "8px",
                        cursor: "pointer",
                        color: "#ef4444",
                        display: "flex",
                        alignItems: "center"
                      }}
                    >
                      <X size={20} />
                    </button>
                  </div>
                ) : (
                  // Empty state
                  <>
                    <Upload
                      size={48}
                      color={darkMode ? "#94a3b8" : "#64748b"}
                      style={{ margin: "0 auto 16px" }}
                    />
                    <p
                      style={{
                        fontSize: "18px",
                        fontWeight: "600",
                        marginBottom: "8px",
                        color: textPrimary,
                      }}
                    >
                      Upload MRI Scan
                    </p>
                    <p style={{ fontSize: "14px", color: textSecondary, marginBottom: "16px" }}>
                      Click or drag & drop an MRI image (JPEG, PNG)
                    </p>
                    {/* Contextual Help */}
                    <div style={{ display: "flex", justifyContent: "center", gap: "16px", marginTop: "8px" }}>
                      <a 
                        href="#" 
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          alert("Sample MRI formats:\n\n• JPEG, JPG, PNG formats\n• Minimum 256x256 pixels\n• Maximum 10MB file size\n• T1-weighted or T2-weighted MRI scans");
                        }}
                        style={{ 
                          fontSize: "12px", 
                          color: "#2563eb", 
                          textDecoration: "none",
                          fontWeight: "600",
                          display: "flex",
                          alignItems: "center",
                          gap: "4px"
                        }}
                      >
                        <Eye size={14} />
                        Requirements
                      </a>
                      <span style={{ color: textSecondary }}>•</span>
                      <a 
                        href="#" 
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          alert("For best results:\n\n• Use axial view MRI scans\n• Ensure clear brain tissue visibility\n• Avoid motion artifacts\n• Preferred resolution: 512x512 or higher");
                        }}
                        style={{ 
                          fontSize: "12px", 
                          color: "#2563eb", 
                          textDecoration: "none",
                          fontWeight: "600",
                          display: "flex",
                          alignItems: "center",
                          gap: "4px"
                        }}
                      >
                        <AlertCircle size={14} />
                        Best Practices
                      </a>
                    </div>
                  </>
                )}
              </div>

              {/* Preview and Analysis */}
              {preview && (
                <div
                  style={{
                    background: darkMode ? "#1e293b" : "white",
                    borderRadius: "16px",
                    padding: "24px",
                    marginBottom: "24px",
                    border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: gradcamData ? "1fr 1fr" : "1fr",
                      gap: "24px",
                    }}
                  >
                    <div>
                      <h4
                        style={{
                          margin: "0 0 16px 0",
                          fontWeight: "600",
                          color: textPrimary,
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                        }}
                      >
                        <ImageIcon size={20} />
                        Original MRI Scan
                      </h4>
                      <img
                        src={preview}
                        alt="MRI Preview"
                        style={{
                          width: "100%",
                          height: "auto",
                          borderRadius: "12px",
                          border: `2px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                        }}
                      />
                    </div>

                    {gradcamData && (
                      <div>
                        <h4
                          style={{
                            margin: "0 0 16px 0",
                            fontWeight: "600",
                            color: textPrimary,
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                          }}
                        >
                          <Eye size={20} />
                          Grad-CAM Visualization
                        </h4>
                        <img
                          src={gradcamData}
                          alt="Grad-CAM"
                          style={{
                            width: "100%",
                            height: "auto",
                            borderRadius: "12px",
                            border: `2px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                          }}
                        />
                        <p
                          style={{
                            marginTop: "12px",
                            fontSize: "13px",
                            color: textSecondary,
                            fontStyle: "italic",
                          }}
                        >
                          Heat map showing regions of interest identified by the
                          AI model
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {error && (
                <div
                  style={{
                    padding: "16px",
                    background: "#fee2e2",
                    border: "1px solid #ef4444",
                    borderRadius: "8px",
                    marginBottom: "16px",
                    display: "flex",
                    gap: "12px",
                    alignItems: "center",
                  }}
                >
                  <AlertCircle size={20} color="#dc2626" />
                  <p style={{ margin: 0, color: "#dc2626" }}>{error}</p>
                </div>
              )}

              <button
                onClick={scanMode === "existing" ? performAnalysis : handleAnalyzeClick}
                disabled={!selectedFile || loading || (scanMode === "existing" && !selectedPatient)}
                onMouseEnter={(e) => {
                  if (!selectedFile || loading) return;
                  e.currentTarget.style.transform = "translateY(-2px)";
                  e.currentTarget.style.boxShadow = "0 12px 24px rgba(37, 99, 235, 0.3)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow = "0 4px 12px rgba(37, 99, 235, 0.2)";
                }}
                style={{
                  width: "100%",
                  padding: "18px",
                  background: (loading || !selectedFile || (scanMode === "existing" && !selectedPatient)) ? (darkMode ? "#334155" : "#cbd5e1") : "linear-gradient(135deg, #2563eb, #4f46e5)",
                  color: (loading || !selectedFile || (scanMode === "existing" && !selectedPatient)) ? (darkMode ? "#94a3b8" : "#64748b") : "white",
                  border: "none",
                  borderRadius: "16px",
                  fontSize: "16px",
                  fontWeight: "700",
                  cursor: (loading || !selectedFile || (scanMode === "existing" && !selectedPatient)) ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "12px",
                  transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                  boxShadow: (loading || !selectedFile || (scanMode === "existing" && !selectedPatient)) ? "none" : "0 4px 12px rgba(37, 99, 235, 0.2)",
                }}
              >
                {loading ? (
                  <>
                    <Loader className="animate-spin" size={24} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain size={24} />
                    Analyze Scan
                  </>
                )}
              </button>

              {prediction && (
                <div ref={resultsRef} style={{ marginTop: "24px" }}>
                  <FixedAnalysisResults
                    prediction={prediction}
                    darkMode={darkMode}
                    onDownloadPDF={() => downloadPDF(prediction.scan_id)}
                  />
                </div>
              )}

              {/* Tumor Progression Tracker */}
              {selectedPatient && patientScans.length > 1 && (
                <div style={{ marginTop: "32px" }}>
                  <TumourProgressionTracker
                    scans={patientScans}
                    darkMode={darkMode}
                  />
                </div>
              )}
            </div>
          )}

          {/* Patients View */}
          {view === "patients" && (
            <div>
              <h2 style={{ fontSize: "32px", fontWeight: "bold", marginBottom: "24px", color: textPrimary }}>
                Patient Management
              </h2>

              {/* Search Bar */}
              <div style={{ marginBottom: "20px", position: "relative" }}>
                <Search size={18} style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)", color: textSecondary }} />
                <input
                  type="text"
                  placeholder="Search patients by name or email..."
                  value={patientSearch}
                  onChange={(e) => setPatientSearch(e.target.value)}
                  style={{
                    width: "100%", padding: "12px 12px 12px 42px",
                    border: `1px solid ${darkMode ? "#475569" : "#d1d5db"}`,
                    borderRadius: "12px", fontSize: "14px",
                    boxSizing: "border-box",
                    background: darkMode ? "#1e293b" : "white",
                    color: textPrimary, outline: "none",
                    transition: "border-color 0.2s"
                  }}
                  onFocus={(e) => e.target.style.borderColor = "#2563eb"}
                  onBlur={(e) => e.target.style.borderColor = darkMode ? "#475569" : "#d1d5db"}
                />
              </div>

              {/* Patients Table */}
              <div style={{
                background: darkMode ? "#1e293b" : "white",
                borderRadius: "16px",
                border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                overflow: "hidden",
                boxShadow: "0 2px 8px rgba(0,0,0,0.04)"
              }}>
                {/* Table Header */}
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "2fr 2fr 1.5fr 1.5fr 60px",
                  padding: "14px 20px",
                  background: darkMode ? "#0f172a" : "#f8fafc",
                  borderBottom: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                  fontSize: "12px", fontWeight: "700", color: textSecondary,
                  textTransform: "uppercase", letterSpacing: "0.5px"
                }}>
                  <span>Name</span>
                  <span>Email</span>
                  <span>Phone</span>
                  <span>Added</span>
                  <span></span>
                </div>

                {/* Table Rows */}
                {patients
                  .filter(p => {
                    if (!patientSearch.trim()) return true;
                    const q = patientSearch.toLowerCase();
                    return (p.full_name || "").toLowerCase().includes(q) || (p.email || "").toLowerCase().includes(q);
                  })
                  .map((patient) => (
                  <div key={patient.id}>
                    {/* Row */}
                    <div
                      onClick={() => {
                        if (expandedPatientId === patient.id) {
                          setExpandedPatientId(null);
                          setExpandedPatientScans([]);
                        } else {
                          setExpandedPatientId(patient.id);
                          setSelectedPatient(patient);
                          loadExpandedPatientScans(patient.id);
                        }
                      }}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "2fr 2fr 1.5fr 1.5fr 60px",
                        padding: "16px 20px",
                        borderBottom: `1px solid ${darkMode ? "#334155" : "#f1f5f9"}`,
                        cursor: "pointer",
                        transition: "background 0.15s",
                        alignItems: "center",
                        background: expandedPatientId === patient.id ? (darkMode ? "#334155" : "#f0f9ff") : "transparent"
                      }}
                      onMouseEnter={(e) => { if (expandedPatientId !== patient.id) e.currentTarget.style.background = darkMode ? "#253044" : "#f8fafc"; }}
                      onMouseLeave={(e) => { if (expandedPatientId !== patient.id) e.currentTarget.style.background = "transparent"; }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                        <div style={{
                          width: "36px", height: "36px", borderRadius: "10px",
                          background: "linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          color: "#2563eb", fontWeight: "700", fontSize: "14px"
                        }}>
                          {(patient.full_name || "?").charAt(0).toUpperCase()}
                        </div>
                        <span style={{ fontWeight: "600", color: textPrimary, fontSize: "14px" }}>{patient.full_name}</span>
                      </div>
                      <span style={{ color: textSecondary, fontSize: "14px" }}>{patient.email || "—"}</span>
                      <span style={{ color: textSecondary, fontSize: "14px" }}>{patient.phone || "—"}</span>
                      <span style={{ color: textSecondary, fontSize: "13px" }}>
                        {patient.created_at ? new Date(patient.created_at).toLocaleDateString() : "—"}
                      </span>
                      <div style={{ display: "flex", justifyContent: "center" }}>
                        {expandedPatientId === patient.id ? <ChevronUp size={18} color={textSecondary} /> : <ChevronDown size={18} color={textSecondary} />}
                      </div>
                    </div>

                    {/* Expanded Panel */}
                    {expandedPatientId === patient.id && (
                      <div style={{
                        padding: "24px 24px 24px 66px",
                        background: darkMode ? "#0f172a" : "#f8fafc",
                        borderBottom: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`
                      }}>
                        {/* Patient Info Summary */}
                        <div style={{ display: "flex", gap: "32px", marginBottom: "24px", flexWrap: "wrap" }}>
                          <div>
                            <p style={{ margin: "0 0 2px 0", fontSize: "11px", color: textSecondary, textTransform: "uppercase", fontWeight: "600", letterSpacing: "0.5px" }}>Patient ID</p>
                            <p style={{ margin: 0, fontSize: "14px", fontWeight: "600", color: textPrimary }}>{patient.id}</p>
                          </div>
                          {patient.date_of_birth && (
                            <div>
                              <p style={{ margin: "0 0 2px 0", fontSize: "11px", color: textSecondary, textTransform: "uppercase", fontWeight: "600", letterSpacing: "0.5px" }}>Date of Birth</p>
                              <p style={{ margin: 0, fontSize: "14px", fontWeight: "600", color: textPrimary }}>{new Date(patient.date_of_birth).toLocaleDateString()}</p>
                            </div>
                          )}
                          {patient.gender && (
                            <div>
                              <p style={{ margin: "0 0 2px 0", fontSize: "11px", color: textSecondary, textTransform: "uppercase", fontWeight: "600", letterSpacing: "0.5px" }}>Gender</p>
                              <p style={{ margin: 0, fontSize: "14px", fontWeight: "600", color: textPrimary }}>{patient.gender}</p>
                            </div>
                          )}
                          <div style={{ marginLeft: "auto", display: "flex", gap: "8px" }}>
                            <button
                              onClick={(e) => { e.stopPropagation(); setSelectedPatient(patient); setScanMode("existing"); setView("scan"); }}
                              style={{
                                padding: "8px 16px", background: "#10b981", color: "white",
                                border: "none", borderRadius: "8px", cursor: "pointer",
                                fontSize: "13px", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px",
                                transition: "background 0.2s"
                              }}
                              onMouseEnter={(e) => e.currentTarget.style.background = "#059669"}
                              onMouseLeave={(e) => e.currentTarget.style.background = "#10b981"}
                            >
                              <Upload size={14} /> Upload New Scan
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); setSelectedPatient(patient); setShowChat(true); setView("chat"); }}
                              style={{
                                padding: "8px 16px", background: "#2563eb", color: "white",
                                border: "none", borderRadius: "8px", cursor: "pointer",
                                fontSize: "13px", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px",
                                transition: "background 0.2s"
                              }}
                              onMouseEnter={(e) => e.currentTarget.style.background = "#4f46e5"}
                              onMouseLeave={(e) => e.currentTarget.style.background = "#2563eb"}
                            >
                              <MessageCircle size={14} /> Message Patient
                            </button>
                          </div>
                        </div>

                        {/* Scan History */}
                        <h4 style={{ margin: "0 0 12px 0", fontSize: "15px", fontWeight: "700", color: textPrimary, display: "flex", alignItems: "center", gap: "8px" }}>
                          <FileText size={16} color="#2563eb" />
                          Scan History ({expandedPatientScans.length})
                        </h4>

                        {expandedPatientScans.length === 0 ? (
                          <div style={{
                            padding: "20px", textAlign: "center",
                            background: darkMode ? "#1e293b" : "white",
                            borderRadius: "10px",
                            border: `1px dashed ${darkMode ? "#475569" : "#d1d5db"}`,
                            color: textSecondary, fontSize: "14px"
                          }}>
                            No scans recorded for this patient yet.
                          </div>
                        ) : (
                          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                            {expandedPatientScans.map((scan) => (
                              <div key={scan.id || scan.scan_id} style={{
                                padding: "14px 16px",
                                background: darkMode ? "#1e293b" : "white",
                                borderRadius: "10px",
                                border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                                display: "flex", alignItems: "center", justifyContent: "space-between",
                                transition: "box-shadow 0.2s"
                              }}
                              onMouseEnter={(e) => e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.08)"}
                              onMouseLeave={(e) => e.currentTarget.style.boxShadow = "none"}
                              >
                                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                                  <div style={{
                                    width: "10px", height: "10px", borderRadius: "50%",
                                    background: scan.is_tumor ? "#ef4444" : "#10b981",
                                    boxShadow: scan.is_tumor ? "0 0 8px rgba(239,68,68,0.4)" : "0 0 8px rgba(16,185,129,0.4)"
                                  }} />
                                  <div>
                                    <p style={{ margin: 0, fontSize: "14px", fontWeight: "600", color: textPrimary }}>
                                      {scan.prediction || (scan.is_tumor ? "Tumor Detected" : "No Tumor")}
                                    </p>
                                    <p style={{ margin: "2px 0 0 0", fontSize: "12px", color: textSecondary }}>
                                      {scan.created_at ? new Date(scan.created_at).toLocaleString() : "Unknown date"}
                                      {scan.confidence ? ` • ${(scan.confidence * 100).toFixed(1)}% confidence` : ""}
                                    </p>
                                  </div>
                                </div>
                                <button
                                  onClick={(e) => { e.stopPropagation(); downloadPDF(scan.id || scan.scan_id); }}
                                  style={{
                                    padding: "6px 14px", background: "linear-gradient(135deg, #2563eb, #4f46e5)",
                                    color: "white", border: "none", borderRadius: "8px",
                                    cursor: "pointer", fontSize: "12px", fontWeight: "600",
                                    display: "flex", alignItems: "center", gap: "6px",
                                    transition: "transform 0.15s, box-shadow 0.15s"
                                  }}
                                  onMouseEnter={(e) => { e.currentTarget.style.transform = "translateY(-1px)"; e.currentTarget.style.boxShadow = "0 4px 12px rgba(37,99,235,0.3)"; }}
                                  onMouseLeave={(e) => { e.currentTarget.style.transform = "translateY(0)"; e.currentTarget.style.boxShadow = "none"; }}
                                >
                                  <Download size={14} /> Download Report
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Tumor Progression Tracker */}
                        {expandedPatientScans.length > 1 && (
                          <div style={{ marginTop: "24px" }}>
                            <TumourProgressionTracker
                              scans={expandedPatientScans}
                              darkMode={darkMode}
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}

                {patients.filter(p => {
                  if (!patientSearch.trim()) return true;
                  const q = patientSearch.toLowerCase();
                  return (p.full_name || "").toLowerCase().includes(q) || (p.email || "").toLowerCase().includes(q);
                }).length === 0 && (
                  <div style={{ padding: "40px", textAlign: "center", color: textSecondary }}>
                    {patients.length === 0 ? "No patients found. Add patients from the New Scan tab." : "No patients match your search."}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Subscription / Plan Management View */}
          {view === "subscription" && (
            <div>
              <h2 style={{ fontSize: "32px", fontWeight: "bold", marginBottom: "8px", color: textPrimary }}>
                Subscription & Billing
              </h2>
              <p style={{ margin: "0 0 32px 0", color: "#64748b", fontSize: "15px" }}>
                Manage your plan, view usage, and upgrade to unlock more features
              </p>

              {/* Current Plan Card */}
              {usage && (
                <div style={{
                  background: "linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%)",
                  borderRadius: "20px",
                  padding: "32px",
                  color: "white",
                  marginBottom: "32px",
                  position: "relative",
                  overflow: "hidden",
                  boxShadow: "0 10px 40px rgba(15, 23, 42, 0.3)"
                }}>
                  {/* Decorative circles */}
                  <div style={{ position: "absolute", top: "-40px", right: "-40px", width: "160px", height: "160px", borderRadius: "50%", background: "rgba(59,130,246,0.15)" }} />
                  <div style={{ position: "absolute", bottom: "-30px", right: "60px", width: "100px", height: "100px", borderRadius: "50%", background: "rgba(139,92,246,0.12)" }} />

                  <div style={{ position: "relative", zIndex: 1 }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px", marginBottom: "24px" }}>
                      <div>
                        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "4px" }}>
                          <h3 style={{ margin: 0, fontSize: "24px", fontWeight: "700" }}>
                            {getPlanDisplayName(usage.plan_type)}
                          </h3>
                          <span style={{
                            padding: "4px 12px",
                            borderRadius: "20px",
                            fontSize: "11px",
                            fontWeight: "700",
                            textTransform: "uppercase",
                            letterSpacing: "0.5px",
                            background: usage.plan_type === "free" ? "rgba(148,163,184,0.25)" : usage.plan_type === "basic" ? "rgba(59,130,246,0.3)" : usage.plan_type === "premium" ? "rgba(168,85,247,0.3)" : "rgba(245,158,11,0.3)",
                            color: "white"
                          }}>
                            {usage.plan_type === "free" ? "Free Tier" : usage.plan_type === "basic" ? "Basic" : usage.plan_type === "premium" ? "Premium" : "Enterprise"}
                          </span>
                        </div>
                        <p style={{ margin: 0, fontSize: "14px", opacity: 0.7 }}>Your current active subscription</p>
                      </div>
                    </div>

                    {/* Usage Stats */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "20px" }}>
                      {/* Scans Used */}
                      <div style={{
                        background: "rgba(255,255,255,0.08)",
                        borderRadius: "14px",
                        padding: "20px",
                        border: "1px solid rgba(255,255,255,0.1)"
                      }}>
                        <div style={{ fontSize: "12px", opacity: 0.6, marginBottom: "8px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>Scans Used</div>
                        <div style={{ fontSize: "28px", fontWeight: "800" }}>{usage.scans_used || 0}</div>
                        <div style={{ fontSize: "13px", opacity: 0.5, marginTop: "4px" }}>
                          of {usage.scan_limit === -1 ? "Unlimited" : usage.scan_limit}
                        </div>
                      </div>
                      {/* Scans Remaining */}
                      <div style={{
                        background: "rgba(255,255,255,0.08)",
                        borderRadius: "14px",
                        padding: "20px",
                        border: "1px solid rgba(255,255,255,0.1)"
                      }}>
                        <div style={{ fontSize: "12px", opacity: 0.6, marginBottom: "8px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>Remaining</div>
                        <div style={{ fontSize: "28px", fontWeight: "800", color: usage.scan_limit !== -1 && (usage.scan_limit - (usage.scans_used || 0)) <= 3 ? "#f87171" : "#4ade80" }}>
                          {usage.scan_limit === -1 ? "∞" : Math.max(0, usage.scan_limit - (usage.scans_used || 0))}
                        </div>
                        <div style={{ fontSize: "13px", opacity: 0.5, marginTop: "4px" }}>scans left</div>
                      </div>
                      {/* Usage Percentage */}
                      <div style={{
                        background: "rgba(255,255,255,0.08)",
                        borderRadius: "14px",
                        padding: "20px",
                        border: "1px solid rgba(255,255,255,0.1)"
                      }}>
                        <div style={{ fontSize: "12px", opacity: 0.6, marginBottom: "8px", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.5px" }}>Usage</div>
                        <div style={{ fontSize: "28px", fontWeight: "800" }}>
                          {usage.scan_limit === -1 ? "—" : `${Math.round(((usage.scans_used || 0) / usage.scan_limit) * 100)}%`}
                        </div>
                        <div style={{ fontSize: "13px", opacity: 0.5, marginTop: "4px" }}>of plan limit</div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    {usage.scan_limit !== -1 && (
                      <div style={{ marginTop: "20px" }}>
                        <div style={{
                          width: "100%",
                          height: "8px",
                          background: "rgba(255,255,255,0.12)",
                          borderRadius: "4px",
                          overflow: "hidden"
                        }}>
                          <div style={{
                            width: `${Math.min(((usage.scans_used || 0) / usage.scan_limit) * 100, 100)}%`,
                            height: "100%",
                            borderRadius: "4px",
                            background: ((usage.scans_used || 0) / usage.scan_limit) > 0.8
                              ? "linear-gradient(90deg, #f87171, #ef4444)"
                              : "linear-gradient(90deg, #3b82f6, #8b5cf6)",
                            transition: "width 0.5s ease"
                          }} />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Billing Cycle Toggle */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
                <h3 style={{ margin: 0, fontSize: "20px", fontWeight: "700", color: textPrimary }}>
                  {usage && usage.plan_type !== "enterprise" ? "Upgrade Your Plan" : "Available Plans"}
                </h3>
                <div style={{
                  display: "inline-flex",
                  background: darkMode ? "#1e293b" : "#f1f5f9",
                  borderRadius: "10px",
                  padding: "4px",
                  border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`
                }}>
                  <button
                    onClick={() => setBillingCycle("monthly")}
                    style={{
                      padding: "8px 20px",
                      borderRadius: "8px",
                      border: "none",
                      background: billingCycle === "monthly" ? (darkMode ? "#3b82f6" : "#2563eb") : "transparent",
                      color: billingCycle === "monthly" ? "white" : textSecondary,
                      fontWeight: "600",
                      fontSize: "13px",
                      cursor: "pointer",
                      transition: "all 0.2s"
                    }}
                  >Monthly</button>
                  <button
                    onClick={() => setBillingCycle("yearly")}
                    style={{
                      padding: "8px 20px",
                      borderRadius: "8px",
                      border: "none",
                      background: billingCycle === "yearly" ? (darkMode ? "#3b82f6" : "#2563eb") : "transparent",
                      color: billingCycle === "yearly" ? "white" : textSecondary,
                      fontWeight: "600",
                      fontSize: "13px",
                      cursor: "pointer",
                      transition: "all 0.2s",
                      position: "relative"
                    }}
                  >
                    Yearly
                    <span style={{
                      position: "absolute",
                      top: "-8px",
                      right: "-12px",
                      background: "#10b981",
                      color: "white",
                      fontSize: "9px",
                      padding: "2px 6px",
                      borderRadius: "4px",
                      fontWeight: "bold"
                    }}>-17%</span>
                  </button>
                </div>
              </div>

              {/* Plan Cards Grid */}
              {loadingPlans ? (
                <div style={{ textAlign: "center", padding: "60px", color: textSecondary }}>
                  <Loader size={32} style={{ animation: "spin 1s linear infinite", marginBottom: "12px" }} />
                  <p>Loading plans...</p>
                </div>
              ) : (
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                  gap: "20px"
                }}>
                  {subscriptionPlans.map((plan) => {
                    const price = billingCycle === "monthly" ? plan.price_monthly : (plan.price_yearly || plan.price_monthly * 10) / 12;
                    const yearlyTotal = plan.price_yearly || plan.price_monthly * 10;
                    const isCurrentPlan = usage && (
                      (plan.name === "free" && usage.plan_type === "free") ||
                      (plan.name === "basic" && usage.plan_type === "basic") ||
                      (plan.name === "premium" && usage.plan_type === "premium") ||
                      (plan.name === "enterprise" && usage.plan_type === "enterprise")
                    );
                    const isFree = plan.price_monthly === 0;
                    const isProcessing = processingPlan === plan.id;
                    const isDowngrade = usage && (
                      (usage.plan_type === "enterprise") ||
                      (usage.plan_type === "premium" && (plan.name === "free" || plan.name === "basic")) ||
                      (usage.plan_type === "basic" && plan.name === "free")
                    );

                    let features = [];
                    try { features = JSON.parse(plan.features || "[]"); } catch (e) { features = []; }

                    const planColors = {
                      free: { gradient: "linear-gradient(135deg, #64748b, #475569)", badge: "#94a3b8" },
                      basic: { gradient: "linear-gradient(135deg, #3b82f6, #2563eb)", badge: "#3b82f6" },
                      premium: { gradient: "linear-gradient(135deg, #8b5cf6, #7c3aed)", badge: "#8b5cf6" },
                      enterprise: { gradient: "linear-gradient(135deg, #f59e0b, #d97706)", badge: "#f59e0b" }
                    };
                    const colors = planColors[plan.name] || planColors.free;

                    return (
                      <div key={plan.id} style={{
                        background: darkMode ? "#1e293b" : "white",
                        borderRadius: "18px",
                        padding: "28px",
                        border: isCurrentPlan
                          ? `2px solid ${colors.badge}`
                          : `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                        boxShadow: isCurrentPlan ? `0 8px 30px ${colors.badge}30` : "0 2px 8px rgba(0,0,0,0.04)",
                        position: "relative",
                        transition: "all 0.3s",
                        transform: isCurrentPlan ? "scale(1.02)" : "scale(1)"
                      }}>
                        {/* Current Plan Badge */}
                        {isCurrentPlan && (
                          <div style={{
                            position: "absolute",
                            top: "-12px",
                            left: "50%",
                            transform: "translateX(-50%)",
                            background: colors.badge,
                            color: "white",
                            padding: "4px 16px",
                            borderRadius: "20px",
                            fontSize: "11px",
                            fontWeight: "bold",
                            letterSpacing: "0.5px"
                          }}>CURRENT PLAN</div>
                        )}

                        {/* Plan Name */}
                        <h4 style={{ margin: "0 0 4px 0", fontSize: "20px", fontWeight: "700", color: textPrimary }}>
                          {plan.display_name}
                        </h4>
                        <p style={{ margin: "0 0 20px 0", fontSize: "13px", color: textSecondary }}>
                          {plan.description || ""}
                        </p>

                        {/* Price */}
                        <div style={{ marginBottom: "20px" }}>
                          <span style={{ fontSize: "36px", fontWeight: "800", color: textPrimary }}>${price.toFixed(0)}</span>
                          <span style={{ fontSize: "14px", color: textSecondary, marginLeft: "4px" }}>/month</span>
                          {billingCycle === "yearly" && !isFree && (
                            <p style={{ margin: "4px 0 0 0", fontSize: "12px", color: textSecondary }}>
                              ${yearlyTotal} billed annually
                            </p>
                          )}
                        </div>

                        {/* Feature highlights */}
                        <div style={{ marginBottom: "20px" }}>
                          <div style={{
                            display: "flex", alignItems: "center", gap: "8px",
                            padding: "8px 12px", background: darkMode ? "#0f172a" : "#f8fafc",
                            borderRadius: "8px", marginBottom: "8px"
                          }}>
                            <Zap size={14} color="#f59e0b" />
                            <span style={{ fontSize: "13px", fontWeight: "600", color: textPrimary }}>
                              {plan.max_scans_per_month === -1 ? "Unlimited scans" : `${plan.max_scans_per_month} scans/month`}
                            </span>
                          </div>
                          <div style={{
                            display: "flex", alignItems: "center", gap: "8px",
                            padding: "8px 12px", background: darkMode ? "#0f172a" : "#f8fafc",
                            borderRadius: "8px", marginBottom: "8px"
                          }}>
                            <Users size={14} color="#3b82f6" />
                            <span style={{ fontSize: "13px", color: textPrimary }}>
                              {plan.max_patients === -1 ? "Unlimited patients" : `Up to ${plan.max_patients} patients`}
                            </span>
                          </div>
                          {features.slice(0, 4).map((feat, idx) => (
                            <div key={idx} style={{ display: "flex", alignItems: "center", gap: "8px", padding: "4px 0" }}>
                              <Check size={14} color="#10b981" />
                              <span style={{ fontSize: "13px", color: textSecondary }}>
                                {feat.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                              </span>
                            </div>
                          ))}
                        </div>

                        {/* CTA Button */}
                        <button
                          onClick={() => handleUpgrade(plan)}
                          disabled={isCurrentPlan || isProcessing || isDowngrade || isFree}
                          style={{
                            width: "100%",
                            padding: "13px",
                            borderRadius: "12px",
                            border: "none",
                            background: isCurrentPlan ? (darkMode ? "#334155" : "#e2e8f0")
                              : (isDowngrade || isFree) ? (darkMode ? "#1e293b" : "#f1f5f9")
                              : colors.gradient,
                            color: (isCurrentPlan || isDowngrade || isFree) ? textSecondary : "white",
                            fontWeight: "700",
                            fontSize: "14px",
                            cursor: (isCurrentPlan || isProcessing || isDowngrade || isFree) ? "not-allowed" : "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: "8px",
                            opacity: isProcessing ? 0.7 : 1,
                            transition: "all 0.2s",
                            boxShadow: !(isCurrentPlan || isDowngrade || isFree) ? `0 4px 12px ${colors.badge}40` : "none"
                          }}
                        >
                          {isProcessing ? (
                            <><Loader size={16} style={{ animation: "spin 1s linear infinite" }} /> Processing...</>
                          ) : isCurrentPlan ? (
                            <><CheckCircle size={16} /> Current Plan</>
                          ) : isDowngrade || isFree ? (
                            "—"
                          ) : (
                            <><CreditCard size={16} /> Subscribe Now</>
                          )}
                        </button>

                        {/* Secure badge */}
                        {!isFree && !isCurrentPlan && !isDowngrade && (
                          <div style={{ marginTop: "10px", textAlign: "center", display: "flex", alignItems: "center", justifyContent: "center", gap: "4px", fontSize: "11px", color: textSecondary }}>
                            <Shield size={12} />
                            <span>Secure payment by Stripe</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Chat View */}
          {view === "chat" && (
            <>
              {/* Upgrade overlay for Free tier */}
              {usage && usage.plan_type === "free" ? (
                <div style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  minHeight: "60vh",
                  padding: "40px",
                  textAlign: "center"
                }}>
                  <div style={{
                    background: darkMode ? "#1e293b" : "white",
                    padding: "60px 80px",
                    borderRadius: "32px",
                    boxShadow: darkMode ? "0 25px 50px rgba(0,0,0,0.4)" : "0 25px 50px rgba(0,0,0,0.1)",
                    border: darkMode ? "1px solid #334155" : "1px solid #e2e8f0",
                    maxWidth: "500px"
                  }}>
                    <div style={{
                      width: "80px",
                      height: "80px",
                      background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                      borderRadius: "20px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 24px",
                      boxShadow: "0 10px 25px rgba(245, 158, 11, 0.3)"
                    }}>
                      <MessageCircle size={40} color="white" />
                    </div>
                    <h3 style={{ 
                      fontSize: "28px", 
                      fontWeight: "800", 
                      color: textPrimary, 
                      marginBottom: "12px" 
                    }}>
                      Chat Feature Locked
                    </h3>
                    <p style={{ 
                      fontSize: "16px", 
                      color: textSecondary, 
                      marginBottom: "32px",
                      lineHeight: "1.6"
                    }}>
                      Upgrade to Basic or Premium plan to unlock the full chat system and communicate directly with your patients.
                    </p>
                    <button
                      onClick={() => { setView("subscription"); loadSubscriptionPlans(); }}
                      style={{
                        padding: "16px 40px",
                        background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                        color: "white",
                        border: "none",
                        borderRadius: "50px",
                        fontWeight: "700",
                        fontSize: "16px",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                        margin: "0 auto",
                        boxShadow: "0 10px 25px rgba(245, 158, 11, 0.3)",
                        transition: "all 0.2s ease"
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = "translateY(-3px)";
                        e.currentTarget.style.boxShadow = "0 15px 35px rgba(245, 158, 11, 0.4)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = "translateY(0)";
                        e.currentTarget.style.boxShadow = "0 10px 25px rgba(245, 158, 11, 0.3)";
                      }}
                    >
                      <Zap size={20} />
                      Upgrade to Unlock
                    </button>
                  </div>
                </div>
              ) : (
                <EnhancedChat
                  user={user}
                  selectedPatient={selectedPatient}
                  patients={patients}
                  onSelectPatient={setSelectedPatient}
                  darkMode={darkMode}
                  socket={socket}
                />
              )}
            </>
          )}
          {/* Scan Limit Upgrade Modal */}
          {showUpgradeModal && (
            <div style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.6)",
              backdropFilter: "blur(8px)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 9999,
              padding: "20px"
            }}>
              <div style={{
                background: darkMode ? "#1e293b" : "white",
                borderRadius: "24px",
                maxWidth: "480px",
                width: "100%",
                padding: "40px",
                textAlign: "center",
                boxShadow: "0 25px 60px rgba(0,0,0,0.3)",
                border: darkMode ? "1px solid #334155" : "1px solid #e2e8f0",
                position: "relative",
                animation: "fadeIn 0.3s ease"
              }}>
                {/* Close button */}
                <button
                  onClick={() => setShowUpgradeModal(false)}
                  style={{
                    position: "absolute",
                    top: "16px",
                    right: "16px",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    color: textSecondary,
                    padding: "4px"
                  }}
                >
                  <X size={20} />
                </button>

                {/* Icon */}
                <div style={{
                  width: "72px",
                  height: "72px",
                  background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                  borderRadius: "18px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 20px",
                  boxShadow: "0 10px 25px rgba(245, 158, 11, 0.35)"
                }}>
                  <AlertTriangle size={36} color="white" />
                </div>

                {/* Title */}
                <h3 style={{
                  fontSize: "22px",
                  fontWeight: "800",
                  color: textPrimary,
                  marginBottom: "10px"
                }}>
                  Scan Limit Reached
                </h3>

                {/* Message */}
                <p style={{
                  fontSize: "15px",
                  color: textSecondary,
                  marginBottom: "8px",
                  lineHeight: "1.6"
                }}>
                  {upgradeMessage}
                </p>

                {/* Usage info */}
                {usage && (
                  <div style={{
                    background: darkMode ? "#0f172a" : "#f8fafc",
                    borderRadius: "12px",
                    padding: "16px",
                    marginBottom: "24px",
                    border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                      <span style={{ fontSize: "13px", color: textSecondary }}>Used</span>
                      <span style={{ fontSize: "13px", fontWeight: "700", color: "#ef4444" }}>
                        {usage.scans_used}/{usage.scan_limit} scans
                      </span>
                    </div>
                    <div style={{
                      width: "100%",
                      height: "6px",
                      background: darkMode ? "#1e293b" : "#e2e8f0",
                      borderRadius: "3px",
                      overflow: "hidden"
                    }}>
                      <div style={{
                        width: "100%",
                        height: "100%",
                        borderRadius: "3px",
                        background: "linear-gradient(90deg, #ef4444, #dc2626)"
                      }} />
                    </div>
                    {usage.days_until_reset > 0 && (
                      <p style={{ fontSize: "12px", color: textSecondary, marginTop: "8px", marginBottom: 0 }}>
                        Resets in {usage.days_until_reset} days
                      </p>
                    )}
                  </div>
                )}

                {/* CTA Buttons */}
                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  <button
                    onClick={() => {
                      setShowUpgradeModal(false);
                      setView("subscription");
                      loadSubscriptionPlans();
                    }}
                    style={{
                      width: "100%",
                      padding: "14px",
                      background: "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "12px",
                      fontWeight: "700",
                      fontSize: "15px",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "8px",
                      boxShadow: "0 6px 20px rgba(245, 158, 11, 0.35)",
                      transition: "all 0.2s ease"
                    }}
                  >
                    <Zap size={18} />
                    Upgrade Plan
                  </button>
                  <button
                    onClick={() => setShowUpgradeModal(false)}
                    style={{
                      width: "100%",
                      padding: "12px",
                      background: "transparent",
                      color: textSecondary,
                      border: `1px solid ${darkMode ? "#334155" : "#e2e8f0"}`,
                      borderRadius: "12px",
                      fontWeight: "600",
                      fontSize: "14px",
                      cursor: "pointer"
                    }}
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Notifications Panel */}
      {showNotifications && (
        <NotificationCentre
          notifications={notifications}
          onClose={() => setShowNotifications(false)}
          onMarkRead={(id) => {
            setNotifications((prev) =>
              prev.map((n) => (n.id === id ? { ...n, read: true } : n)),
            );
            setUnreadCount((prev) => Math.max(0, prev - 1));
          }}
          darkMode={darkMode}
        />
      )}

      {/* Video Call Modal (removed) */}
      {/* {showVideoCall && selectedPatient && (
        <VideoCall
          currentUserId={user?.id}
          currentUserType="hospital"
          remoteUserId={selectedPatient.id}
          remoteUserType="patient"
          onClose={() => setShowVideoCall(false)}
          darkMode={darkMode}
          audioOnly={callType === "audio"}
        />
      )} */}


      <style>{`
        .dark { background-color: #0f172a; color: #e5e7eb; }
        .dark input, .dark textarea, .dark select {
          background-color: #1e293b;
          color: #e5e7eb;
          border: 1px solid #334155;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes slideIn {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
          from { transform: translateX(0); opacity: 1; }
          to { transform: translateX(100%); opacity: 0; }
        }
        .animate-spin { animation: spin 1s linear infinite; }
      `}</style>
    </div>
  );
}

function NavItem({ icon, label, active, onClick, badge }) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: "100%",
        padding: "14px 16px",
        marginBottom: "8px",
        display: "flex",
        alignItems: "center",
        gap: "14px",
        background: active
          ? "linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)"
          : hovered
            ? "rgba(255, 255, 255, 0.08)"
            : "transparent",
        color: active ? "white" : hovered ? "#e2e8f0" : "rgba(148, 163, 184, 0.9)",
        border: active ? "1px solid rgba(59, 130, 246, 0.5)" : "1px solid transparent",
        borderLeft: active ? "3px solid rgba(96, 165, 250, 0.9)" : "3px solid transparent",
        borderRadius: "12px",
        cursor: "pointer",
        transition: "all 0.25s ease",
        position: "relative",
        boxShadow: active ? "0 0 20px rgba(59, 130, 246, 0.4), 0 4px 12px rgba(59, 130, 246, 0.2)" : "none",
        fontWeight: active ? "600" : "500",
        fontSize: "14px",
        backdropFilter: hovered && !active ? "blur(8px)" : "none",
      }}
    >
      {icon}
      <span style={{ flex: 1, textAlign: "left" }}>{label}</span>
      {badge && (
        <span
          style={{
            background: "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
            color: "white",
            borderRadius: "50%",
            width: "22px",
            height: "22px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "11px",
            fontWeight: "bold",
            boxShadow: "0 0 10px rgba(239, 68, 68, 0.5)",
          }}
        >
          {badge}
        </span>
      )}
    </button>
  );
}

function SimplePatientInfoModal({ onClose, onSubmit, darkMode }) {
  const [formData, setFormData] = useState({
    patient_name: "",
    email: "",
    phone: "",
  });

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          background: darkMode ? "#1e293b" : "white",
          borderRadius: "12px",
          padding: "24px",
          maxWidth: "500px",
          width: "90%",
        }}
      >
        <h3 style={{ margin: "0 0 20px 0" }}>Patient Information</h3>

        <div style={{ marginBottom: "16px" }}>
          <label style={{ display: "block", marginBottom: "8px" }}>
            Patient Name
          </label>
          <input
            type="text"
            value={formData.patient_name}
            onChange={(e) =>
              setFormData({ ...formData, patient_name: e.target.value })
            }
            style={{
              width: "100%",
              padding: "10px",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
            }}
          />
        </div>

        <div style={{ display: "flex", gap: "12px", marginTop: "24px" }}>
          <button
            onClick={() => onSubmit(null)}
            style={{
              flex: 1,
              padding: "12px",
              background: "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
            }}
          >
            Skip
          </button>
          <button
            onClick={() => onSubmit(formData)}
            style={{
              flex: 1,
              padding: "12px",
              background: "#10b981",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
            }}
          >
            Save
          </button>
          <button
            onClick={onClose}
            style={{
              padding: "12px 20px",
              background: "#ef4444",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function FixedAnalysisResults({ prediction, darkMode, onDownloadPDF }) {
  if (!prediction) return null;

  const colors = {
    glioma: "#ef4444",
    meningioma: "#f59e0b",
    pituitary: "#8b5cf6",
    notumor: "#10b981",
  };

  const labels = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    pituitary: "Pituitary Tumor",
    notumor: "No Tumor Detected",
  };

  const predictionType = (prediction.prediction || "notumor").toLowerCase();
  const confidence = parseFloat(prediction.confidence) || 0;
  const probabilities = prediction.probabilities || {};

  return (
    <div
      style={{
        padding: "24px",
        background: darkMode ? "#0f172a" : "#f8fafc",
        borderRadius: "12px",
        border: `2px solid ${colors[predictionType]}`,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "16px",
          marginBottom: "20px",
        }}
      >
        {prediction.is_tumor ? (
          <AlertCircle size={32} color={colors[predictionType]} />
        ) : (
          <CheckCircle size={32} color={colors[predictionType]} />
        )}
        <div style={{ flex: 1 }}>
          <h3
            style={{
              margin: "0 0 4px 0",
              fontSize: "24px",
              color: colors[predictionType],
            }}
          >
            {labels[predictionType]}
          </h3>
          <p style={{ margin: 0, fontSize: "18px", fontWeight: "600" }}>
            Confidence: {confidence.toFixed(2)}%
          </p>
        </div>

        <button
          onClick={onDownloadPDF}
          style={{
            padding: "12px 20px",
            background: colors[predictionType],
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "8px",
            fontWeight: "600",
          }}
        >
          <Download size={18} />
          Download PDF
        </button>
      </div>

      <div
        style={{
          padding: "16px",
          background: darkMode ? "#1e293b" : "white",
          borderRadius: "8px",
          marginTop: "16px",
        }}
      >
        <h4 style={{ margin: "0 0 12px 0" }}>All Probabilities:</h4>
        {Object.entries(probabilities).map(([key, value]) => (
          <div key={key} style={{ marginBottom: "8px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "4px",
              }}
            >
              <span style={{ fontSize: "14px" }}>{labels[key] || key}</span>
              <span
                style={{
                  fontSize: "14px",
                  fontWeight: "600",
                  color: colors[key],
                }}
              >
                {parseFloat(value).toFixed(2)}%
              </span>
            </div>
            <div
              style={{
                width: "100%",
                height: "6px",
                background: darkMode ? "#0f172a" : "#f1f5f9",
                borderRadius: "3px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${value}%`,
                  height: "100%",
                  background: colors[key],
                  borderRadius: "3px",
                  transition: "width 0.5s ease",
                }}
              />
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: "16px",
          padding: "12px",
          background: darkMode ? "#451a03" : "#fef3c7",
          borderRadius: "8px",
          border: `1px solid ${darkMode ? "#78350f" : "#fbbf24"}`,
          fontSize: "13px",
          color: darkMode ? "#fde68a" : "#78350f",
        }}
      >
        <strong>Note:</strong> This analysis is AI-generated and should be
        verified by a qualified medical professional.
      </div>
    </div>
  );
}
function DashboardCard({ title, value, icon, darkMode, subtitle, bgGradient, iconBg }) {
  // Default pastel gradient if not provided
  const gradient = bgGradient || (darkMode 
    ? 'linear-gradient(135deg, #1e293b 0%, #334155 100%)' 
    : 'linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%)');
  
  return (
    <div style={{ 
      background: gradient,
      padding: "14px 20px",
      borderRadius: "20px",
      position: "relative",
      overflow: "hidden",
      minHeight: "100px",
      border: "1px solid rgba(255, 255, 255, 0.2)",
      boxShadow: "0 4px 15px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.3)",
      backdropFilter: "blur(10px)",
      transition: "all 0.3s ease"
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = 'translateY(-4px)';
      e.currentTarget.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.12)';
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.08)';
    }}
    >
      {/* Hexagonal Pattern Overlay */}
      <div style={{
        position: "absolute",
        top: 0,
        right: 0,
        width: "60%",
        height: "100%",
        opacity: 0.4,
        backgroundImage: `
          radial-gradient(circle, rgba(255,255,255,0.3) 1px, transparent 1px),
          radial-gradient(circle, rgba(255,255,255,0.2) 1px, transparent 1px)
        `,
        backgroundSize: "20px 35px, 20px 35px",
        backgroundPosition: "0 0, 10px 17px",
        pointerEvents: "none"
      }} />
      
      {/* Decorative Hexagons */}
      <div style={{
        position: "absolute",
        top: "20%",
        right: "10%",
        width: "80px",
        height: "80px",
        background: "rgba(255, 255, 255, 0.15)",
        borderRadius: "12px",
        transform: "rotate(15deg)",
        pointerEvents: "none"
      }} />
      <div style={{
        position: "absolute",
        bottom: "10%",
        right: "25%",
        width: "40px",
        height: "40px",
        background: "rgba(255, 255, 255, 0.1)",
        borderRadius: "8px",
        transform: "rotate(-10deg)",
        pointerEvents: "none"
      }} />
      
      {/* Icon Container */}
      <div style={{ 
        width: "36px", 
        height: "36px", 
        borderRadius: "10px", 
        background: "white",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
        display: "flex", 
        alignItems: "center", 
        justifyContent: "center",
        marginBottom: "8px",
        position: "relative",
        zIndex: 1
      }}>
        {icon}
      </div>
      
      {/* Content */}
      <div style={{ position: "relative", zIndex: 1 }}>
        <h3 style={{ 
          margin: "0 0 2px 0", 
          fontSize: "14px", 
          fontWeight: "700", 
          color: "#1e293b",
          letterSpacing: "-0.01em",
          opacity: 0.7
        }}>
          {title}
        </h3>
        <p style={{ 
          margin: 0, 
          fontSize: "22px", 
          fontWeight: "800", 
          color: "#1e293b" 
        }}>
          {value}
        </p>
        {subtitle && (
          <p style={{ 
            margin: "4px 0 0 0", 
            fontSize: "12px", 
            color: "#64748b" 
          }}>
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
}

function ActionButton({ icon, label, onClick, color, darkMode }) {
  return (
    <button 
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: "12px",
        padding: "16px",
        background: darkMode ? "#0f172a" : "white",
        border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
        borderRadius: "12px",
        cursor: "pointer",
        textAlign: "left",
        width: "100%",
        transition: "all 0.2s",
        boxShadow: "0 2px 4px rgba(0,0,0,0.02)"
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = color;
        e.currentTarget.style.transform = "translateY(-2px)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = darkMode ? "#334155" : "#e5e7eb";
        e.currentTarget.style.transform = "translateY(0)";
      }}
    >
      <div style={{ color: color }}>{icon}</div>
      <span style={{ fontWeight: "600", color: darkMode ? "#f1f5f9" : "#1e293b", fontSize: "14px" }}>{label}</span>
    </button>
  );
}
