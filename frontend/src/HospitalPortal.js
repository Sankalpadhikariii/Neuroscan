import React, { useState, useEffect, useRef } from "react";
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
} from "lucide-react";
import { io } from "socket.io-client";
import EnhancedChat from "./EnhancedChat";
import NotificationCentre from "./NotificationCentre";
import GradCamvisualization from "./GradCamvisualization";
import TumourProgressionTracker from "./TumourProgressionTracker";
// import VideoCall from "./Videocall";
import AddPatientModal from "./AddPatientModal";

const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";
const socket = io(API_BASE, { withCredentials: true });

export default function HospitalPortalEnhanced({ user, onLogout }) {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("hospitalTheme") === "dark",
  );

  const [view, setView] = useState("scan");
  const [patients, setPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [usage, setUsage] = useState(null);
  const [showPatientInfoModal, setShowPatientInfoModal] = useState(false);
  const [showAddPatientModal, setShowAddPatientModal] = useState(false);

  // New states for enhanced features
  const [showChat, setShowChat] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showNotifications, setShowNotifications] = useState(false);
  const [gradcamData, setGradcamData] = useState(null);
  const [validationWarning, setValidationWarning] = useState(null);
  const [patientScans, setPatientScans] = useState([]);

  // Video call states (removed)
  // const [showVideoCall, setShowVideoCall] = useState(false);
  // const [callType, setCallType] = useState("video");

  const fileInputRef = useRef(null);

  useEffect(() => {
    loadPatients();
    loadUsageStatus();
    loadNotifications();
    setupSocketListeners();

    return () => {
      socket.off("new_notification");
      socket.off("new_message");
      socket.off("patient_update");
    };
  }, []);

  useEffect(() => {
    if (selectedPatient) {
      loadPatientScans(selectedPatient.id);
    }
  }, [selectedPatient]);

  function setupSocketListeners() {
    socket.on("new_notification", (notification) => {
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
    setShowAddPatientModal(false);
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

  function handleAnalyzeClick() {
    if (!selectedFile) {
      setError("Please select an image file first");
      return;
    }

    if (!selectedPatient) {
      setShowPatientInfoModal(true);
      return;
    }

    performAnalysis();
  }

  async function handlePatientInfoSubmit(formData) {
    setShowPatientInfoModal(false);

    try {
      setLoading(true);
      setError(null);

      const payload =
        formData === null
          ? {
              full_name: `Anonymous Scan ${new Date().toISOString()}`,
              email: "",
              phone: "",
            }
          : {
              full_name:
                formData.patient_name || `Patient ${new Date().toISOString()}`,
              email: formData.email || "",
              phone: formData.phone || "",
            };

      const res = await fetch(`${API_BASE}/hospital/patients`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Failed to create patient`);
      }

      const data = await res.json();
      const patient = data.patient;

      if (patient && patient.id) {
        setSelectedPatient(patient);
        await loadPatients();
        await performAnalysisWithPatient(patient.id);
      }
    } catch (err) {
      console.error("Patient creation failed:", err);
      setError("Unable to create patient for scan.");
      setLoading(false);
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
      }

      await loadUsageStatus();
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
      background: ${type === "success" ? "#10b981" : type === "error" ? "#ef4444" : "#6366f1"};
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

  return (
    <div className={darkMode ? "dark" : ""}>
      <div style={{ display: "flex", minHeight: "100vh", background: bgColor }}>
        {/* Sidebar */}
        <aside
          style={{
            width: 260,
            background: darkMode ? "#1e293b" : "white",
            padding: "20px",
            display: "flex",
            flexDirection: "column",
            borderRight: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
          }}
        >
          <div style={{ marginBottom: "32px" }}>
            <h1
              style={{
                fontSize: "24px",
                fontWeight: "bold",
                color: "#6366f1",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              <Brain size={28} /> NeuroScan
            </h1>
            <p
              style={{
                fontSize: "14px",
                color: textSecondary,
                marginTop: "4px",
              }}
            >
              Hospital Portal
            </p>
          </div>

          {/* Notification Bell */}
          <div
            style={{
              position: "relative",
              marginBottom: "20px",
              display: "flex",
              justifyContent: "flex-end",
            }}
          >
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              style={{
                padding: "10px",
                background: darkMode ? "#334155" : "#f1f5f9",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                position: "relative",
              }}
            >
              <Bell size={20} color={textPrimary} />
              {unreadCount > 0 && (
                <span
                  style={{
                    position: "absolute",
                    top: "-5px",
                    right: "-5px",
                    background: "#ef4444",
                    color: "white",
                    borderRadius: "50%",
                    width: "20px",
                    height: "20px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "11px",
                    fontWeight: "bold",
                  }}
                >
                  {unreadCount}
                </span>
              )}
            </button>
          </div>

          {/* Business Model Banner */}
          {usage && (
            <div
              style={{
                padding: "16px",
                background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
                borderRadius: "12px",
                marginBottom: "24px",
                color: "white",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  marginBottom: "8px",
                }}
              >
                <Zap size={18} />
                <h3 style={{ margin: 0, fontSize: "14px", fontWeight: "600" }}>
                  {usage.plan_type === "free"
                    ? "Free Plan"
                    : usage.plan_type === "basic"
                      ? "Basic Plan"
                      : usage.plan_type === "premium"
                        ? "Premium Plan"
                        : "Enterprise Plan"}
                </h3>
              </div>
              <p style={{ margin: "4px 0", fontSize: "12px", opacity: 0.9 }}>
                {usage.scans_used || 0} /{" "}
                {usage.scan_limit === -1 ? "∞" : usage.scan_limit} scans used
              </p>
              {usage.scan_limit !== -1 && (
                <div
                  style={{
                    width: "100%",
                    height: "4px",
                    background: "rgba(255,255,255,0.3)",
                    borderRadius: "2px",
                    marginTop: "8px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${Math.min((usage.scans_used / usage.scan_limit) * 100, 100)}%`,
                      height: "100%",
                      background: "white",
                      borderRadius: "2px",
                      transition: "width 0.3s ease",
                    }}
                  />
                </div>
              )}
            </div>
          )}

          <nav style={{ flex: 1 }}>
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
              badge={showChat && unreadCount > 0 ? unreadCount : null}
            />
            <NavItem
              icon={<BarChart3 size={20} />}
              label="Analytics"
              active={view === "analytics"}
              onClick={() => setView("analytics")}
            />
            <NavItem
              icon={<Settings size={20} />}
              label="Settings"
              active={view === "settings"}
              onClick={() => setView("settings")}
            />
          </nav>

          <button
            onClick={onLogout}
            style={{
              width: "100%",
              padding: "14px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "10px",
              background: "#ef4444",
              color: "white",
              border: "none",
              borderRadius: "12px",
              cursor: "pointer",
              fontWeight: "600",
              transition: "all 0.2s",
            }}
          >
            <LogOut size={20} />
            Logout
          </button>
        </aside>

        {/* Main Content */}
        <main style={{ flex: 1, padding: "32px", overflowY: "auto" }}>
          {/* Quick Action Bar for Selected Patient */}
          {selectedPatient && (
            <div
              style={{
                marginBottom: "24px",
                padding: "20px",
                background: darkMode ? "#1e293b" : "#ffffff",
                borderRadius: "16px",
                border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                boxShadow: darkMode
                  ? "0 4px 12px rgba(0,0,0,0.2)"
                  : "0 4px 12px rgba(0,0,0,0.05)",
              }}
            >
              <div>
                <h3
                  style={{
                    margin: "0 0 4px 0",
                    fontSize: "18px",
                    fontWeight: "600",
                    color: textPrimary,
                  }}
                >
                  Current Patient: {selectedPatient.full_name}
                </h3>
                <p
                  style={{ margin: 0, fontSize: "14px", color: textSecondary }}
                >
                  {selectedPatient.email} • ID: {selectedPatient.id}
                </p>
              </div>

              <div style={{ display: "flex", gap: "12px" }}>
{/* Video and Audio call buttons removed */}

                <button
                  onClick={() => {
                    setShowChat(true);
                    setView("chat");
                  }}
                  style={{
                    padding: "10px 20px",
                    background: "#6366f1",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    fontSize: "14px",
                    fontWeight: "500",
                    transition: "all 0.2s",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = "#4f46e5")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background = "#6366f1")
                  }
                >
                  <MessageCircle size={18} />
                  Message
                </button>
              </div>
            </div>
          )}

          {/* Scan View */}
          {view === "scan" && (
            <div>
              <h2
                style={{
                  fontSize: "32px",
                  fontWeight: "bold",
                  marginBottom: "24px",
                  color: textPrimary,
                }}
              >
                Brain Tumor Analysis
              </h2>

              {/* Patient Selection */}
              <div style={{ marginBottom: "24px" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "12px",
                    flexWrap: "wrap",
                  }}
                >
                  <div>
                    <label
                      style={{
                        display: "block",
                        marginBottom: "4px",
                        fontWeight: "600",
                        color: textPrimary,
                      }}
                    >
                      Select Patient
                    </label>
                    <p
                      style={{
                        margin: 0,
                        color: textSecondary,
                        fontSize: "13px",
                      }}
                    >
                      Choose an existing patient or add a new one before
                      uploading the scan
                    </p>
                  </div>
                  <button
                    onClick={() => setShowAddPatientModal(true)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      padding: "10px 14px",
                      background: "#10b981",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      cursor: "pointer",
                      fontWeight: "600",
                    }}
                  >
                    <Plus size={18} />
                    Add patient
                  </button>
                </div>

                {patients.length > 0 ? (
                  <select
                    value={selectedPatient?.id || ""}
                    onChange={(e) => {
                      const patient = patients.find(
                        (p) => p.id === parseInt(e.target.value),
                      );
                      setSelectedPatient(patient);
                    }}
                    style={{
                      width: "100%",
                      marginTop: "12px",
                      padding: "12px",
                      borderRadius: "8px",
                      border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                      background: darkMode ? "#1e293b" : "white",
                      color: textPrimary,
                    }}
                  >
                    <option value="">-- Select a patient --</option>
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.full_name} - {p.email}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div
                    style={{
                      marginTop: "12px",
                      padding: "14px",
                      borderRadius: "8px",
                      border: `1px dashed ${darkMode ? "#334155" : "#e5e7eb"}`,
                      background: darkMode ? "#1e293b" : "#ffffff",
                      color: textSecondary,
                      fontSize: "14px",
                    }}
                  >
                    No patients yet. Click "Add patient" to create one and
                    attach their MRI scan.
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
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: `2px dashed ${darkMode ? "#475569" : "#cbd5e1"}`,
                  borderRadius: "16px",
                  padding: "60px 40px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: darkMode ? "#1e293b" : "white",
                  transition: "all 0.3s",
                  marginBottom: "24px",
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.borderColor = "#6366f1";
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
                <p style={{ fontSize: "14px", color: textSecondary }}>
                  Click or drag & drop an MRI image (JPEG, PNG)
                </p>
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
                onClick={handleAnalyzeClick}
                disabled={!selectedFile || loading}
                style={{
                  width: "100%",
                  padding: "16px",
                  background: loading ? "#94a3b8" : "#6366f1",
                  color: "white",
                  border: "none",
                  borderRadius: "12px",
                  fontSize: "16px",
                  fontWeight: "600",
                  cursor: loading ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "12px",
                  transition: "all 0.2s",
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
                <div style={{ marginTop: "24px" }}>
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
              <h2
                style={{
                  fontSize: "32px",
                  fontWeight: "bold",
                  marginBottom: "24px",
                  color: textPrimary,
                }}
              >
                Patient Management
              </h2>

              <div style={{ display: "grid", gap: "16px" }}>
                {patients.map((patient) => (
                  <div
                    key={patient.id}
                    style={{
                      padding: "20px",
                      background: darkMode ? "#1e293b" : "white",
                      borderRadius: "12px",
                      border: `1px solid ${darkMode ? "#334155" : "#e5e7eb"}`,
                      cursor: "pointer",
                      transition: "all 0.2s",
                    }}
                    onClick={() => {
                      setSelectedPatient(patient);
                      setView("scan");
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "start",
                      }}
                    >
                      <div style={{ flex: 1 }}>
                        <h3 style={{ margin: "0 0 8px 0", color: textPrimary }}>
                          {patient.full_name}
                        </h3>
                        <p
                          style={{
                            margin: "0 0 4px 0",
                            fontSize: "14px",
                            color: textSecondary,
                          }}
                        >
                          {patient.email}
                        </p>
                        <p
                          style={{
                            margin: 0,
                            fontSize: "14px",
                            color: textSecondary,
                          }}
                        >
                          {patient.phone}
                        </p>
                      </div>

                      <div style={{ display: "flex", gap: "8px" }}>
{/* Video Call button removed */}

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedPatient(patient);
                            setShowChat(true);
                            setView("chat");
                          }}
                          style={{
                            padding: "8px",
                            background: "#6366f1",
                            color: "white",
                            border: "none",
                            borderRadius: "8px",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            transition: "all 0.2s",
                          }}
                          title="Message"
                          onMouseEnter={(e) =>
                            (e.currentTarget.style.background = "#4f46e5")
                          }
                          onMouseLeave={(e) =>
                            (e.currentTarget.style.background = "#6366f1")
                          }
                        >
                          <MessageCircle size={18} />
                        </button>

                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedPatient(patient);
                            setView("scan");
                          }}
                          style={{
                            padding: "8px 16px",
                            background: "#8b5cf6",
                            color: "white",
                            border: "none",
                            borderRadius: "8px",
                            cursor: "pointer",
                            fontSize: "14px",
                            fontWeight: "500",
                            transition: "all 0.2s",
                          }}
                          onMouseEnter={(e) =>
                            (e.currentTarget.style.background = "#7c3aed")
                          }
                          onMouseLeave={(e) =>
                            (e.currentTarget.style.background = "#8b5cf6")
                          }
                        >
                          View Details
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Chat View */}
          {view === "chat" && (
            <EnhancedChat
              user={user}
              selectedPatient={selectedPatient}
              patients={patients}
              onSelectPatient={setSelectedPatient}
              darkMode={darkMode}
              socket={socket}
            />
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

      {showAddPatientModal && (
        <AddPatientModal
          isOpen={showAddPatientModal}
          onClose={() => setShowAddPatientModal(false)}
          onPatientAdded={handlePatientAdded}
          darkMode={darkMode}
        />
      )}

      {showPatientInfoModal && (
        <SimplePatientInfoModal
          onClose={() => setShowPatientInfoModal(false)}
          onSubmit={handlePatientInfoSubmit}
          darkMode={darkMode}
        />
      )}

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
  return (
    <button
      onClick={onClick}
      style={{
        width: "100%",
        padding: "14px 16px",
        marginBottom: "6px",
        display: "flex",
        alignItems: "center",
        gap: "14px",
        background: active ? "#6366f1" : "transparent",
        color: active ? "white" : "#64748b",
        border: "none",
        borderRadius: "12px",
        cursor: "pointer",
        transition: "all 0.2s",
        position: "relative",
      }}
    >
      {icon}
      <span style={{ flex: 1, textAlign: "left" }}>{label}</span>
      {badge && (
        <span
          style={{
            background: "#ef4444",
            color: "white",
            borderRadius: "50%",
            width: "20px",
            height: "20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "11px",
            fontWeight: "bold",
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
              background: "#6366f1",
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
