// ============================================================
// FRUIT RIPENESS CLASSIFIER - CLIENT-SIDE LOGIC
// ============================================================

console.log('🍌🥭 Fruit Classifier Script Loaded!');

// ============================================================
// GLOBAL VARIABLES
// ============================================================

let selectedFile = null;
let predictionChart = null;

// ============================================================
// DOM ELEMENTS
// ============================================================

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultSection = document.getElementById('resultSection');
const newPredictionBtn = document.getElementById('newPredictionBtn');

// Result elements
const fruitType = document.getElementById('fruitType');
const ripenessLevel = document.getElementById('ripenessLevel');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const inferenceTime = document.getElementById('inferenceTime');

// ============================================================
// EVENT LISTENERS
// ============================================================

// File input change
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop events
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);

// Click upload area to trigger file input
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Button clicks
predictBtn.addEventListener('click', predictImage);
resetBtn.addEventListener('click', resetUpload);
newPredictionBtn.addEventListener('click', resetUpload);

// ============================================================
// FILE HANDLING FUNCTIONS
// ============================================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showAlert('Format file tidak valid! Gunakan JPG, JPEG, atau PNG.', 'danger');
        return;
    }
    
    // Validate file size (max 5MB)
    const maxSize = 5 * 1024 * 1024; // 5MB in bytes
    if (file.size > maxSize) {
        showAlert('Ukuran file terlalu besar! Maksimal 5MB.', 'danger');
        return;
    }
    
    selectedFile = file;
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        showPreview();
    };
    reader.readAsDataURL(file);
}

// ============================================================
// UI STATE FUNCTIONS
// ============================================================

function showPreview() {
    uploadArea.classList.add('d-none');
    previewSection.classList.remove('d-none');
    previewSection.classList.add('fade-in');
    resultSection.classList.add('d-none');
    loadingIndicator.classList.add('d-none');
}

function showLoading() {
    predictBtn.disabled = true;
    resetBtn.disabled = true;
    loadingIndicator.classList.remove('d-none');
    loadingIndicator.classList.add('fade-in');
    resultSection.classList.add('d-none');
}

function hideLoading() {
    predictBtn.disabled = false;
    resetBtn.disabled = false;
    loadingIndicator.classList.add('d-none');
}

function showResults() {
    resultSection.classList.remove('d-none');
    resultSection.classList.add('fade-in');
    
    // Smooth scroll to results
    setTimeout(() => {
        resultSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    
    uploadArea.classList.remove('d-none');
    previewSection.classList.add('d-none');
    resultSection.classList.add('d-none');
    loadingIndicator.classList.add('d-none');
    
    // Destroy chart if exists
    if (predictionChart) {
        predictionChart.destroy();
        predictionChart = null;
    }
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// PREDICTION FUNCTION
// ============================================================

async function predictImage() {
    if (!selectedFile) {
        showAlert('Silakan pilih gambar terlebih dahulu!', 'warning');
        return;
    }
    
    showLoading();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        console.log('📤 Sending request to /predict...');
        
        // Send POST request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 Response received:', response.status);
        
        // Parse JSON response
        const data = await response.json();
        
        console.log('📊 Data:', data);
        
        if (data.success) {
            // Display results
            displayResults(data);
        } else {
            // Show error message
            showAlert(data.message || 'Prediksi gagal. Silakan coba lagi.', 'danger');
            hideLoading();
        }
        
    } catch (error) {
        console.error('❌ Error:', error);
        showAlert('Terjadi kesalahan saat menghubungi server. Pastikan server Flask sedang berjalan.', 'danger');
        hideLoading();
    }
}

// ============================================================
// DISPLAY RESULTS FUNCTION
// ============================================================

function displayResults(data) {
    hideLoading();
    
    // Display fruit type with emoji
    fruitType.innerHTML = `${data.emoji} ${data.fruit_type}`;
    
    // Display ripeness level with badge color
    const colorMap = {
        'success': 'bg-success',
        'warning': 'bg-warning',
        'danger': 'bg-danger',
        'info': 'bg-info'
    };
    
    const badgeClass = colorMap[data.color] || 'bg-secondary';
    ripenessLevel.innerHTML = `<span class="badge ${badgeClass} fs-4">${data.ripeness}</span>`;
    
    // Display confidence score
    const confidence = data.confidence;
    confidenceText.textContent = confidence.toFixed(2) + '%';
    confidenceBar.style.width = confidence + '%';
    
    // Color confidence bar based on value
    let barColorClass = 'bg-success';
    if (confidence >= 90) {
        barColorClass = 'bg-success';
    } else if (confidence >= 70) {
        barColorClass = 'bg-warning';
    } else {
        barColorClass = 'bg-danger';
    }
    
    confidenceBar.className = `progress-bar ${barColorClass} progress-bar-striped progress-bar-animated`;
    
    // Display inference time
    if (data.inference_time_ms) {
        inferenceTime.textContent = `${data.inference_time_ms.toFixed(2)}ms`;
    }
    
    // Create prediction chart
    createChart(data.all_predictions);
    
    // Show results section
    showResults();
    
    // Show success notification
    showAlert(`✅ ${data.message}`, 'success');
}

// ============================================================
// CHART FUNCTION
// ============================================================

function createChart(predictions) {
    // Destroy existing chart
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    // Prepare data
    const labels = [];
    const values = [];
    const colors = [
        'rgba(255, 99, 132, 0.8)',   // Red
        'rgba(54, 162, 235, 0.8)',   // Blue
        'rgba(255, 206, 86, 0.8)',   // Yellow
        'rgba(75, 192, 192, 0.8)',   // Teal
        'rgba(153, 102, 255, 0.8)',  // Purple
        'rgba(255, 159, 64, 0.8)'    // Orange
    ];
    
    // Extract data from predictions object
    let index = 0;
    for (const [className, predData] of Object.entries(predictions)) {
        labels.push(predData.display_name || className.replace('_', ' '));
        values.push(predData.probability || 0);
        index++;
    }
    
    // Create chart
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: colors.slice(0, labels.length).map(c => c.replace('0.8', '1')),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
                    callbacks: {
                        label: function(context) {
                            return `Confidence: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        },
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 10,
                            weight: 'bold'
                        },
                        maxRotation: 45,
                        minRotation: 45
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// ============================================================
// ALERT/NOTIFICATION FUNCTION
// ============================================================

function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlert = document.querySelector('.alert-custom');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show alert-custom`;
    alertDiv.style.position = 'fixed';
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.style.minWidth = '300px';
    alertDiv.style.maxWidth = '500px';
    alertDiv.style.boxShadow = '0 4px 20px rgba(0,0,0,0.2)';
    alertDiv.style.borderRadius = '10px';
    
    alertDiv.innerHTML = `
        <strong>${getAlertIcon(type)}</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

function getAlertIcon(type) {
    const icons = {
        'success': '✅',
        'danger': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    };
    return icons[type] || 'ℹ️';
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ DOM fully loaded');
    console.log('🎨 UI initialized');
    
    // Check if backend is accessible
    checkBackendHealth();
});

async function checkBackendHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('✅ Backend is healthy');
            console.log('📊 Model loaded:', data.model_loaded);
            console.log('🎯 Classes:', data.classes);
            
            if (!data.model_loaded) {
                console.warn('⚠️ Model not loaded! Predictions will not work.');
            }
        }
    } catch (error) {
        console.error('❌ Backend health check failed:', error);
        console.warn('⚠️ Make sure Flask server is running!');
    }
}

// ============================================================
// PREVENT DEFAULT DRAG & DROP ON WINDOW
// ============================================================

window.addEventListener('dragover', function(e) {
    e.preventDefault();
}, false);

window.addEventListener('drop', function(e) {
    e.preventDefault();
}, false);

// ============================================================
// LOG SCRIPT LOADED
// ============================================================

console.log('✅ Script initialization complete!');
console.log('📍 Ready to classify fruits! 🍌🥭');