// ===== Global Variables =====
let selectedFile = null;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];

// ===== DOM Elements =====
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const uploadSection = document.getElementById('upload-section');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const analyzeAnotherBtn = document.getElementById('analyze-another');
const tryAgainBtn = document.getElementById('try-again');
const fileSelectedInfo = document.getElementById('file-selected-info');

// ===== Event Listeners =====
document.addEventListener('DOMContentLoaded', () => {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Upload button
    uploadBtn.addEventListener('click', handleUpload);
    
    // Analyze another button
    analyzeAnotherBtn.addEventListener('click', resetInterface);
    
    // Try again button
    tryAgainBtn.addEventListener('click', resetInterface);
});

// ===== File Handling Functions =====

/**
 * Handle file selection from file input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Process and validate the selected file
 */
function processFile(file) {
    const validation = validateFile(file);
    
    if (!validation.valid) {
        displayError(validation.error);
        return;
    }
    
    selectedFile = file;
    
    // Update UI to show file is selected
    fileSelectedInfo.style.display = 'block';
    fileSelectedInfo.querySelector('.selected-file-name').textContent = `✓ ${file.name}`;
    fileSelectedInfo.querySelector('.selected-file-size').textContent = formatFileSize(file.size);
    
    // Enable upload button
    uploadBtn.disabled = false;
}

/**
 * Validate file type and size
 */
function validateFile(file) {
    // Check file type
    if (!ALLOWED_TYPES.includes(file.type)) {
        return {
            valid: false,
            error: 'Invalid file type. Please upload a JPG, PNG, BMP, or TIFF image.'
        };
    }
    
    // Check file size
    if (file.size > MAX_FILE_SIZE) {
        return {
            valid: false,
            error: `File is too large. Maximum size is ${formatFileSize(MAX_FILE_SIZE)}.`
        };
    }
    
    if (file.size === 0) {
        return {
            valid: false,
            error: 'File is empty. Please select a valid image.'
        };
    }
    
    return { valid: true };
}

// ===== Upload and Prediction Functions =====

/**
 * Handle upload button click
 */
async function handleUpload() {
    if (!selectedFile) {
        displayError('Please select an image file first.');
        return;
    }
    
    await uploadAndPredict(selectedFile);
}

/**
 * Upload image and get prediction
 */
async function uploadAndPredict(file) {
    try {
        showLoading();
        
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.success) {
            displayResults(data, file);
        } else {
            displayError(data.error || 'An error occurred during prediction.');
        }
        
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        displayError('Network error. Please check your connection and try again.');
    }
}

// ===== UI Update Functions =====

/**
 * Show loading state
 */
function showLoading() {
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'block';
}

/**
 * Hide loading state
 */
function hideLoading() {
    loadingSection.style.display = 'none';
}

/**
 * Display prediction results
 */
function displayResults(data, file) {
    // Hide other sections
    uploadSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Show image preview
    const imagePreview = document.getElementById('image-preview');
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Display prediction label
    const predictionLabel = document.getElementById('prediction-label');
    const prediction = data.prediction.toUpperCase();
    const emoji = prediction === 'REAL' ? '✅' : '⚠️';
    predictionLabel.textContent = `${emoji} ${prediction} ${emoji}`;
    predictionLabel.className = `prediction-label ${data.prediction}`;
    
    // Display confidence score
    const confidenceScore = document.getElementById('confidence-score');
    confidenceScore.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    
    // Update probability bars
    updateProbabilityBars(data.probabilities.real, data.probabilities.fake);
    
    // Display metadata
    document.getElementById('proc-time').textContent = formatProcessingTime(data.processing_time);
    document.getElementById('timestamp').textContent = formatTimestamp(data.timestamp);
    
    // Show results section
    resultsSection.style.display = 'block';
}

/**
 * Update probability bars with animation
 */
function updateProbabilityBars(realProb, fakeProb) {
    const realBar = document.getElementById('real-bar');
    const fakeBar = document.getElementById('fake-bar');
    const realPercentage = document.getElementById('real-percentage');
    const fakePercentage = document.getElementById('fake-percentage');
    
    // Convert to percentages
    const realPercent = (realProb * 100).toFixed(1);
    const fakePercent = (fakeProb * 100).toFixed(1);
    
    // Animate bars
    setTimeout(() => {
        realBar.style.width = `${realPercent}%`;
        fakeBar.style.width = `${fakePercent}%`;
    }, 100);
    
    // Update percentage text
    realPercentage.textContent = `${realPercent}%`;
    fakePercentage.textContent = `${fakePercent}%`;
}

/**
 * Display error message
 */
function displayError(message) {
    uploadSection.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingSection.style.display = 'none';
    
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    
    errorSection.style.display = 'block';
}

/**
 * Reset interface to initial state
 */
function resetInterface() {
    // Reset file selection
    selectedFile = null;
    fileInput.value = '';
    uploadBtn.disabled = true;
    
    // Hide file selected info
    fileSelectedInfo.style.display = 'none';
    
    // Reset probability bars
    document.getElementById('real-bar').style.width = '0%';
    document.getElementById('fake-bar').style.width = '0%';
    
    // Show upload section
    uploadSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    loadingSection.style.display = 'none';
}

// ===== Utility Functions =====

/**
 * Format file size in human-readable format
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format timestamp in readable format
 */
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * Format processing time
 */
function formatProcessingTime(seconds) {
    if (seconds < 1) {
        return `${(seconds * 1000).toFixed(0)}ms`;
    }
    return `${seconds.toFixed(2)}s`;
}
