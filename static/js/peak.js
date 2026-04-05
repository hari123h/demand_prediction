/**
 * PowerGrid AI - Peak Demand Forecasting
 * Premium Interactive UI Script
 */

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    createParticles();
    setDefaultDate();
    initializeForm();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Check model status
    checkModelStatus();
    
    // Add smooth reveal animations
    observeElements();
}

/**
 * Check if the ML model is loaded
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        const statusEl = document.querySelector('.header-status');
        if (data.loaded) {
            statusEl.innerHTML = `
                <div class="status-dot"></div>
                <span>Peak Model Active</span>
            `;
            statusEl.style.borderColor = 'rgba(16, 185, 129, 0.2)';
            statusEl.style.background = 'rgba(16, 185, 129, 0.1)';
            statusEl.style.color = '#10b981';
        } else {
            statusEl.innerHTML = `
                <div class="status-dot" style="background: #f59e0b;"></div>
                <span>Model Not Loaded</span>
            `;
            statusEl.style.borderColor = 'rgba(245, 158, 11, 0.2)';
            statusEl.style.background = 'rgba(245, 158, 11, 0.1)';
            statusEl.style.color = '#f59e0b';
        }
    } catch (error) {
        console.log('Could not check model status');
    }
}

/**
 * Create floating particles animation
 */
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 4 + 2;
        const left = Math.random() * 100;
        const delay = Math.random() * 8;
        const duration = Math.random() * 4 + 6;
        const opacity = Math.random() * 0.5 + 0.2;
        
        const colors = ['#6366f1', '#8b5cf6', '#22d3ee', '#10b981', '#a855f7'];
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${left}%;
            background: ${color};
            animation-delay: ${delay}s;
            animation-duration: ${duration}s;
            opacity: ${opacity};
        `;
        
        particlesContainer.appendChild(particle);
    }
}

/**
 * Set default date to current
 */
function setDefaultDate() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    document.getElementById('date').value = `${year}-${month}-${day}`;
}

/**
 * Initialize form submission
 */
function initializeForm() {
    const form = document.getElementById('forecastForm');
    const predictBtn = document.getElementById('predictBtn');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        
        // Collect form data (only date needed for peak)
        const formData = {
            date: document.getElementById('date').value,
        };
        
        try {
            const response = await fetch('/api/predict-peak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                displayResults(result);
            } else {
                showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            showError('Connection error. Please make sure the server is running.');
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });
}

/**
 * Display prediction results with animations
 */
function displayResults(data) {
    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');
    
    // Hide empty state, show results
    emptyState.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Animate demand value
    animateCounter('demandValue', data.predicted_peak_demand_mw, 0, 1500);
    
    // Update prediction meta
    const predictionDate = new Date(data.date);
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric'
    };
    document.getElementById('predictionMeta').textContent = 
        predictionDate.toLocaleDateString('en-US', options);
    
    // Scroll to results on mobile
    if (window.innerWidth < 1024) {
        document.getElementById('resultsPanel').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }
}

/**
 * Animate counter from start to end
 */
function animateCounter(elementId, endValue, startValue = 0, duration = 1000) {
    const element = document.getElementById(elementId);
    const startTime = performance.now();
    const range = endValue - startValue;
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out-expo)
        const easeOutExpo = 1 - Math.pow(2, -10 * progress);
        const currentValue = startValue + (range * easeOutExpo);
        
        element.textContent = currentValue.toFixed(2);
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

/**
 * Show error message
 */
function showError(message) {
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>${message}</span>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 24px;
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        color: #f43f5e;
        font-size: 0.95rem;
        z-index: 1000;
        animation: slideInRight 0.4s ease-out;
    `;
    
    notification.querySelector('svg').style.cssText = `
        width: 20px;
        height: 20px;
        flex-shrink: 0;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease-out forwards';
        setTimeout(() => notification.remove(), 400);
    }, 5000);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideOutRight {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100px); }
    }
`;
document.head.appendChild(style);

function observeElements() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.panel, .stat-card').forEach(el => {
        observer.observe(el);
    });
}
