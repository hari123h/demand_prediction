/**
 * PowerGrid AI - Electricity Demand Forecasting
 * Premium Interactive UI Script
 */

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    createParticles();
    setDefaultDateTime();
    initializeSliders();
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
                <span>Model Active</span>
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
        
        // Random properties
        const size = Math.random() * 4 + 2;
        const left = Math.random() * 100;
        const delay = Math.random() * 8;
        const duration = Math.random() * 4 + 6;
        const opacity = Math.random() * 0.5 + 0.2;
        
        // Random color from palette
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
 * Set default date and time to current
 */
function setDefaultDateTime() {
    const now = new Date();
    
    // Format date as YYYY-MM-DD
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    document.getElementById('date').value = `${year}-${month}-${day}`;
    
    // Format time as HH:MM
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    document.getElementById('time').value = `${hours}:${minutes}`;
}

/**
 * Initialize premium sliders
 */
function initializeSliders() {
    // Temperature slider
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    
    tempSlider.addEventListener('input', (e) => {
        tempValue.textContent = e.target.value;
        updateSliderBackground(e.target);
    });
    updateSliderBackground(tempSlider);
    
    // Humidity slider
    const humiditySlider = document.getElementById('humidity');
    const humidityValue = document.getElementById('humidityValue');
    
    humiditySlider.addEventListener('input', (e) => {
        humidityValue.textContent = e.target.value;
        updateSliderBackground(e.target);
    });
    updateSliderBackground(humiditySlider);
}

/**
 * Update slider track fill based on value
 */
function updateSliderBackground(slider) {
    const min = slider.min || 0;
    const max = slider.max || 100;
    const value = slider.value;
    const percentage = ((value - min) / (max - min)) * 100;
    
    // Update thermometer icon if it's the temperature slider
    if (slider.id === 'temperature') {
        const tempIcon = document.querySelector('.temp-icon');
        if (tempIcon) {
            if (value > 30) {
                tempIcon.classList.add('hot');
            } else {
                tempIcon.classList.remove('hot');
            }
        }
    }
    
    slider.style.background = `linear-gradient(to right, 
        rgba(99, 102, 241, 0.8) 0%, 
        rgba(139, 92, 246, 0.8) ${percentage}%, 
        rgba(99, 102, 241, 0.15) ${percentage}%, 
        rgba(99, 102, 241, 0.15) 100%)`;
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
        
        // Collect form data
        const formData = {
            date: document.getElementById('date').value,
            time: document.getElementById('time').value,
            temperature: document.getElementById('temperature').value,
            humidity: document.getElementById('humidity').value,
            windSpeed: document.getElementById('windSpeed').value
        };
        
        try {
            const response = await fetch('/api/predict', {
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
    animateCounter('demandValue', data.predicted_demand_mw, 0, 1500);
    
    // Update prediction meta
    const predictionDate = new Date(data.datetime);
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    document.getElementById('predictionMeta').textContent = 
        predictionDate.toLocaleDateString('en-US', options);
    
    // Update stats with animation
    setTimeout(() => {
        document.getElementById('statTemp').textContent = `${data.temperature}°C`;
    }, 200);
    setTimeout(() => {
        document.getElementById('statHumidity').textContent = `${data.humidity}%`;
    }, 300);
    setTimeout(() => {
        document.getElementById('statWind').textContent = `${data.wind_speed} km/h`;
    }, 400);
    
    // Update chart
    if (data.hourly_forecast) {
        updateChart(data.hourly_forecast);
    }
    
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
 * Custom Chart.js Plugin for vertical hover line
 */
const verticalLinePlugin = {
    id: 'verticalLine',
    afterDraw: (chart) => {
        if (chart.tooltip?._active?.length) {
            const ctx = chart.ctx;
            const x = chart.tooltip._active[0].element.x;
            const topY = chart.scales.y.top;
            const bottomY = chart.scales.y.bottom;

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, topY);
            ctx.lineTo(x, bottomY);
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.restore();
        }
    }
};

/**
 * Update the forecast chart
 */
let forecastChart = null;

function updateChart(hourlyData) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    const labels = hourlyData.map(d => d.hour);
    const data = hourlyData.map(d => d.demand);
    
    // Confidence Interval (+/- 5%)
    const upperData = data.map(v => v * 1.05);
    const lowerData = data.map(v => v * 0.95);
    // Destroy existing chart
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');
    
    const datasets = [
        {
            label: 'Predicted Demand (MW)',
            data: data,
            borderColor: '#6366f1',
            backgroundColor: gradient,
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 8,
            pointHoverBackgroundColor: '#6366f1',
            pointHoverBorderColor: '#ffffff',
            pointHoverBorderWidth: 3,
            order: 2
        },
        {
            label: 'Upper Bound',
            data: upperData,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            borderWidth: 0,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 0,
            order: 1
        },
        {
            label: 'Lower Bound',
            data: lowerData,
            borderColor: 'transparent',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderWidth: 0,
            fill: '-1', // Fill to previous dataset (Upper Bound)
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 0,
            order: 3
        }
    ];
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#ffffff',
                    titleColor: '#000000',
                    bodyColor: '#1a1a1a',
                    borderColor: 'rgba(255, 255, 255, 1)',
                    borderWidth: 0,
                    cornerRadius: 100, /* Pill shape */
                    padding: {
                        top: 8,
                        bottom: 8,
                        left: 16,
                        right: 16
                    },
                    titleFont: {
                        family: 'Outfit',
                        size: 13,
                        weight: '600'
                    },
                    bodyFont: {
                        family: 'Outfit',
                        size: 13
                    },
                    displayColors: false,
                    caretSize: 0, /* No pointer triangle */
                    yAlign: 'bottom',
                    filter: function(item) {
                       return !item.dataset.label.includes('Bound');
                    },
                    callbacks: {
                        title: () => null, /* Hide default title */
                        label: (item) => `Predicted Load: ${item.raw.toFixed(2)} MW | Time: ${item.label} | Temp: ${document.getElementById('tempValue').textContent}°C`
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.02)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.3)',
                        font: {
                            family: 'Outfit',
                            size: 11
                        },
                        maxRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.02)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.3)',
                        font: {
                            family: 'Outfit',
                            size: 11
                        },
                        callback: (value) => value.toFixed(0) + ' MW'
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        },
        plugins: [verticalLinePlugin]
    });
}

/**
 * Show error message
 */
function showError(message) {
    // Create error notification
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
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease-out forwards';
        setTimeout(() => notification.remove(), 400);
    }, 5000);
}

// Add keyframe animations via JavaScript
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
`;
document.head.appendChild(style);

/**
 * Intersection Observer for scroll animations
 */
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

