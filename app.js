/* ====================================================================
   JavaScript for ML Visualization Project
   ==================================================================== */

// Sample Data (Replace with real data from your Python analysis)
const sampleData = {
    correlations: {
        "prev_sem_gpa": 0.89,
        "study_hours_per_week": 0.67,
        "attendance_percentage": 0.58,
        "sleep_hours_per_day": 0.45,
        "english_proficiency_score": 0.42,
        "social_support_score": 0.38,
        "financial_support": 0.22,
        "family_income_usd": 0.18,
        "stress_level": -0.52,
        "work_hours_per_week": -0.35,
        "screen_time_hours": -0.28,
        "homesickness_level": -0.31
    },
    gpaStats: {
        mean: 2.52,
        median: 2.51,
        std: 0.43,
        min: 1.98,
        max: 3.98
    },
    modelPerformance: {
        "Gradient Boosting": { r2: 0.824, rmse: 0.312, mae: 0.189 },
        "Random Forest": { r2: 0.798, rmse: 0.356, mae: 0.234 },
        "CatBoost": { r2: 0.785, rmse: 0.378, mae: 0.251 },
        "Linear Regression": { r2: 0.654, rmse: 0.521, mae: 0.389 },
        "Ridge Regression": { r2: 0.651, rmse: 0.525, mae: 0.394 },
        "Lasso Regression": { r2: 0.628, rmse: 0.562, mae: 0.426 },
        "SVR": { r2: 0.612, rmse: 0.589, mae: 0.451 },
        "KNN": { r2: 0.598, rmse: 0.608, mae: 0.472 }
    },
    featureImportance: {
        "prev_sem_gpa": 0.285,
        "study_hours_per_week": 0.198,
        "attendance_percentage": 0.156,
        "english_proficiency_score": 0.089,
        "sleep_hours_per_day": 0.067,
        "stress_level": 0.062,
        "social_support_score": 0.051,
        "screen_time_hours": 0.041,
        "work_hours_per_week": 0.034,
        "family_income_usd": 0.018
    },
    profiles: {
        "high-performer": {
            name: "High Performer",
            prev_gpa: 3.8,
            study_hours: 35,
            attendance: 95,
            sleep_hours: 7,
            stress_level: 4,
            english_score: 110
        },
        "average-performer": {
            name: "Average Performer",
            prev_gpa: 2.8,
            study_hours: 20,
            attendance: 75,
            sleep_hours: 6,
            stress_level: 7,
            english_score: 85
        },
        "at-risk": {
            name: "At Risk",
            prev_gpa: 2.0,
            study_hours: 8,
            attendance: 50,
            sleep_hours: 5,
            stress_level: 9,
            english_score: 70
        }
    }
};

// ====================================================================
// UTILITY FUNCTIONS
// ====================================================================

function updateSliderValue(inputId, displayId, unit = '') {
    const input = document.getElementById(inputId);
    const display = document.getElementById(displayId);
    
    if (input && display) {
        input.addEventListener('input', (e) => {
            display.textContent = e.target.value + unit;
        });
    }
}

function setupSliders() {
    updateSliderValue('prev-gpa', 'prev-gpa-value');
    updateSliderValue('study-hours', 'study-hours-value', ' hours');
    updateSliderValue('attendance', 'attendance-value', '%');
    updateSliderValue('sleep-hours', 'sleep-hours-value', ' hours');
    updateSliderValue('stress-level', 'stress-level-value');
    updateSliderValue('english-score', 'english-score-value');
}

// ====================================================================
// VISUALIZATION FUNCTIONS
// ====================================================================

function createCorrelationChart() {
    const data = sampleData.correlations;
    const sortedKeys = Object.keys(data).sort((a, b) => data[b] - data[a]);
    const values = sortedKeys.map(key => data[key]);
    const colors = values.map(v => v > 0 ? '#10b981' : '#ef4444');
    
    const trace = {
        x: values,
        y: sortedKeys.map(key => key.replace(/_/g, ' ').toUpperCase()),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: colors,
            line: {
                color: 'rgba(0,0,0,0.1)',
                width: 1
            }
        }
    };
    
    const layout = {
        title: 'Correlation with Semester GPA',
        xaxis: {
            title: 'Correlation Coefficient',
            range: [-1, 1],
            zeroline: true,
            zerolinewidth: 2,
            zerolinecolor: '#ccc'
        },
        yaxis: {
            automargin: true
        },
        height: 500,
        margin: {
            l: 200,
            r: 50,
            t: 80,
            b: 50
        },
        plot_bgcolor: '#f8fafc',
        paper_bgcolor: 'white',
        font: {
            family: 'system-ui, -apple-system, sans-serif',
            color: '#1e293b'
        }
    };
    
    Plotly.newPlot('correlation-chart', [trace], layout, { responsive: true });
}

function createDistributionChart() {
    const trace = {
        x: Array.from({ length: 100 }, () => {
            // Normal distribution approximation
            let u = 0, v = 0;
            while(u === 0) u = Math.random();
            while(v === 0) v = Math.random();
            let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            return sampleData.gpaStats.mean + z * sampleData.gpaStats.std;
        }),
        type: 'histogram',
        nbinsx: 30,
        marker: {
            color: '#2563eb',
            line: {
                color: '#1e40af',
                width: 1
            }
        }
    };
    
    const layout = {
        title: 'Semester GPA Distribution',
        xaxis: {
            title: 'Semester GPA'
        },
        yaxis: {
            title: 'Frequency'
        },
        height: 400,
        plot_bgcolor: '#f8fafc',
        paper_bgcolor: 'white',
        font: {
            family: 'system-ui, -apple-system, sans-serif',
            color: '#1e293b'
        },
        showlegend: false
    };
    
    Plotly.newPlot('distribution-chart', [trace], layout, { responsive: true });
}

function createModelPerformanceCharts() {
    const models = Object.keys(sampleData.modelPerformance);
    const r2Scores = models.map(m => sampleData.modelPerformance[m].r2);
    const rmseScores = models.map(m => sampleData.modelPerformance[m].rmse);
    
    // R² Chart
    const r2Trace = {
        x: models,
        y: r2Scores,
        type: 'bar',
        marker: {
            color: r2Scores.map(r => r > 0.8 ? '#10b981' : r > 0.7 ? '#f59e0b' : '#ef4444'),
            line: {
                color: '#1e293b',
                width: 1
            }
        },
        text: r2Scores.map(r => r.toFixed(3)),
        textposition: 'outside'
    };
    
    const r2Layout = {
        title: 'R² Score Comparison',
        yaxis: {
            title: 'R² Score',
            range: [0, 1]
        },
        height: 400,
        xaxis: { automargin: true },
        margin: { b: 100 },
        plot_bgcolor: '#f8fafc',
        paper_bgcolor: 'white',
        font: {
            family: 'system-ui, -apple-system, sans-serif',
            color: '#1e293b'
        }
    };
    
    Plotly.newPlot('r2-chart', [r2Trace], r2Layout, { responsive: true });
    
    // RMSE Chart
    const rmseTrace = {
        x: models,
        y: rmseScores,
        type: 'bar',
        marker: {
            color: rmseScores.map(r => r < 0.4 ? '#10b981' : r < 0.6 ? '#f59e0b' : '#ef4444'),
            line: {
                color: '#1e293b',
                width: 1
            }
        },
        text: rmseScores.map(r => r.toFixed(3)),
        textposition: 'outside'
    };
    
    const rmseLayout = {
        title: 'RMSE Comparison',
        yaxis: {
            title: 'RMSE (lower is better)'
        },
        height: 400,
        xaxis: { automargin: true },
        margin: { b: 100 },
        plot_bgcolor: '#f8fafc',
        paper_bgcolor: 'white',
        font: {
            family: 'system-ui, -apple-system, sans-serif',
            color: '#1e293b'
        }
    };
    
    Plotly.newPlot('rmse-chart', [rmseTrace], rmseLayout, { responsive: true });
}

function createFeatureImportanceChart() {
    const data = sampleData.featureImportance;
    const sortedKeys = Object.keys(data).sort((a, b) => data[b] - data[a]);
    const values = sortedKeys.map(key => data[key]);
    
    const trace = {
        x: values,
        y: sortedKeys.map(key => key.replace(/_/g, ' ').toUpperCase()),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: '#2563eb',
            line: {
                color: '#1e40af',
                width: 1
            }
        }
    };
    
    const layout = {
        title: 'Feature Importance (Gradient Boosting)',
        xaxis: {
            title: 'Importance Score'
        },
        yaxis: {
            automargin: true
        },
        height: 400,
        margin: {
            l: 200,
            r: 50,
            t: 80,
            b: 50
        },
        plot_bgcolor: '#f8fafc',
        paper_bgcolor: 'white',
        font: {
            family: 'system-ui, -apple-system, sans-serif',
            color: '#1e293b'
        }
    };
    
    Plotly.newPlot('feature-importance', [trace], layout, { responsive: true });
}

function populateFactorsLists() {
    const positiveFactors = Object.entries(sampleData.correlations)
        .filter(([_, val]) => val > 0.3)
        .sort((a, b) => b[1] - a[1]);
    
    const negativeFactors = Object.entries(sampleData.correlations)
        .filter(([_, val]) => val < -0.2)
        .sort((a, b) => a[1] - b[1]);
    
    const positiveList = document.getElementById('positive-factors');
    if (positiveList) {
        positiveList.innerHTML = positiveFactors.map(([key, val]) => `
            <div class="factor-item">
                <strong>${key.replace(/_/g, ' ')}</strong>
                <span class="factor-correlation">r = ${val.toFixed(2)}</span>
            </div>
        `).join('');
    }
    
    const negativeList = document.getElementById('negative-factors');
    if (negativeList) {
        negativeList.innerHTML = negativeFactors.map(([key, val]) => `
            <div class="factor-item">
                <strong>${key.replace(/_/g, ' ')}</strong>
                <span class="factor-correlation">r = ${val.toFixed(2)}</span>
            </div>
        `).join('');
    }
}

// ====================================================================
// PREDICTION LOGIC
// ====================================================================

function predictGPA(prevGPA, studyHours, attendance, sleepHours, stressLevel, englishScore) {
    // Simplified prediction model based on correlations
    const baseGPA = 2.52; // mean GPA
    
    // Normalize inputs (0-1 scale)
    const normPrevGPA = (prevGPA / 4.0) * 0.89; // correlation 0.89
    const normStudyHours = Math.min(studyHours / 50, 1.0) * 0.67; // correlation 0.67
    const normAttendance = (attendance / 100) * 0.58; // correlation 0.58
    const normSleepHours = Math.min(sleepHours / 10, 1.0) * 0.45; // correlation 0.45
    const normStress = (1 - stressLevel / 10) * 0.52; // negative correlation
    const normEnglish = Math.min(englishScore / 120, 1.0) * 0.42; // correlation 0.42
    
    // Weighted sum
    const prediction = baseGPA + 
        (normPrevGPA * 0.4) +
        (normStudyHours * 0.25) +
        (normAttendance * 0.15) +
        (normSleepHours * 0.08) +
        (normStress * 0.07) +
        (normEnglish * 0.05);
    
    // Clamp to reasonable range
    return Math.max(1.5, Math.min(4.0, prediction));
}

function handlePrediction() {
    const prevGPA = parseFloat(document.getElementById('prev-gpa').value);
    const studyHours = parseInt(document.getElementById('study-hours').value);
    const attendance = parseInt(document.getElementById('attendance').value);
    const sleepHours = parseFloat(document.getElementById('sleep-hours').value);
    const stressLevel = parseInt(document.getElementById('stress-level').value);
    const englishScore = parseInt(document.getElementById('english-score').value);
    
    const predicted = predictGPA(prevGPA, studyHours, attendance, sleepHours, stressLevel, englishScore);
    
    // Update result display
    document.getElementById('result-placeholder').classList.add('hidden');
    const resultBox = document.getElementById('prediction-result');
    resultBox.classList.remove('hidden');
    
    document.getElementById('predicted-gpa').textContent = predicted.toFixed(2);
    document.getElementById('comp-prediction').textContent = predicted.toFixed(2);
    document.getElementById('comp-diff').textContent = (predicted - 2.52).toFixed(2);
    
    // Generate interpretation
    const interpretation = getInterpretation(predicted, prevGPA, studyHours, stressLevel);
    document.getElementById('result-text').textContent = interpretation;
}

function getInterpretation(gpa, prevGPA, studyHours, stress) {
    let text = '';
    
    if (gpa > 3.5) {
        text = `Excellent! Predicted GPA of ${gpa.toFixed(2)} suggests strong academic performance. `;
    } else if (gpa > 3.0) {
        text = `Good! Predicted GPA of ${gpa.toFixed(2)} indicates solid academic standing. `;
    } else if (gpa > 2.5) {
        text = `Moderate. Predicted GPA of ${gpa.toFixed(2)} is close to the average. `;
    } else {
        text = `At Risk. Predicted GPA of ${gpa.toFixed(2)} suggests challenges ahead. `;
    }
    
    if (studyHours < 15) {
        text += 'Increasing study hours could significantly improve your GPA. ';
    }
    
    if (stress > 7) {
        text += 'High stress levels are negatively impacting your academic performance. Consider stress management strategies. ';
    }
    
    if (prevGPA < 2.0) {
        text += 'Your previous semester GPA is a strong predictor - focused improvement efforts could help turn things around.';
    }
    
    return text;
}

function loadProfile(profileKey) {
    const profile = sampleData.profiles[profileKey];
    if (!profile) return;
    
    document.getElementById('prev-gpa').value = profile.prev_gpa;
    document.getElementById('study-hours').value = profile.study_hours;
    document.getElementById('attendance').value = profile.attendance;
    document.getElementById('sleep-hours').value = profile.sleep_hours;
    document.getElementById('stress-level').value = profile.stress_level;
    document.getElementById('english-score').value = profile.english_score;
    
    // Update all slider displays
    document.getElementById('prev-gpa-value').textContent = profile.prev_gpa;
    document.getElementById('study-hours-value').textContent = profile.study_hours + ' hours';
    document.getElementById('attendance-value').textContent = profile.attendance + '%';
    document.getElementById('sleep-hours-value').textContent = profile.sleep_hours + ' hours';
    document.getElementById('stress-level-value').textContent = profile.stress_level;
    document.getElementById('english-score-value').textContent = profile.english_score;
    
    // Auto-predict
    handlePrediction();
}

// ====================================================================
// EVENT LISTENERS
// ====================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Setup sliders
    setupSliders();
    
    // Create charts
    createCorrelationChart();
    createDistributionChart();
    createModelPerformanceCharts();
    createFeatureImportanceChart();
    populateFactorsLists();
    
    // Prediction button
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
    
    // Profile buttons
    document.querySelectorAll('.profile-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const profileKey = e.currentTarget.getAttribute('data-profile');
            loadProfile(profileKey);
        });
    });
    
    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
    
    console.log('✅ Visualization dashboard loaded successfully!');
});

// Add some style for factors items
const style = document.createElement('style');
style.textContent = `
.factor-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    margin-bottom: 0.75rem;
    background: white;
    border-radius: 8px;
    border-left: 4px solid #2563eb;
}

.factor-correlation {
    background: #f0f9ff;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-weight: 700;
    color: #2563eb;
    font-size: 0.85rem;
}
`;
document.head.appendChild(style);
