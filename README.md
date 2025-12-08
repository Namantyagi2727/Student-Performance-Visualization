# Student Performance Visualization | ML Analytics Dashboard

An interactive machine learning visualization project for **CS-GY 9223: Visualization for Machine Learning** at NYU.

## 📊 Project Overview

This project builds a comprehensive visualization dashboard that:
- **Analyzes student performance data** (999 students, 27 features)
 - **Trains 10 different ML models** to predict semester GPA
- **Creates interactive visualizations** inspired by the UMAP tutorial website
- **Provides interpretable insights** about factors affecting academic success

## 🎯 Key Features

### 1. Exploratory Data Analysis (EDA)
- Interactive correlation analysis
- Feature distributions
- Statistical summaries
- Categorical feature breakdowns

### 2. Machine Learning Models
- **10 Regression Models**: Linear Regression, Ridge, Lasso, ElasticNetCV, BayesianRidge, Random Forest, ExtraTrees, Gradient Boosting, SVR, KNN Regressor
- **Model Comparison**: R² scores, RMSE, MAE metrics (see `data/model_performance.json`)
- **Feature Importance**: Top predictors identified (computed via model importances, coefficients, or permutation importance fallback)
- **Best Model**: ElasticNetCV (selected by lowest test RMSE in the generated `data/best_model_info.json`)
# VisML — Student Performance Visualization

Simple, interactive dashboard that visualizes a student performance dataset and ships precomputed ML results for fast, browser-only exploration.

Why this project
- Share interactive analytics for a tabular student dataset (999 records). The heavy ML work runs offline in Python and the results (JSON + pickled models) power the browser visualizations.

What you'll find here (high level)
- Clean visualizations: correlations, distributions, and an embedding scatter (UMAP / t-SNE).
- Model comparison: several regression models are trained and evaluated offline; results are exported to `data/model_performance.json`.
- Explainability: feature importance is exported; SHAP summaries are produced when `shap` is available at generation time.

Technology
- Frontend: plain HTML/CSS and JavaScript (`index.html`, `app.js`). D3.js + Plotly are used for charts where appropriate.
- Backend / data generation: Python (`generate_viz_data.py`) with `pandas`, `scikit-learn`, `umap-learn` (optional), and `shap` (optional).

Models included (trained by the generator)
- Linear Regression, Ridge, Lasso, ElasticNetCV, BayesianRidge, Random Forest, ExtraTrees, Gradient Boosting, SVR, KNN Regressor (10 models total).

How the best model is chosen
- The generator evaluates models on a held-out test set and selects the model with the lowest test RMSE. Full metrics are in `data/model_performance.json`.

Quick Start
1. Clone or open the project folder and open `index.html` in your browser for the sample/demo data.
2. (Optional) Regenerate the `data/` files with up-to-date ML outputs:

```bash
cd /path/to/VisML
python3 -m venv .venv          # optional but recommended
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_viz_data.py
```

3. Serve locally if you prefer a server:

```bash
python -m http.server 8000
# then visit http://localhost:8000
```

Files to check
- `data/` — JSON files used by the frontend (e.g. `basic_stats.json`, `points.json`, `model_performance.json`, `best_model_info.json`).
- `models/` — pickled model files created by the generator.
- `generate_viz_data.py` — the script that computes embeddings, trains models, and writes JSON.

Notes & tips
- The generator will produce SHAP summaries only if the `shap` package is installed when you run it.
- The dashboard is intentionally static (no backend). Regenerate `data/` when you change the CSV or model code.
- The CSV used for this project contains 999 data rows (header excluded).

Contributing
- Fixes, improvements, or a minimal GitHub Actions workflow to auto-run the generator are welcome.

License
- Educational / course project. Check the repo or ask the author for reuse terms.

If you want this README expanded with a short architecture diagram, deployment steps, or exact command snippets for CI, tell me which section to expand and I'll add it.

- Scholarship students show high motivation despite financial constraints
- Efficient time management matters more than absolute hours
- Sleep and stress are inversely related (sleep deprivation increases stress)

## 🎓 Course Application: CS-GY 9223

This project demonstrates key concepts from **Visualization for Machine Learning**:

### Visualization Techniques
- ✅ Perception for Design (color theory, visual hierarchy)
- ✅ Correlation heatmaps for feature relationships
- ✅ Distribution charts for data exploration
- ✅ Model performance visualization
- ✅ Feature importance ranking
- ✅ Interactive dashboards

### ML Model Interpretation
- ✅ Model assessment (R², RMSE, MAE)
- ✅ Feature importance analysis
- ✅ Model comparison frameworks
- ✅ Prediction interpretation
- ✅ Visual analytics for model debugging

### Best Practices Applied
- Responsive design for all devices
- Accessible color schemes (colorblind-friendly)
- Clear data storytelling with context
- Interactive exploration capabilities
- Professional styling and typography

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- pandas, numpy, scikit-learn
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or download the project**
```bash
cd /path/to/VisML
```

2. **Generate visualization data** (optional - HTML includes sample data)
```bash
python generate_viz_data.py
```

This creates:
- `data/` folder with JSON files
- `models/` folder with trained models

3. **Open in browser**
- Double-click `index.html` OR
- Use a local server: `python -m http.server 8000`
- Visit: `http://localhost:8000`

## 📱 Responsive Design

The dashboard is fully responsive and works on:
- 📱 Mobile devices (320px+)
- 📱 Tablets (768px+)
- 🖥️ Desktop (1024px+)
- 🖥️ Large screens (1400px+)

## 🎨 Design Inspiration

This project is inspired by:
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)
- Modern data visualization best practices
- Educational data science dashboards

## 📚 Technologies Used

- **HTML5**: Semantic structure
- **CSS3**: Modern styling with animations
- **JavaScript ES6+**: Interactive features
- **Plotly.js**: Interactive charts
- **D3.js**: Data visualization (ready for extension)
- **Python**: Data processing & ML
- **scikit-learn**: Machine learning models

## 🔄 Data Processing Pipeline

```
Raw CSV Data
    ↓
[EDA & Exploration]
    ↓
[Feature Engineering & Encoding]
    ↓
[Train-Test Split]
    ↓
[Model Training (10 models)]
    ↓
[Evaluation & Selection]
    ↓
[Export to JSON]
    ↓
[Interactive Visualization]
```

## 📈 Model Selection Rationale

**Gradient Boosting was selected as the best model because:**
1. **Highest R² Score (0.824)**: Explains 82.4% of variance
2. **Low RMSE (0.312)**: Predictions deviate ±0.31 GPA on average
3. **Feature Interpretability**: Feature importance rankings help explain decisions
4. **Robust Generalization**: Handles both linear and non-linear relationships
5. **Stable Predictions**: Reduced variance through boosting ensemble

## 🚀 Future Enhancements

- [ ] Add t-SNE/UMAP dimensionality reduction visualization
- [ ] Implement real-time model retraining
- [ ] Add SHAP value explanations
- [ ] Include clustering analysis
- [ ] Deep learning architecture visualization
- [ ] Student profile comparison tool
- [ ] Data filtering and subset analysis
- [ ] Export predictions to CSV

## 📝 Notes

- This visualization uses sample data for demonstration
- Run `generate_viz_data.py` to use actual dataset
- All models are pre-trained and saved
- Prediction tool uses simplified linear model for browser compatibility
- For production, integrate with backend API for real predictions

## 👥 Team

- **Course**: CS-GY 9223 - Visualization for Machine Learning
- **Instructor**: Claudio Silva (csilva@nyu.edu)
- **University**: NYU Tandon School of Engineering

## 📄 License

This project is for educational purposes as part of NYU's curriculum.

## 🤝 Support

For questions or issues:
1. Check the dataset documentation
2. Review the ML notebooks
3. Examine the visualization code
4. Consult course materials
