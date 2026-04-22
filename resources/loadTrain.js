// Model configurations with hyperparameters and defaults
const modelConfigs = {
  xgb: {
    name: 'XGBoost',
    description: 'XGBoost classifier optimized for fraud detection with gradient boosting.',
    params: [
      { name: 'n_estimators', label: 'Number of Trees', type: 'number', min: 100, max: 2000, step: 100, default: 500, info: 'More trees = better but slower' },
      { name: 'max_depth', label: 'Tree Depth', type: 'number', min: 3, max: 15, step: 1, default: 6, info: 'Controls tree complexity (prevent overfitting)' },
      { name: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.001, max: 0.5, step: 0.001, default: 0.1, info: 'Speed of learning (0.001-0.5), lower = slower but better' },
      { name: 'scale_pos_weight', label: 'Positive Class Weight', type: 'number', min: 1, max: 500, step: 0.1, default: 99.9, info: 'Weight for imbalanced data (fraud weight)' },
      { name: 'eval_metric', label: 'Evaluation Metric', type: 'select', options: ['logloss', 'aucpr', 'auc'], default: 'logloss', info: 'Metric to optimize during training' },
    ]
  },
  xgb_smote: {
    name: 'XGBoost with SMOTE',
    description: 'XGBoost with SMOTE (Synthetic Minority Over-sampling Technique) for better handling of imbalanced data.',
    params: [
      { name: 'n_estimators', label: 'Number of Trees', type: 'number', min: 100, max: 2000, step: 100, default: 1000, info: 'More trees = better but slower' },
      { name: 'max_depth', label: 'Tree Depth', type: 'number', min: 3, max: 15, step: 1, default: 6, info: 'Controls tree complexity (prevent overfitting)' },
      { name: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.001, max: 0.5, step: 0.001, default: 0.01, info: 'Speed of learning (0.001-0.5), lower = slower but better' },
      { name: 'subsample', label: 'Subsample Ratio', type: 'number', min: 0.1, max: 1.0, step: 0.1, default: 0.8, info: 'Fraction of samples used for fitting trees' },
      { name: 'smote_sampling_strategy', label: 'SMOTE Sampling Strategy', type: 'number', min: 0.1, max: 1.0, step: 0.1, default: 0.5, info: 'Ratio of minority to majority class after SMOTE' },
      { name: 'eval_metric', label: 'Evaluation Metric', type: 'select', options: ['logloss', 'aucpr', 'auc'], default: 'aucpr', info: 'Metric to optimize during training' },
    ]
  },
  ensemble: {
    name: 'Ensemble (Random Forest + XGBoost)',
    description: 'Stacking ensemble combining Random Forest and XGBoost predictions with Logistic Regression meta-learner.',
    params: [
      { name: 'rf_n_estimators', label: 'RF: Number of Trees', type: 'number', min: 100, max: 1000, step: 100, default: 500, info: 'Random Forest trees' },
      { name: 'rf_max_depth', label: 'RF: Tree Depth', type: 'number', min: 3, max: 15, step: 1, default: 6, info: 'Random Forest tree depth' },
      { name: 'xgb_n_estimators', label: 'XGB: Number of Trees', type: 'number', min: 100, max: 2000, step: 100, default: 1000, info: 'XGBoost trees in ensemble' },
      { name: 'xgb_max_depth', label: 'XGB: Tree Depth', type: 'number', min: 3, max: 15, step: 1, default: 6, info: 'XGBoost tree depth' },
      { name: 'xgb_learning_rate', label: 'XGB: Learning Rate', type: 'number', min: 0.001, max: 0.5, step: 0.001, default: 0.01, info: 'XGBoost learning rate' },
      { name: 'smote_sampling_strategy', label: 'SMOTE Sampling Strategy', type: 'number', min: 0.1, max: 1.0, step: 0.1, default: 0.1, info: 'Ratio of minority to majority class' },
    ]
  }
};

document.addEventListener('DOMContentLoaded', function() {
  const baseModelSelect = document.getElementById('baseModel');
  const trainButton = document.getElementById('trainButton');
  const resetButton = document.getElementById('resetButton');

  baseModelSelect.addEventListener('change', function() {
    const selected = this.value;
    const descDiv = document.getElementById('modelDescription');
    const paramsContainer = document.getElementById('parametersContainer');

    if (!selected) {
      descDiv.style.display = 'none';
      paramsContainer.innerHTML = '';
      trainButton.disabled = true;
      return;
    }

    const config = modelConfigs[selected];
    descDiv.textContent = config.description;
    descDiv.style.display = 'block';
    trainButton.disabled = false;

    paramsContainer.innerHTML = '';
    config.params.forEach(param => {
      const paramDiv = document.createElement('div');
      paramDiv.className = 'col-md-6 mb-3';

      if (param.type === 'select') {
        paramDiv.innerHTML = `
          <label for="${param.name}" class="form-label">${param.label}</label>
          <select id="${param.name}" name="${param.name}" class="form-control">
            ${param.options.map(opt => `<option value="${opt}" ${opt === param.default ? 'selected' : ''}>${opt}</option>`).join('')}
          </select>
          <small class="form-text text-muted">${param.info}</small>
        `;
      } else {
        paramDiv.innerHTML = `
          <label for="${param.name}" class="form-label">${param.label}</label>
          <input type="${param.type}" id="${param.name}" name="${param.name}" 
                 min="${param.min}" max="${param.max}" step="${param.step}" 
                 value="${param.default}" class="form-control" required>
          <small class="form-text text-muted">${param.info}</small>
        `;
      }
      paramsContainer.appendChild(paramDiv);
    });
  });

  trainButton.addEventListener('click', async function() {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.style.display = 'none';
    errorDiv.textContent = '';

    const modelName = document.getElementById('modelName').value.trim();
    const baseModel = document.getElementById('baseModel').value;

    let errors = [];
    if (!modelName) {
      errors.push('Model name is required.');
    }
    if (!baseModel) {
      errors.push('Base model must be selected.');
    }

    const config = modelConfigs[baseModel];
    const hyperparameters = {};
    for (const param of config.params) {
      const input = document.getElementById(param.name);
      if (input) {
        const value = param.type === 'number' ? parseFloat(input.value) : input.value;
        if (param.type === 'number') {
          if (isNaN(value) || value < param.min || value > param.max) {
            errors.push(`${param.label} must be between ${param.min} and ${param.max}.`);
          }
        }
        hyperparameters[param.name] = value;
      }
    }

    if (errors.length > 0) {
      errorDiv.innerHTML = errors.join('<br>');
      errorDiv.style.display = 'block';
      return;
    }

    const upgradable = 1;

    if (baseModel == "ensemble") {
      upgradable = 0;
    }

    const trainData = {
      model_name: modelName,
      base_model: baseModel,
      upgradable: upgradable,
      hyperparameters: hyperparameters
    };

    trainButton.disabled = true;

    console.log("whih");

    try {
      await fetch('/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(trainData)
      });
    } catch (error) {
      console.error('Training request failed:', error);
      errorDiv.textContent = error.message;
      errorDiv.style.display = 'block';
    } finally {
      trainButton.disabled = false;
    }
  });



  // Reset button functionality
  resetButton.addEventListener('click', function() {
    document.getElementById('modelName').value = '';
    document.getElementById('baseModel').value = '';
    document.getElementById('modelDescription').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('parametersContainer').innerHTML = '';
    trainButton.disabled = true;
  });
});
