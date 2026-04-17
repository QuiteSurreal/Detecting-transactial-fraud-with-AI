
fetch('/models')
      .then(response => response.json())
      .then(models => {
        const predict = document.getElementById('model_select');
        const upgradeButton = document.getElementById('upgrade-button');

        const updateUpgradeButton = () => {
          const selectedOption = predict.options[predict.selectedIndex];
          const upgradable = selectedOption && selectedOption.dataset.upgradable === '1';
          upgradeButton.disabled = !upgradable;
          upgradeButton.textContent = upgradable ? 'Upgrade this model' : 'This model cannot be upgraded';
        };

        for (const [key, value] of Object.entries(models)) {
          const option = document.createElement('option');
          option.value = key;
          option.textContent = `${key} – ${value.description}`;
          option.dataset.upgradable = value.upgradable;
          predict.appendChild(option);
        }

        predict.addEventListener('change', updateUpgradeButton);
        updateUpgradeButton();
      });