


fetch('/models')
      .then(response => response.json())
      .then(models => {
        const predict = document.getElementById('model_select_predict');
        for (const [key, value] of Object.entries(models)) {
          const option = document.createElement('option');
          option.value = key;
          option.textContent = `${key} – ${value.description}`;
          predict.appendChild(option);
        }
        const upgrade = document.getElementById('model_select_upgrade');
        for (const [key, value] of Object.entries(models)) {
          const option = document.createElement('option');
          option.value = key;
          option.textContent = `${key} – ${value.description}`;
          upgrade.appendChild(option);
        }
      });