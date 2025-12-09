
let allStatsData = [];

function loadStatsData() {
    // Load statistics data
    fetch('/statsData')
      .then(response => response.json())
      .then(data => {
        allStatsData = data;
        populateStatistics(data);
        populateModelDropdown(data);
      })
      .catch(error => console.error('Error loading statistics:', error));

    // Handle model selection
    document.getElementById('modelSelector').addEventListener('change', function() {
      const selectedModel = this.value;
      if (selectedModel) {
        const modelData = allStatsData.find(item => item.id === selectedModel);
        if (modelData) {
          displayModelDashboard(modelData);
        }
      } else {
        document.getElementById('modelDashboard').style.display = 'none';
      }
    });

    function populateStatistics(data) {
      if (!data || data.length === 0) return;

      // Find total entry
      const totalStats = data.find(item => item.id === 'Total');
      
      if (totalStats) {
        // Update summary cards
        document.getElementById('totalRecords').textContent = totalStats.records.toLocaleString();
        document.getElementById('totalFrauds').textContent = totalStats.frauds.toLocaleString();
        document.getElementById('totalLegit').textContent = totalStats.legit.toLocaleString();
        
        const fraudRate = totalStats.records > 0 
          ? ((totalStats.frauds / totalStats.records) * 100).toFixed(2)
          : '0.00';
        document.getElementById('fraudRate').textContent = fraudRate + '%';

        // Update metrics
        document.getElementById('accuracy').textContent = parseFloat(totalStats.acc).toFixed(3);
        document.getElementById('precision').textContent = parseFloat(totalStats.prec).toFixed(3);
        document.getElementById('recall').textContent = parseFloat(totalStats.rec).toFixed(3);
        document.getElementById('f1score').textContent = parseFloat(totalStats.F1).toFixed(3);

        loadConfusionMatrixImage("Total");

        // Draw data distribution chart
        drawDataDistribution(totalStats.frauds, totalStats.legit);
      }
    }

    function populateModelDropdown(data) {
      const selector = document.getElementById('modelSelector');
      const models = data.filter(item => item.id !== 'Total');
      
      models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.id;
        selector.appendChild(option);
      });
    }

    function displayModelDashboard(modelData) {
      // Show dashboard
      document.getElementById('modelDashboard').style.display = 'block';
      document.getElementById('modelDashboardTitle').textContent = modelData.id + ' Statistics';

      // Update model summary cards
      document.getElementById('modelRecords').textContent = modelData.records.toLocaleString();
      document.getElementById('modelFrauds').textContent = modelData.frauds.toLocaleString();
      document.getElementById('modelLegit').textContent = modelData.legit.toLocaleString();
      
      const fraudRate = modelData.records > 0 
        ? ((modelData.frauds / modelData.records) * 100).toFixed(2)
        : '0.00';
      document.getElementById('modelFraudRate').textContent = fraudRate + '%';

      // Update model metrics
      document.getElementById('modelAccuracy').textContent = parseFloat(modelData.acc).toFixed(3);
      document.getElementById('modelPrecision').textContent = parseFloat(modelData.prec).toFixed(3);
      document.getElementById('modelRecall').textContent = parseFloat(modelData.rec).toFixed(3);
      document.getElementById('modelF1').textContent = parseFloat(modelData.F1).toFixed(3);

      // Load confusion matrix image
      loadConfusionMatrixImage(modelData.id);

      // Scroll to model dashboard
      setTimeout(() => {
        document.getElementById('modelDashboard').scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }

    function loadConfusionMatrixImage(modelName) {
      var img;
      var placeholder;
      if (modelName == "Total")
      {
        img = document.getElementById('confusionMatrixImg');
        placeholder = document.getElementById('confusionMatrixPlaceholder');
      }
      else
      {
        img = document.getElementById('modelConfusionMatrixImg');
        placeholder = document.getElementById('modelConfusionMatrixPlaceholder');
      }
      
      
      const imagePath = `../resources/temp/cm_${modelName}.png?t=${new Date().getTime()}`;
      
      
      img.src = imagePath;
      img.onload = function() {
        placeholder.style.display = 'none';
        img.style.display = 'block';
      };
      img.onerror = function() {
        placeholder.style.display = 'flex';
        img.style.display = 'none';
      };
    }

    function drawDataDistribution(frauds, legit) {
      const ctx = document.getElementById('dataDistributionChart').getContext('2d');
      
      new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: ['Fraudulent', 'Legitimate'],
          datasets: [{
            data: [frauds, legit],
            backgroundColor: [
              'rgba(255, 99, 132, 0.7)',
              'rgba(76, 175, 80, 0.7)'
            ],
            borderColor: [
              'rgba(255, 99, 132, 1)',
              'rgba(76, 175, 80, 1)'
            ],
            borderWidth: 2
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              position: 'bottom'
            }
          }
        }
      });
    }
}

function ImageExist(url) 
{
   var img = new Image();
   img.src = url;
   return img.height != 0;
}
    
document.addEventListener('DOMContentLoaded', loadStatsData);