function getTaskId() {
            const params = new URLSearchParams(window.location.search);
            return params.get('id');
        }

function loadTaskDetails() {
    const taskId = getTaskId();
    const url = taskId ? `/tasks/data?task_id=${encodeURIComponent(taskId)}` : '/tasks/data';
    
    fetch(url)
      .then(response => response.json())
      .then(data => {
        const detailsDiv = document.getElementById('task-details');

        const totalRecords = data.desc?.total_records ?? 0;
        const fraudsDetected = data.desc?.frauds_detected;
        const anomaliesDetected = data.desc?.anomalies_detected;
        const legitimateEntries = data.desc?.legitimate;
        const normalEntries = data.desc?.normal;
        const clusters = data.desc?.clusters;

        let statisticsHtml = `
              <li><strong>Total Records:</strong> ${totalRecords}</li>
        `;

        if (fraudsDetected !== undefined) {
          statisticsHtml += `
              <li><strong>Frauds Detected:</strong> ${fraudsDetected}</li>
              <li><strong>Legitimate Entries:</strong> ${legitimateEntries ?? 0}</li>
          `;
        } else if (anomaliesDetected !== undefined) {
          statisticsHtml += `
              <li><strong>Anomalies Detected:</strong> ${anomaliesDetected}</li>
              <li><strong>Normal Entries:</strong> ${normalEntries ?? 0}</li>
              <li><strong>Clusters:</strong> ${clusters ?? 0}</li>
          `;
        }

        let anomalyListHtml = '';
        if (Array.isArray(data.frauds) && data.frauds.length > 0) {
            anomalyListHtml = data.frauds.map(f => {
                const t = f.type ?? 'N/A';
                const amt = f.amount ?? 'N/A';
                const from = f.nameOrig ?? 'N/A';
                const to = f.nameDest ?? 'N/A';
                const cluster = f.cluster !== undefined ? ` &nbsp; <strong>Cluster:</strong> ${f.cluster}` : '';
                const anomalyTag = f.is_anomaly !== undefined ? ` &nbsp; <strong>Anomaly:</strong> ${f.is_anomaly}` : '';
                return `<li><strong>Type:</strong> ${t} &nbsp; <strong>Amount:</strong> ${amt} &nbsp; <strong>From:</strong> ${from} &nbsp; <strong>To:</strong> ${to}${cluster}${anomalyTag}</li>`;
            }).join('');
        } else {
            anomalyListHtml = '<li>No anomalous entries</li>';
        }

        detailsDiv.innerHTML = `
            <div class="card-body">
                <h2>Task ID: ${data.id}</h2>
                <p><strong>Status:</strong> <span class="status-badge ${data.status.toLowerCase()}">${data.status}</span></p>
                <h3>Statistics</h3>
                <ul>
                  ${statisticsHtml}
                </ul>
                <h3>Detected Entries</h3>
                <ul>
                  ${anomalyListHtml}
                </ul>
            </div>
        `;
      })
      .catch(error => {
        console.error('Error loading task details:', error);
        document.getElementById('task-details').innerHTML = '<p class="text-danger">Error loading task details</p>';
      });
}
document.addEventListener('DOMContentLoaded', loadTaskDetails);