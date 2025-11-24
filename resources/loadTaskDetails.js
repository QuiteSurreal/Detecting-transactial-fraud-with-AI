function getTaskId() {
            const params = new URLSearchParams(window.location.search);
            return params.get('id');
        }

        function loadTaskDetails() {
            const taskId = getTaskId();
            fetch(`/tasks/data?id=${taskId}`)
              .then(response => response.json())
              .then(data => {
                const detailsDiv = document.getElementById('task-details');

                let fraudListHtml = '';
                if (Array.isArray(data.frauds) && data.frauds.length > 0) {
                    fraudListHtml = data.frauds.map(f => {
                        const t = f.type;
                        const amt = f.amount;
                        const from = f.nameOrig;
                        const to = f.nameDest;
                        return `<li><strong>Type:</strong> ${t} &nbsp; <strong>Amount:</strong> ${amt} &nbsp; <strong>From:</strong> ${from} &nbsp; <strong>To:</strong> ${to}</li>`;
                    }).join('');
                } else {
                    fraudListHtml = '<li>No frauds</li>';
                }

                detailsDiv.innerHTML = `
                    <div class="card-body">
                        <h2>Task ID: ${data.id}</h2>
                        <p><strong>Status:</strong> <span class="status-badge ${data.status.toLowerCase()}">${data.status}</span></p>
                        <h3>Statistics</h3>
                        <ul>
                            <li><strong>Total Records:</strong> ${data.desc.total_records}</li>
                            <li><strong>Frauds Detected:</strong> ${data.desc.frauds_detected}</li>
                            <li><strong>Legitimate Entries:</strong> ${data.desc.legitimate}</li>
                        </ul>
                        <h3>Frauds</h3>
                        <ul>
                          ${fraudListHtml}
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