function loadTasks() {
        fetch('/tasks/data')
          .then(response => response.json())
          .then(data => {
            const tableBody = document.getElementById('tasks-table-body');
            tableBody.innerHTML = '';

            if (data.length === 0) {
              tableBody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">No tasks found</td></tr>';
              return;
            }

            data.reverse().forEach(task => {
              const statusClass = task.status.toLowerCase();
              const statusBadge = `<span class="status-badge ${statusClass}">${task.status}</span>`;
              
              let rowHTML = '';
              if (task.status == "SUCCESS")
              {
                let summaryHtml = `Total records: ${task.desc["total_records"]}`;
                if (task.desc["frauds_detected"] !== undefined) {
                  summaryHtml += ` <br> Detected frauds: ${task.desc["frauds_detected"]}`;
                  summaryHtml += ` <br> Legitimate entries: ${task.desc["legitimate"]}`;
                } else if (task.desc["anomalies_detected"] !== undefined) {
                  summaryHtml += ` <br> Detected anomalies: ${task.desc["anomalies_detected"]}`;
                  summaryHtml += ` <br> Normal entries: ${task.desc["normal"]}`;
                  summaryHtml += ` <br> Clusters: ${task.desc["clusters"]}`;
                }

                rowHTML = `
                <td>${task.id}</td>
                <td>${statusBadge}</td>
                <td>${summaryHtml}</td>
                `;
              }
              else if (task.status == "FAIL")
              {
                const errorList = Array.isArray(task.desc) 
                  ? task.desc.join('<br>') 
                  : task.desc;
                rowHTML = `
                <td>${task.id}</td>
                <td>${statusBadge}</td>
                <td>${errorList}</td>
                `;
              }
              else
              {
                rowHTML = `
                <td>${task.id}</td>
                <td>${statusBadge}</td>
                <td> </td>
                `;
              }

              const row = document.createElement('tr');
              row.className = 'task-row';
              row.innerHTML = rowHTML;
              
              if (task.status == "SUCCESS") {
                row.style.cursor = 'pointer';
                row.addEventListener('click', () => {
                  window.location.href = `/taskDetails?id=${task.id}`;
                });
              }
              
              tableBody.appendChild(row);
            });
          })
          .catch(error => {
            console.error('Error loading tasks:', error);
            const tableBody = document.getElementById('tasks-table-body');
            tableBody.innerHTML = '<tr><td colspan="3" class="text-center text-danger">Error loading tasks</td></tr>';
          });
      }

      function cancelTask(taskId) {
        fetch(`/cancel/${taskId}`, { method: 'POST' })
          .then(response => response.json())
          .then(() => loadTasks())
          .catch(error => console.error('Error cancelling task:', error));
      }

      document.addEventListener('DOMContentLoaded', loadTasks);