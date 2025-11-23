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
              const row = document.createElement('tr');
              row.className = 'task-row';
              
              const statusClass = task.status.toLowerCase();
              const statusBadge = `<span class="status-badge ${statusClass}">${task.status}</span>`;
              
              row.innerHTML = `
                <td>${task.id}</td>
                <td>${statusBadge}</td>
                <td>Total records: ${task.desc["total_records"] || ''} <br> 
                Detected frauds: ${task.desc["frauds_detected"] || ''} <br> 
                Legitimate entries: ${task.desc["legitimate"] || ''} </td>
              `;
              
              tableBody.appendChild(row);
            });
          })
          .catch(error => {
            console.error('Error loading tasks:', error);
            const tableBody = document.getElementById('tasks-table-body');
            tableBody.innerHTML = '<tr><td colspan="3" class="text-center text-danger">Error loading tasks</td></tr>';
          });
      }

      // Load tasks on page load
      document.addEventListener('DOMContentLoaded', loadTasks);

      // Auto-refresh every 5 seconds
      setInterval(loadTasks, 30000);