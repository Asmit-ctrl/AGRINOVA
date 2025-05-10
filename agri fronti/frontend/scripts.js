// Function to toggle visibility of sections
function toggleSection(sectionId) {
  const section = document.getElementById(sectionId);
  if (section) {
    section.style.display = section.style.display === 'none' || section.style.display === '' ? 'block' : 'none';
  }
}

// Function to show a specific section
function showSection(sectionId) {
  const sections = document.querySelectorAll('.flex-1 > div');
  sections.forEach((section) => {
    section.classList.add('hidden'); // Hide all sections
    section.classList.remove('fade-in');
  });

  const targetSection = document.getElementById(sectionId);
  if (targetSection) {
    targetSection.classList.remove('hidden'); // Show the target section
    targetSection.classList.add('fade-in'); // Add fade-in animation
  } else {
    console.error(`Section with ID "${sectionId}" not found.`);
  }
}

const BASE_URL = 'http://localhost:5500'; // Ensure this matches the backend server URL

async function fetchJSON(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching chart data: ${error}`);
    throw error; // Re-throw the error for further handling
  }
}

// Fetch weather data from the backend
async function fetchWeatherData() {
  try {
    const response = await fetch(`${BASE_URL}/api/weather`);
    if (!response.ok) throw new Error('Failed to fetch weather data');
    const data = await response.json();

    console.log('Weather data received:', data); // Debug log to inspect the data structure

    // Ensure data is valid before updating the DOM
    if (data && data.city && data.forecast && data.forecast.length > 0) {
      // Update current weather details
      document.getElementById('currentCity').textContent = data.city || 'N/A';
      document.getElementById('currentDate').textContent = new Date(data.forecast[0].date).toLocaleDateString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
      document.getElementById('currentTemp').textContent = `${data.forecast[0].temp}°C` || 'N/A';
      document.getElementById('currentCondition').textContent = data.forecast[0].condition || 'N/A';
      document.getElementById('currentFeelsLike').textContent = `Feels like ${data.forecast[0].feels_like}°C` || 'N/A';

      // Add time to the weather data
      const currentTime = new Date(data.forecast[0].date).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
      });
      document.getElementById('currentTime').textContent = `Time: ${currentTime}`;

      // Update weather icon
      const weatherIcon = document.getElementById('currentWeatherIcon');
      weatherIcon.src = data.forecast[0].icon || '';
      weatherIcon.alt = data.forecast[0].condition || 'Weather Icon';

      // Populate forecast cards
      const forecastContainer = document.getElementById('forecastContainer');
      forecastContainer.innerHTML = ''; // Clear existing content
      data.forecast.forEach(day => {
        const date = new Date(day.date).toLocaleDateString('en-US', { weekday: 'long' });
        const time = new Date(day.date).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        const card = document.createElement('div');
        card.className = 'card p-4 text-center';
        card.innerHTML = `
          <h2 class="text-lg font-bold">${date}</h2>
          <p class="text-sm text-gray-500">${time}</p>
          <img src="${day.icon}" alt="${day.condition}" class="mx-auto mb-2" />
          <p class="text-gray-500 capitalize">${day.condition}</p>
          <p class="text-xl font-semibold">${day.temp}°C</p>
        `;
        forecastContainer.appendChild(card);
      });
    } else {
      console.error('Invalid weather data structure:', data);
      alert('Failed to load weather data. Please try again later.');
    }
  } catch (error) {
    console.error('Error fetching weather data:', error);
    const forecastContainer = document.getElementById('forecastContainer');
    forecastContainer.innerHTML = '<p class="text-red-500">Failed to load weather data.</p>';
  }
}

// Function to fetch and update weather data
async function updateWeatherData() {
  try {
    const response = await fetch(`${BASE_URL}/api/fetch-weather`);
    if (!response.ok) throw new Error('Failed to update weather data');
    const result = await response.json();
    console.log('Weather data updated:', result);
    fetchWeatherData(); // Refresh the weather data on the frontend
  } catch (error) {
    console.error('Error updating weather data:', error);
    alert('Failed to update weather data. Please try again.');
  }
}

// Add a button to trigger weather data update
document.addEventListener('DOMContentLoaded', () => {
  const updateButton = document.getElementById('updateWeatherButton');
  if (updateButton) {
    updateButton.addEventListener('click', updateWeatherData);
  }
});

// Add a fetchWeather function to handle button click
function fetchWeather() {
  fetchWeatherData();
}

// Call fetchWeatherData when the Weather Section is shown
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('weatherSection').addEventListener('show', fetchWeatherData);
});

// Fetch equipment data from the backend
async function fetchEquipmentData() {
  try {
    const data = await fetchJSON(`${BASE_URL}/api/equipment`);

    // Safely access properties
    const overview = data.overview || {};
    document.getElementById('totalEquipment').textContent = overview.total || 'N/A';
    document.getElementById('operationalEquipment').textContent = overview.operational || 'N/A';
    document.getElementById('maintenanceEquipment').textContent = overview.underMaintenance || 'N/A';
    document.getElementById('idleEquipment').textContent = overview.idle || 'N/A';

    // Update maintenance schedule
    const scheduleTable = document.getElementById('maintenanceSchedule');
    scheduleTable.innerHTML = '';
    data.maintenanceSchedule.forEach((entry) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="py-2 px-4 border-b">${entry.equipment}</td>
        <td class="py-2 px-4 border-b">${entry.nextMaintenance}</td>
        <td class="py-2 px-4 border-b">${entry.status}</td>
      `;
      scheduleTable.appendChild(row);
    });
    // Update maintenance history
    const historyTable = document.getElementById('maintenanceHistory');
    historyTable.innerHTML = '';
    data.maintenanceHistory.forEach((entry) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td class="py-2 px-4 border-b">${entry.equipment}</td>
        <td class="py-2 px-4 border-b">${entry.lastMaintenance}</td>
        <td class="py-2 px-4 border-b">${entry.details}</td>
      `;
      historyTable.appendChild(row);
    });
    // Update recommendations
    const recommendationsList = document.getElementById('equipmentRecommendations');
    recommendationsList.innerHTML = '';
    data.recommendations.forEach((recommendation) => {
      const li = document.createElement('li');
      li.textContent = recommendation;
      recommendationsList.appendChild(li);
    });
  } catch (error) {
    console.error('Error fetching equipment data:', error);
  }
}

let currentTaskPage = 1; // Track the current page of tasks
const tasksPerPage = 5; // Number of tasks to load per page

function fetchTasks(page = 1) {
  fetch(`http://localhost:5500/api/tasks?page=${page}&limit=${tasksPerPage}`)
    .then(response => response.json())
    .then(tasks => {
      const taskTableBody = document.getElementById('taskManagementTableBody');
      if (page === 1) taskTableBody.innerHTML = ''; // Clear table only for the first page

      tasks.forEach(task => {
        const row = `
          <tr>
            <td class="py-2 px-4 border-b">${task.name}</td>
            <td class="py-2 px-4 border-b">${task.assignedTo}</td>
            <td class="py-2 px-4 border-b">${task.dueDate}</td>
            <td class="py-2 px-4 border-b">${task.status}</td>
          </tr>`;
        taskTableBody.innerHTML += row;
      });

      // Show or hide the "Load More" button based on the number of tasks returned
      const loadMoreButton = document.getElementById('loadMoreTasks');
      if (tasks.length < tasksPerPage) {
        loadMoreButton.classList.add('hidden');
      } else {
        loadMoreButton.classList.remove('hidden');
      }
    })
    .catch(err => console.error('Error fetching tasks:', err));
}

function loadMoreTasks() {
  currentTaskPage++;
  fetchTasks(currentTaskPage);
}

// Fetch the first page of tasks on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchTasks();
});

async function updateTaskStatus(taskId, newStatus) {
  try {
    const response = await fetch(`${BASE_URL}/api/tasks/status`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: taskId, status: newStatus }),
    });
    if (!response.ok) {
      throw new Error(`Failed to update task status: ${response.statusText}`);
    }
    console.log('Task status updated successfully:', await response.json());
    fetchTasks(); // Refresh the task list
  } catch (error) {
    console.error('Error updating task status:', error);
  }
}

// Open the Add Task Modal
function openTaskModal() {
  document.getElementById('taskModal').classList.remove('hidden');
}

// Close the Add Task Modal
function closeTaskModal() {
  document.getElementById('taskModal').classList.add('hidden');
}

// Handle Add Task Form Submission
document.getElementById('taskForm').addEventListener('submit', async (event) => {
  event.preventDefault();

  const name = document.getElementById('taskName').value;
  const assignedTo = document.getElementById('assignedTo').value;
  const dueDate = document.getElementById('dueDate').value;

  try {
    const response = await fetch('http://localhost:5500/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, assignedTo, dueDate }),
    });

    if (response.ok) {
      const result = await response.json();
      alert(result.message);

      // Clear the form
      document.getElementById('taskForm').reset();

      // Refresh the task list
      fetchTasks();

      // Close the modal
      closeTaskModal();
    } else {
      const error = await response.json();
      alert(error.error || 'Failed to add task.');
    }
  } catch (err) {
    console.error('Error adding task:', err);
    alert('An error occurred. Please try again.');
  }
});

// Fetch analytics data from the backend
async function fetchAnalyticsData() {
  try {
    // Correct the API endpoint to match the backend route
    const data = await fetchJSON(`${BASE_URL}/api/analytics`);
    renderRevenueTrendsChart(data.revenueTrends);
    renderCropYieldChart(data.cropYield);
    renderEquipmentUsageChart(data.equipmentUsage);
    renderWaterUsageChart(data.waterUsage);
  } catch (error) {
    console.error('Error fetching analytics data:', error);
  }
}

function renderRevenueTrendsChart(data) {
  const ctx = document.getElementById('revenueTrendsChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Revenue',
        data: data.data,
        borderColor: '#3182ce',
        backgroundColor: 'rgba(49, 130, 206, 0.8)',
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Revenue: $${tooltipItem.raw}`,
          },
        },
      },
    },
  });
}

function renderCropYieldChart(data) {
  const ctx = document.getElementById('cropYieldChart').getContext('2d');
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Yield (tons)',
        data: data.data,
        backgroundColor: ['#38a169', '#3182ce', '#805ad5'],
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `${tooltipItem.label}: ${tooltipItem.raw} tons`,
          },
        },
      },
    },
  });
}

function renderEquipmentUsageChart(data) {
  const ctx = document.getElementById('equipmentUsageChart').getContext('2d');
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: data.labels,
      datasets: [{
        data: data.data,
        backgroundColor: ['#38a169', '#3182ce', '#805ad5', '#e53e3e'],
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Usage: ${tooltipItem.raw} hours`,
          },
        },
      },
    },
  });
}

function renderWaterUsageChart(data) {
  const ctx = document.getElementById('waterUsageChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Water Usage (liters)',
        data: data.data,
        borderColor: '#3182ce',
        backgroundColor: 'rgba(49, 130, 206, 0.2)',
        fill: true,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Usage: ${tooltipItem.raw} liters`,
          },
        },
      },
    },
  });
}

function renderProductionOverviewChart(data) {
  const ctx = document.getElementById('productionOverviewChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Production (tons)',
        data: data.data,
        backgroundColor: ['#38a169', '#3182ce', '#805ad5'],
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Production: ${tooltipItem.raw} tons`,
          },
        },
      },
    },
  });
}

function renderMonthlyYieldChart(data) {
  const ctx = document.getElementById('monthlyYieldChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Yield (tons)',
        data: data.data,
        borderColor: '#3182ce',
        backgroundColor: 'rgba(49, 130, 206, 0.2)',
        fill: true,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Yield: ${tooltipItem.raw} tons`,
          },
        },
      },
    },
  });
}

function renderWaterUsageTrendsChart(data) {
  const ctx = document.getElementById('waterUsageTrendsChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Water Usage (liters)',
        data: data.data,
        borderColor: '#805ad5',
        backgroundColor: 'rgba(128, 90, 213, 0.2)',
        fill: true,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Usage: ${tooltipItem.raw} liters`,
          },
        },
      },
    },
  });
}

function renderEquipmentUsageAnalyticsChart(data) {
  const ctx = document.getElementById('equipmentUsageAnalyticsChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Usage Hours',
        data: data.data,
        backgroundColor: ['#38a169', '#3182ce', '#805ad5', '#e53e3e'],
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        tooltip: {
          enabled: true,
          callbacks: {
            label: (tooltipItem) => `Usage: ${tooltipItem.raw} hours`,
          },
        },
      },
    },
  });
}

function fetchEquipmentUsageAnalytics() {
  fetch('http://localhost:5500/api/equipment-usage?timestamp=' + new Date().getTime()) // Cache-busting query parameter
    .then(response => response.json())
    .then(data => {
      console.log('Equipment Usage Data:', data); // Log data for debugging
      const ctx = document.getElementById('equipmentUsageAnalyticsChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: data.map(item => item.equipment),
          datasets: [
            {
              label: 'Usage Hours',
              data: data.map(item => item.usageHours),
              backgroundColor: [
                'rgba(49, 130, 206, 0.8)',
                'rgba(56, 161, 105, 0.8)',
                'rgba(229, 62, 62, 0.8)',
                'rgba(255, 193, 7, 0.8)',
                'rgba(102, 16, 242, 0.8)',
                'rgba(23, 162, 184, 0.8)',
                'rgba(40, 167, 69, 0.8)',
                'rgba(220, 53, 69, 0.8)',
              ],
              borderColor: [
                'rgba(49, 130, 206, 1)',
                'rgba(56, 161, 105, 1)',
                'rgba(229, 62, 62, 1)',
                'rgba(255, 193, 7, 1)',
                'rgba(102, 16, 242, 1)',
                'rgba(23, 162, 184, 1)',
                'rgba(40, 167, 69, 1)',
                'rgba(220, 53, 69, 1)',
              ],
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true, // Enable responsiveness
          maintainAspectRatio: true, // Maintain aspect ratio
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Equipment',
              },
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Usage Hours',
              },
            },
          },
        },
      });
    })
    .catch(err => console.error('Error fetching equipment usage analytics:', err));
}

// Fetch and render Production Overview chart
function fetchProductionOverview() {
  fetch('http://localhost:5500/api/production-overview')
    .then(response => response.json())
    .then(data => {
      const ctx = document.getElementById('productionOverviewChart').getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.monthlyProduction.map(item => item.month),
          datasets: [{
            label: 'Production',
            data: data.monthlyProduction.map(item => item.production),
            borderColor: '#38a169',
            backgroundColor: 'rgba(56, 161, 105, 0.2)',
            fill: true,
          }],
        },
      });
    });
}

// Fetch and render Vegetable Harvest Summary chart
function fetchVegetableHarvestSummary() {
  fetch('http://localhost:5500/api/vegetable-harvest')
    .then(response => response.json())
    .then(data => {
      const ctx = document.getElementById('vegetableHarvestChart').getContext('2d');
      new Chart(ctx, {
        type: 'pie',
        data: {
          labels: ['Wasted', 'Planted', 'Collected'],
          datasets: [{
            data: [data.harvestSummary.wasted, data.harvestSummary.planted, data.harvestSummary.collected],
            backgroundColor: ['#e53e3e', '#3182ce', '#38a169'],
          }],
        },
      });
    });
}

// Fetch and render Crop Harvest Summary chart
function fetchCropHarvestSummary() {
  fetch('http://localhost:5500/api/crop-harvest')
    .then(response => response.json())

    .then(data => {
      const ctx = document.getElementById('cropHarvestChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: data.harvestSummary.map(item => item.crop),
          datasets: [{
            label: 'Harvested',
            data: data.harvestSummary.map(item => item.harvested),
            backgroundColor: '#3182ce',
          }],
        },
      });
    });
}

// Fetch and render Field Metrics Overview chart
function fetchFieldMetricsOverview() {
  fetch('http://localhost:5500/api/field-metrics')
    .then(response => response.json())
    .then(data => {
      const ctx = document.getElementById('fieldMetricsChart').getContext('2d');
      new Chart(ctx, {
        type: 'radar',
        data: {
          labels: ['PH Level', 'Moisture', 'Nutrients', 'Temperature'],
          datasets: [{
            label: 'Field Metrics',
            data: [data.metrics.phLevel, data.metrics.moisture, data.metrics.nutrients, data.metrics.temperature],
            backgroundColor: 'rgba(56, 161, 105, 0.2)',
            borderColor: '#38a169',
          }],
        },
      });
    });
}

// Fetch and render Equipment Usage chart
function fetchEquipmentUsage() {
  fetch(`${BASE_URL}/api/equipment-usage?timestamp=` + new Date().getTime()) // Cache-busting query parameter
    .then(response => response.json())
    .then(data => {
      const ctx = document.getElementById('equipmentUsageChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar', // Match the chart type with Equipment Usage Analytics
        data: {
          labels: data.map(item => item.equipment),
          datasets: [
            {
              label: 'Usage Hours',
              data: data.map(item => item.usageHours),
              backgroundColor: [
                'rgba(49, 130, 206, 0.8)',
                'rgba(56, 161, 105, 0.8)',
                'rgba(229, 62, 62, 0.8)',
                'rgba(255, 193, 7, 0.8)',
                'rgba(102, 16, 242, 0.8)',
                'rgba(23, 162, 184, 0.8)',
                'rgba(40, 167, 69, 0.8)',
                'rgba(220, 53, 69, 0.8)',
              ],
              borderColor: [
                'rgba(49, 130, 206, 1)',
                'rgba(56, 161, 105, 1)',
                'rgba(229, 62, 62, 1)',
                'rgba(255, 193, 7, 1)',
                'rgba(102, 16, 242, 1)',
                'rgba(23, 162, 184, 1)',
                'rgba(40, 167, 69, 1)',
                'rgba(220, 53, 69, 1)',
              ],
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Equipment',
              },
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Usage Hours',
              },
            },
          },
        },
      });
    })
    .catch(err => console.error('Error fetching equipment usage:', err));
}

// Fetch all charts on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchProductionOverview();
  fetchVegetableHarvestSummary();
  fetchCropHarvestSummary();
  fetchFieldMetricsOverview();
  fetchEquipmentUsage();
  fetchEquipmentUsageAnalytics();
});

async function fetchAndRenderCharts() {
  try {
    // Correct the API endpoint to match the backend route
    const data = await fetchJSON(`${BASE_URL}/api/charts`);
    renderProductionOverviewChart(data.productionOverview);
    renderMonthlyYieldChart(data.monthlyYield);
    renderWaterUsageTrendsChart(data.waterUsage);
    renderEquipmentUsageAnalyticsChart(data.equipmentUsage);
  } catch (error) {
    console.error('Error fetching chart data:', error);
  }
}

function downloadReport() {
  const reportData = {
    title: 'AgriNova Analytics Report',
    date: new Date().toLocaleDateString(),
    sections: [
      { title: 'Revenue Trends', description: 'Revenue trends over the past months.' },
      { title: 'Crop Yield Analysis', description: 'Yield analysis for different crops.' },
      { title: 'Equipment Usage', description: 'Usage distribution of equipment.' },
      { title: 'Water Usage Trends', description: 'Water usage trends for irrigation.' },
    ],
  };
  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'AgriNova_Report.json';
  a.click();
  URL.revokeObjectURL(url);
}

function compareMetrics() {
  const metric = document.getElementById('comparisonMetric').value;
  const timePeriod = document.getElementById('comparisonTimePeriod').value;
  let result = `Comparison for ${metric} during ${timePeriod}:<br>`;
  if (metric === 'revenue') {
    result += 'Revenue increased by 12% compared to the previous period.';
  } else if (metric === 'cropYield') {
    result += 'Crop yield improved by 8% for corn and 5% for wheat.';
  } else if (metric === 'equipmentUsage') {
    result += 'Equipment usage efficiency increased by 10%.';
  } else if (metric === 'waterUsage') {
    result += 'Water usage decreased by 15% due to optimized irrigation.';
  }
  document.getElementById('comparisonResult').innerHTML = result;
}

function savePreferences() {
  const theme = document.getElementById('themeSelector').value;
  const language = document.getElementById('languageSelector').value;
  localStorage.setItem('theme', theme);
  localStorage.setItem('language', language);
  alert('Preferences saved successfully!');
}

function saveNotificationSettings() {
  const emailNotifications = document.getElementById('emailNotifications').checked;
  const smsNotifications = document.getElementById('smsNotifications').checked;
  const pushNotifications = document.getElementById('pushNotifications').checked;
  localStorage.setItem('emailNotifications', emailNotifications);
  localStorage.setItem('smsNotifications', smsNotifications);
  localStorage.setItem('pushNotifications', pushNotifications);
  alert('Notification settings saved successfully!');
}

function saveSystemSettings() {
  const refreshInterval = document.getElementById('refreshInterval').value;
  if (refreshInterval < 1) {
    alert('Refresh interval must be at least 1 minute.');
    return;
  }
  localStorage.setItem('refreshInterval', refreshInterval);
  alert('System settings saved successfully!');
}

function logout() {
  // Simulate logout by clearing local storage and redirecting to a login page
  localStorage.clear();
  alert('You have been logged out.');
  window.location.href = 'Login/login.html'; // Replace with the actual login page URL
}

// Close the profile menu when clicking outside
document.addEventListener('click', (event) => {
  const profileMenu = document.getElementById('profileMenu');
  const profileIcon = event.target.closest('.flex.items-center.cursor-pointer');
  if (!profileIcon && profileMenu && profileMenu.contains(event.target)) {
    profileMenu.classList.add('hidden');
  }
});

// Toggle dark theme
function toggleDarkTheme() {
  document.body.classList.toggle('dark-theme');
  const isDarkTheme = document.body.classList.contains('dark-theme');
  localStorage.setItem('darkTheme', isDarkTheme);
}

// Initialize theme and data fetching
document.addEventListener('DOMContentLoaded', () => {
  fetchWeatherData();
  fetchEquipmentData();
  fetchTasks();
  fetchAnalyticsData();
  showSection('dashboardSection'); // Show the default section
  // Load preferences
  const theme = localStorage.getItem('theme') || 'light';
  const language = localStorage.getItem('language') || 'en';
  document.getElementById('themeSelector').value = theme;
  document.getElementById('languageSelector').value = language;
  // Load notification settings
  document.getElementById('emailNotifications').checked = localStorage.getItem('emailNotifications') === 'true';
  document.getElementById('smsNotifications').checked = localStorage.getItem('smsNotifications') === 'true';
  document.getElementById('pushNotifications').checked = localStorage.getItem('pushNotifications') === 'true';
  // Load system settings
  const refreshInterval = localStorage.getItem('refreshInterval') || 5;
  document.getElementById('refreshInterval').value = refreshInterval;
  // Load dark theme preference
  const isDarkTheme = localStorage.getItem('darkTheme') === 'true';
  if (isDarkTheme) {
    document.body.classList.add('dark-theme');
  }
  // Fetch initial tasks
  fetchTasks();
  fetchAndRenderCharts();
});

async function getBotResponse(message) {
  const lowerCaseMessage = message.toLowerCase();
  try {
    const intentsFiles = [
      'intents3.json', 'intents9.json', 'intents2.json', 'intents4.json',
      'intents5.json', 'intents6.json', 'intents7.json', 'intents8.json',
      'intents10.json', 'intents11.json', 'intents12.json', 'intents13.json',
      'intents14.json', 'intents15.json', 'intents16.json'
    ];
    let allIntents = [];

    for (const file of intentsFiles) {
      const response = await fetch(`${BASE_URL}/intentsbruh/${file}`);
      if (!response.ok) {
        console.error(`Failed to load intents from ${file}: ${response.statusText}`);
        continue;
      }
      const data = await response.json();
      allIntents = allIntents.concat(data.intents);
    }

    for (const intent of allIntents) {
      for (const pattern of intent.patterns) {
        const regex = new RegExp(pattern, 'i');
        if (regex.test(lowerCaseMessage)) {
          return intent.responses[Math.floor(Math.random() * intent.responses.length)];
        }
      }
    }
  } catch (error) {
    console.error('Error fetching intents:', error);
    return "I'm sorry, I encountered an issue. Please try again later.";
  }

  return "I'm sorry, I didn't understand that. Can you please rephrase?";
}

function sendMessage() {
  const chatInput = document.getElementById('chatInput');
  const chatWindow = document.getElementById('chatWindow');
  const typingIndicator = document.getElementById('typingIndicator');
  const userMessage = chatInput.value.trim();

  if (!userMessage) return;

  // Display user message
  const userMessageElement = document.createElement('div');
  userMessageElement.className = 'text-right mb-2 flex items-center justify-end';
  userMessageElement.innerHTML = `
    <span class="bg-green-500 text-white px-3 py-1 rounded-lg inline-block">${userMessage}</span>
    <img src="https://via.placeholder.com/30" alt="User Avatar" class="w-6 h-6 rounded-full ml-2">
  `;
  chatWindow.appendChild(userMessageElement);

  // Clear input
  chatInput.value = '';

  // Show typing indicator
  typingIndicator.classList.remove('hidden');

  // Fetch and display bot response
  getBotResponse(userMessage).then((botResponse) => {
    typingIndicator.classList.add('hidden');
    const botMessageElement = document.createElement('div');
    botMessageElement.className = 'text-left mb-2 flex items-center';
    botMessageElement.innerHTML = `
      <img src="https://via.placeholder.com/30" alt="Bot Avatar" class="w-6 h-6 rounded-full mr-2">
      <span class="bg-gray-300 px-3 py-1 rounded-lg inline-block">
        ${botResponse}
      </span>
    `;
    chatWindow.appendChild(botMessageElement);

    // Scroll to the bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }).catch((error) => {
    console.error('Error handling chatbot response:', error);
    typingIndicator.classList.add('hidden');
    const errorMessageElement = document.createElement('div');
    errorMessageElement.className = 'text-left mb-2 flex items-center';
    errorMessageElement.innerHTML = `
      <img src="https://via.placeholder.com/30" alt="Bot Avatar" class="w-6 h-6 rounded-full mr-2">
      <span class="bg-gray-300 px-3 py-1 rounded-lg inline-block">
        Sorry, I encountered an issue. Please try again later.
      </span>
    `;
    chatWindow.appendChild(errorMessageElement);
  });
}

// Ensure messages array is initialized
let messages = [];

// Function to send a message
function sendMessage() {
  const chatInput = document.getElementById('chatInput');
  const userMessage = chatInput.value.trim();
  if (userMessage === '') return;
  if (event) event.preventDefault();

  console.log(userMessage);

  let msg1 = { name: "🙎‍♂️", message: userMessage };
  messages.push(msg1); // Store user message
  displayMessage(userMessage, "🙎‍♂️"); // Display user message
  chatInput.value = '';

  // onUserMessageReceived(message); // <-- Remove or comment out this line if not defined

  if (shouldFetchFromWelcome) {
    displayNextWelcomeMessage(); // Display welcome message if it's a greeting
  } else {
    // Send message to the server and get chatbot response
    fetch('http://127.0.0.1:5001/predict', {
      method: 'POST',
      body: JSON.stringify({ message: userMessage }),
      headers: {
        'Content-Type': 'application/json'
      },
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      if (data.answer) {
        const lastResponseText = data.answer;
        let msg2 = { name: "🤖", message: lastResponseText };
        messages.push(msg2); // Store chatbot response
        displayMessage(lastResponseText, "🤖"); // Display chatbot response
      } else {
        throw new Error("Unexpected response format");
      }
    })
    .catch((error) => {
      console.error('Error:', error);
      let errorMsg = { name: "🤖", message: "Sorry, there was an error processing your request." };
      messages.push(errorMsg); // Store error message
      displayMessage(errorMsg.message, "🤖"); // Display error message
    })
    .finally(() => {
      // You can add logic to stop any other processes, like stopping voice input
      isVoiceInput = false; // If voice input was used, stop it
    });
  }
}

// Dummy handler for user message reception
function onUserMessageReceived(message) {
  // This is a placeholder. Implement your logic here if needed.
}

// Example Notification on Page Load
document.addEventListener('DOMContentLoaded', () => {
  showNotification('Welcome to AgriNova Dashboard!');
});

// Enhanced Notification Function
function showNotification(message, isError = false) {
  const notification = document.getElementById('notification');
  notification.textContent = message;
  notification.classList.remove('hidden', 'error');
  if (isError) {
    notification.classList.add('error');
  }
  notification.classList.add('show');

  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.classList.add('hidden'), 300);
  }, 3000);
}
document.getElementById('chatInput').addEventListener('keydown', function(event) {
  if (event.key === 'Enter') {
    event.preventDefault(); // Prevent default action (like a form submit)
    sendMessage(); // Trigger sendMessage function
  }
});

// Function to display a message in the chat window
function displayMessage(message, sender) {
  const chatWindow = document.getElementById('chatWindow');
  const messageElement = document.createElement('div');
  messageElement.className = sender === "🤖" ? 'text-left mb-2 flex items-center' : 'text-right mb-2 flex items-center justify-end';
  messageElement.innerHTML = sender === "🤖"
    ? `<img src="https://via.placeholder.com/30" alt="Bot Avatar" class="w-6 h-6 rounded-full mr-2">
       <span class="bg-gray-300 px-3 py-1 rounded-lg inline-block">${message}</span>`
    : `<span class="bg-green-500 text-white px-3 py-1 rounded-lg inline-block">${message}</span>
       <img src="https://via.placeholder.com/30" alt="User Avatar" class="w-6 h-6 rounded-full ml-2">`;
  chatWindow.appendChild(messageElement);

  // Scroll to the bottom of the chat window
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function fetch7DayWeather() {
  const apiKey = '0b5b26decc0b7785a1271999c218bb97';
  const city = 'Chicago';
  const url = `https://api.openweathermap.org/data/2.5/forecast/daily?q=${city}&cnt=7&appid=${apiKey}&units=metric`;

  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch 7-day weather data');
    const data = await response.json();

    const forecastContainer = document.getElementById('forecastContainer');
    forecastContainer.innerHTML = ''; // Clear existing content

    data.list.forEach((day, index) => {
      const date = new Date(day.dt * 1000).toLocaleDateString('en-US', { weekday: 'long' });
      const temp = `${day.temp.day}°C`;
      const condition = day.weather[0].description;
      const icon = `https://openweathermap.org/img/wn/${day.weather[0].icon}.png`;

      const card = document.createElement('div');
      card.className = 'card p-4 text-center';
      card.innerHTML = `
        <h2 class="text-lg font-bold">${date}</h2>
        <img src="${icon}" alt="${condition}" class="mx-auto mb-2" />
        <p class="text-gray-500 capitalize">${condition}</p>
        <p class="text-xl font-semibold">${temp}</p>
      `;
      forecastContainer.appendChild(card);
    });
  } catch (error) {
    console.error('Error fetching 7-day weather data:', error);
    const forecastContainer = document.getElementById('forecastContainer');
    forecastContainer.innerHTML = '<p class="text-red-500">Failed to load 7-day forecast.</p>';
  }
}

// Call fetch7DayWeather when the Weather Section is shown
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('weatherSection').addEventListener('show', fetch7DayWeather);
});