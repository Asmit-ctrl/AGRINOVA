const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const { Chart, registerables } = require('chart.js');
const fs = require('fs');
const path = require('path');

// Register all necessary Chart.js components
Chart.register(...registerables);

// Paths to JSON files
const dashboardDataPath = path.join(__dirname, '../datasets/dashboard.json');
const reportsDataPath = path.join(__dirname, '../datasets/reports.json');
const productionOverviewPath = path.join(__dirname, '../datasets/productionOverview.json');
const vegetableHarvestPath = path.join(__dirname, '../datasets/vegetableHarvest.json');
const cropHarvestPath = path.join(__dirname, '../datasets/cropHarvest.json');
const fieldMetricsPath = path.join(__dirname, '../datasets/fieldMetrics.json');
const equipmentUsagePath = path.join(__dirname, '../datasets/equipmentUsage.json');
const revenueDataPath = path.join(__dirname, '../datasets/revenueData.json');

// Chart output folder
const outputFolder = path.join(__dirname, '../charts');

// Ensure the output folder exists
if (!fs.existsSync(outputFolder)) {
  fs.mkdirSync(outputFolder, { recursive: true });
}

// Chart configuration
const width = 800; // Chart width
const height = 600; // Chart height
const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height });

// Helper function to load JSON data
function loadJSON(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  const data = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(data);
}

// Generate and save a chart
async function generateChart(chartConfig, filename) {
  const imageBuffer = await chartJSNodeCanvas.renderToBuffer(chartConfig);
  const filePath = path.join(outputFolder, filename);
  fs.writeFileSync(filePath, imageBuffer);
  console.log(`Chart saved: ${filePath}`);
}

// Generate charts based on JSON data
async function generateCharts() {
  try {
    const dashboardData = loadJSON(dashboardDataPath);
    const reportsData = loadJSON(reportsDataPath);
    const productionOverviewData = loadJSON(productionOverviewPath);
    const vegetableHarvestData = loadJSON(vegetableHarvestPath);
    const cropHarvestData = loadJSON(cropHarvestPath);
    const fieldMetricsData = loadJSON(fieldMetricsPath);
    const equipmentUsageData = loadJSON(equipmentUsagePath);

    // Monthly Yield Chart
    await generateChart(
      {
        type: 'line',
        data: {
          labels: dashboardData.monthlyYield.map(item => item.month),
          datasets: [
            {
              label: 'Monthly Yield',
              data: dashboardData.monthlyYield.map(item => item.yield),
              borderColor: '#38a169',
              backgroundColor: 'rgba(56, 161, 105, 0.2)',
              fill: true,
            },
          ],
        },
      },
      'monthlyYieldChart.png'
    );

    // Revenue Trends Chart
    await generateChart(
      {
        type: 'bar',
        data: {
          labels: reportsData.revenueTrends.map(item => item.month),
          datasets: [
            {
              label: 'Revenue',
              data: reportsData.revenueTrends.map(item => item.revenue),
              backgroundColor: '#3182ce',
            },
          ],
        },
      },
      'revenueTrendsChart.png'
    );

    // Production Overview Chart
    await generateChart(
      {
        type: 'line',
        data: {
          labels: productionOverviewData.monthlyProduction.map(item => item.month),
          datasets: [
            {
              label: 'Production',
              data: productionOverviewData.monthlyProduction.map(item => item.production),
              borderColor: '#3182ce',
              backgroundColor: 'rgba(49, 130, 206, 0.2)',
              fill: true,
            },
          ],
        },
      },
      'productionOverviewChart.png'
    );

    // Vegetable Harvest Summary Chart
    await generateChart(
      {
        type: 'pie',
        data: {
          labels: ['Wasted', 'Planted', 'Collected'],
          datasets: [
            {
              data: [
                vegetableHarvestData.harvestSummary.wasted,
                vegetableHarvestData.harvestSummary.planted,
                vegetableHarvestData.harvestSummary.collected,
              ],
              backgroundColor: ['#e53e3e', '#3182ce', '#38a169'],
            },
          ],
        },
      },
      'vegetableHarvestChart.png'
    );

    // Crop Harvest Summary Chart
    await generateChart(
      {
        type: 'bar',
        data: {
          labels: cropHarvestData.harvestSummary.map(item => item.crop),
          datasets: [
            {
              label: 'Harvested',
              data: cropHarvestData.harvestSummary.map(item => item.harvested),
              backgroundColor: '#3182ce',
            },
          ],
        },
      },
      'cropHarvestChart.png'
    );

    // Field Metrics Overview Chart
    await generateChart(
      {
        type: 'radar',
        data: {
          labels: ['PH Level', 'Moisture', 'Nutrients', 'Temperature'],
          datasets: [
            {
              label: 'Field Metrics',
              data: [
                fieldMetricsData.metrics.phLevel,
                fieldMetricsData.metrics.moisture,
                fieldMetricsData.metrics.nutrients,
                fieldMetricsData.metrics.temperature,
              ],
              backgroundColor: 'rgba(56, 161, 105, 0.2)',
              borderColor: '#38a169',
            },
          ],
        },
      },
      'fieldMetricsChart.png'
    );

    // Equipment Usage Analytics Chart with Fixed Size
    const chartJSNodeCanvas = new ChartJSNodeCanvas({ width: 400, height: 200 });
    const configuration = {
      type: 'bar',
      data: {
        labels: equipmentUsageData.equipmentUsage.map(item => item.equipment),
        datasets: [
          {
            label: 'Usage Hours',
            data: equipmentUsageData.equipmentUsage.map(item => item.usageHours),
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
        responsive: false,
        maintainAspectRatio: false,
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
    };

    const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration);
    const filePath = path.join(outputFolder, 'equipmentUsageChart.png');
    fs.writeFileSync(filePath, imageBuffer);
    console.log('Equipment Usage Analytics Chart generated successfully.');

    // Revenue Chart
    const revenueData = loadJSON(revenueDataPath);
    await generateChart(
      {
        type: 'bar',
        data: {
          labels: revenueData.months,
          datasets: [
            {
              label: 'Revenue ($)',
              data: revenueData.revenue,
              backgroundColor: 'rgba(75, 192, 192, 0.8)',
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: false,
          maintainAspectRatio: false,
          scales: {
            x: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Months',
              },
            },
            y: {
              beginAtZero: true,
              title: {
                display: true,
                text: 'Revenue ($)',
              },
            },
          },
          plugins: {
            tooltip: {
              enabled: true, // Enable tooltips
              callbacks: {
                label: function (context) {
                  return `Revenue: $${context.raw}`;
                },
              },
            },
          },
        },
      },
      'revenueChart.png'
    );

    console.log('Revenue Chart generated successfully.');

    console.log('All charts generated successfully.');
  } catch (err) {
    console.error('Error generating charts:', err);
  }
}

// Run the chart generation
generateCharts();
