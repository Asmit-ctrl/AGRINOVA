const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 5010;

app.use(cors());
app.use(express.json());
app.use('/datasets', express.static(path.join(__dirname, '../datasets')));

const demoDataPath = path.join(__dirname, '../datasets/demoData.json');

// Helper function to load data from the JSON file
function loadDemoData() {
  if (!fs.existsSync(demoDataPath)) {
    throw new Error('Demo data file not found.');
  }
  const data = fs.readFileSync(demoDataPath, 'utf8');
  return JSON.parse(data);
}

// API to fetch weather data
app.get('/api/weather', (req, res) => {
  try {
    const data = loadDemoData();
    res.json(data.weather);
  } catch (err) {
    console.error('Error loading weather data:', err);
    res.status(500).json({ error: 'Failed to load weather data' });
  }
});

// API to fetch tasks
app.get('/api/tasks', (req, res) => {
  try {
    const data = loadDemoData();
    res.json(data.tasks);
  } catch (err) {
    console.error('Error loading tasks data:', err);
    res.status(500).json({ error: 'Failed to load tasks data' });
  }
});

// API to add a new task
app.post('/api/tasks', (req, res) => {
  try {
    const { name, assignedTo, dueDate, status } = req.body;
    const data = loadDemoData();
    const newTask = {
      id: data.tasks.length + 1,
      name,
      assignedTo,
      dueDate,
      status: status || 'Pending',
    };
    data.tasks.push(newTask);
    fs.writeFileSync(demoDataPath, JSON.stringify(data, null, 2));
    res.status(201).json({ message: 'Task added successfully', task: newTask });
  } catch (err) {
    console.error('Error adding task:', err);
    res.status(500).json({ error: 'Failed to add task' });
  }
});

// API to fetch equipment data
app.get('/api/equipment', (req, res) => {
  try {
    const data = loadDemoData();
    res.json(data.equipment);
  } catch (err) {
    console.error('Error loading equipment data:', err);
    res.status(500).json({ error: 'Failed to load equipment data' });
  }
});

// API to fetch pest management data
app.get('/api/pest-management', (req, res) => {
  const pestManagementData = {
    activePests: 5,
    highRiskFields: 2,
    lastInspection: '2024-09-10',
    controlStrategies: [
      'Use organic pesticides for Field A.',
      'Install pheromone traps in Field B.',
    ],
    activityTrends: {
      labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
      data: [10, 15, 8, 12, 20],
    },
  };
  res.json(pestManagementData);
});

// API to fetch luxury pest management data
app.get('/api/luxury-pest-management', (req, res) => {
  const pestData = {
    crops: ['Corn', 'Wheat', 'Rice'],
    pests: {
      Corn: ['Armyworm', 'Cutworm'],
      Wheat: ['Rust', 'Aphids'],
      Rice: ['Stem Borer', 'Leaf Folder'],
    },
    severityLevels: ['Low', 'Medium', 'High'],
    recommendations: {
      Corn: {
        Low: 'Use neem oil spray.',
        Medium: 'Apply Bacillus thuringiensis (Bt).',
        High: 'Use synthetic pesticides like Lambda-cyhalothrin.',
      },
      Wheat: {
        Low: 'Use sulfur-based fungicides.',
        Medium: 'Apply copper oxychloride.',
        High: 'Use systemic fungicides like Propiconazole.',
      },
      Rice: {
        Low: 'Spray potassium silicate.',
        Medium: 'Apply carbendazim.',
        High: 'Use chlorpyrifos for severe infestations.',
      },
    },
  };
  res.json(pestData);
});

// Example of a valid route definition
app.get('/api/resource/:id', (req, res) => {
  const { id } = req.params;
  res.json({ message: `Resource ID: ${id}` });
});

// Start the server
app.listen(port, () => console.log(`New server running on http://localhost:${port}`));
