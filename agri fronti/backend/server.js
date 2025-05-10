const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const axios = require('axios'); // Add axios for external API requests
const app = express();
const port = 5500;

app.use(cors());
app.use(express.json());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// Serve chart images
app.use('/charts', express.static(path.join(__dirname, '../charts')));

// Serve datasets statically
app.use('/datasets', express.static(path.join(__dirname, '../datasets')));

// Paths to JSON files
const paths = {
  weather: path.join(__dirname, '../datasets/weather.json'),
  equipment: path.join(__dirname, '../datasets/equipment.json'),
  tasks: path.join(__dirname, '../datasets/tasks.json'),
  dashboard: path.join(__dirname, '../datasets/dashboard.json'),
  cropManagement: path.join(__dirname, '../datasets/cropManagement.json'),
  soilWater: path.join(__dirname, '../datasets/soilWater.json'),
  reports: path.join(__dirname, '../datasets/reports.json'),
  productionOverview: path.join(__dirname, '../datasets/productionOverview.json'),
  vegetableHarvest: path.join(__dirname, '../datasets/vegetableHarvest.json'),
  cropHarvest: path.join(__dirname, '../datasets/cropHarvest.json'),
  fieldMetrics: path.join(__dirname, '../datasets/fieldMetrics.json'),
  equipmentUsage: path.join(__dirname, '../datasets/equipmentUsage.json'),
};

// Helper function to load JSON data
function loadJSON(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  const data = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(data);
}

// Helper function to save JSON data
function saveJSON(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

// API to fetch and save weather data
app.get('/api/fetch-weather', async (req, res) => {
  const apiKey = '0b5b26decc0b7785a1271999c218bb97'; // Replace with your OpenWeatherMap API key
  const city = 'Chicago'; // Replace with the desired city
  const url = `https://api.openweathermap.org/data/2.5/forecast?q=${city}&appid=${apiKey}&units=metric`;

  try {
    const response = await axios.get(url);
    const weatherData = {
      city: response.data.city.name,
      forecast: response.data.list.map(item => ({
        date: item.dt_txt,
        temp: item.main.temp,
        condition: item.weather[0].description,
        icon: `https://openweathermap.org/img/wn/${item.weather[0].icon}.png`,
      })),
    };

    // Save the weather data to a JSON file
    saveJSON(paths.weather, weatherData);

    res.json({ message: 'Weather data fetched and saved successfully.', data: weatherData });
  } catch (error) {
    console.error('Error fetching weather data:', error);
    res.status(500).json({ error: 'Failed to fetch weather data.' });
  }
});

// API to serve saved weather data
app.get('/api/weather', (req, res) => {
  try {
    const weatherData = loadJSON(paths.weather);
    if (!weatherData || !weatherData.city || !weatherData.forecast) {
      throw new Error('Weather data is missing or invalid');
    }

    const formattedData = {
      city: weatherData.city || 'Unknown City',
      forecast: weatherData.forecast.map(day => ({
        date: day.date,
        temp: day.temp,
        condition: day.condition,
        feels_like: day.feels_like,
        icon: day.icon || 'https://via.placeholder.com/50', // Default icon if not available
      })),
    };
    res.json(formattedData);
  } catch (error) {
    console.error('Error fetching weather data:', error);
    res.status(500).json({ error: 'Failed to fetch weather data' });
  }
});

// API to fetch equipment data
app.get('/api/equipment', (req, res) => {
  try {
    const equipmentData = loadJSON(paths.equipment);
    res.json(equipmentData);
  } catch (err) {
    console.error('Error loading equipment data:', err);
    res.status(500).json({ error: 'Failed to load equipment data' });
  }
});

// API to fetch tasks with pagination
app.get('/api/tasks', (req, res) => {
  try {
    const tasks = loadJSON(paths.tasks);
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || tasks.length;

    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;

    const paginatedTasks = tasks.slice(startIndex, endIndex);
    res.json(paginatedTasks);
  } catch (err) {
    console.error('Error loading tasks data:', err);
    res.status(500).json({ error: 'Failed to load tasks data' });
  }
});

// API to add a new task
app.post('/api/tasks', (req, res) => {
  try {
    const { name, assignedTo, dueDate, status } = req.body;

    // Validate input
    if (!name || !assignedTo || !dueDate) {
      return res.status(400).json({ error: 'Name, assignedTo, and dueDate are required.' });
    }

    const tasks = loadJSON(paths.tasks);
    const newTask = {
      id: tasks.length + 1,
      name,
      assignedTo,
      dueDate,
      status: status || 'Pending',
    };

    tasks.push(newTask);
    saveJSON(paths.tasks, tasks);

    res.status(201).json({ message: 'Task added successfully', task: newTask });
  } catch (err) {
    console.error('Error adding task:', err);
    res.status(500).json({ error: 'Failed to add task' });
  }
});

// API to fetch dashboard data
app.get('/api/dashboard', (req, res) => {
  try {
    const dashboardData = loadJSON(paths.dashboard);
    res.json(dashboardData);
  } catch (err) {
    console.error('Error loading dashboard data:', err);
    res.status(500).json({ error: 'Failed to load dashboard data' });
  }
});

// API to fetch crop management data
app.get('/api/crop-management', (req, res) => {
  try {
    const cropManagementData = loadJSON(paths.cropManagement);
    res.json(cropManagementData);
  } catch (err) {
    console.error('Error loading crop management data:', err);
    res.status(500).json({ error: 'Failed to load crop management data' });
  }
});

// API to fetch soil and water data
app.get('/api/soil-water', (req, res) => {
  try {
    const soilWaterData = loadJSON(paths.soilWater);
    res.json(soilWaterData);
  } catch (err) {
    console.error('Error loading soil and water data:', err);
    res.status(500).json({ error: 'Failed to load soil and water data' });
  }
});

// API to fetch reports and analytics data
app.get('/api/reports', (req, res) => {
  try {
    const reportsData = loadJSON(paths.reports);
    if (!reportsData.revenueTrends) {
      throw new Error('Missing revenueTrends data in reports.json');
    }
    res.json(reportsData); // Ensure this includes `revenueTrends`
  } catch (err) {
    console.error('Error loading reports data:', err);
    res.status(500).json({ error: 'Failed to load reports data' });
  }
});

// API to fetch Production Overview data
app.get('/api/production-overview', (req, res) => {
  try {
    const data = loadJSON(paths.productionOverview);
    res.json(data);
  } catch (err) {
    console.error('Error loading production overview data:', err);
    res.status(500).json({ error: 'Failed to load production overview data' });
  }
});

// API to fetch Vegetable Harvest Summary data
app.get('/api/vegetable-harvest', (req, res) => {
  try {
    const data = loadJSON(paths.vegetableHarvest);
    res.json(data);
  } catch (err) {
    console.error('Error loading vegetable harvest data:', err);
    res.status(500).json({ error: 'Failed to load vegetable harvest data' });
  }
});

// API to fetch Crop Harvest Summary data
app.get('/api/crop-harvest', (req, res) => {
  try {
    const data = loadJSON(paths.cropHarvest);
    res.json(data);
  } catch (err) {
    console.error('Error loading crop harvest data:', err);
    res.status(500).json({ error: 'Failed to load crop harvest data' });
  }
});

// API to fetch Field Metrics Overview data
app.get('/api/field-metrics', (req, res) => {
  try {
    const data = loadJSON(paths.fieldMetrics);
    res.json(data);
  } catch (err) {
    console.error('Error loading field metrics data:', err);
    res.status(500).json({ error: 'Failed to load field metrics data' });
  }
});

// API to fetch Equipment Usage data
app.get('/api/equipment-usage', (req, res) => {
  try {
    const data = loadJSON(paths.equipmentUsage);
    res.json(data.equipmentUsage); // Ensure the correct property is returned
  } catch (err) {
    console.error('Error loading equipment usage data:', err);
    res.status(500).json({ error: 'Failed to load equipment usage data' });
  }
});

// Save user data route
app.post('/api/save-user-data', (req, res) => {
  const userData = req.body;
  const filePath = path.join(__dirname, '../datasets/userData.json');

  try {
    let existingData = [];
    if (fs.existsSync(filePath)) {
      existingData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }

    existingData.push(userData);
    fs.writeFileSync(filePath, JSON.stringify(existingData, null, 2));

    res.status(200).json({ message: 'User data saved successfully!' });
  } catch (error) {
    console.error('Error saving user data:', error);
    res.status(500).json({ message: 'Failed to save user data.' });
  }
});

// Login route
app.post('/api/login', (req, res) => {
  const { username, password } = req.body;

  const usersFilePath = path.join(__dirname, '../datasets/users.json');
  if (!fs.existsSync(usersFilePath)) {
    return res.status(500).json({ message: 'Users file not found.' });
  }

  const users = JSON.parse(fs.readFileSync(usersFilePath, 'utf8'));
  const user = users.find(u => u.username === username && u.password === password);

  if (user) {
    res.status(200).json({ message: 'Login successful!' });
  } else {
    res.status(401).json({ message: 'Invalid username or password.' });
  }
});

// Register route
app.post('/api/register', (req, res) => {
  const { username, email, password } = req.body;

  const usersFilePath = path.join(__dirname, '../datasets/users.json');
  try {
    let users = [];
    if (fs.existsSync(usersFilePath)) {
      users = JSON.parse(fs.readFileSync(usersFilePath, 'utf8'));
    }

    if (users.some(u => u.username === username)) {
      return res.status(400).json({ message: 'Username already exists.' });
    }

    users.push({ username, email, password });
    fs.writeFileSync(usersFilePath, JSON.stringify(users, null, 2));

    res.status(201).json({ message: 'Registration successful!' });
  } catch (error) {
    console.error('Error registering user:', error);
    res.status(500).json({ message: 'Failed to register user.' });
  }
});

// Serve the main.html file
app.get('/frontend/main.html', (req, res) => {
  const mainFilePath = path.join(__dirname, '../frontend/main.html');
  if (fs.existsSync(mainFilePath)) {
    res.sendFile(mainFilePath);
  } else {
    res.status(404).send('Main file not found.');
  }
});

// Example of a valid route definition
app.get('/api/resource/:id', (req, res) => {
  const { id } = req.params;
  res.json({ message: `Resource ID: ${id}` });
});

// Start the server
app.listen(port, () => console.log(`Server running on http://localhost:${port}`));