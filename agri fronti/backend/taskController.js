const fs = require('fs');
const path = require('path');

const tasksFilePath = path.join(__dirname, '../datasets/tasks.json');

// Load tasks from the JSON file
function loadTasks() {
  if (!fs.existsSync(tasksFilePath)) {
    fs.writeFileSync(tasksFilePath, JSON.stringify([]));
  }
  const data = fs.readFileSync(tasksFilePath, 'utf8');
  return JSON.parse(data);
}

// Save tasks to the JSON file
function saveTasks(tasks) {
  fs.writeFileSync(tasksFilePath, JSON.stringify(tasks, null, 2));
}

// Get all tasks
function getTasks(req, res) {
  const tasks = loadTasks();
  res.json(tasks);
}

// Add a new task
function addTask(req, res) {
  const { name, assignedTo, dueDate, status } = req.body;

  if (!name || !assignedTo || !dueDate || !status) {
    return res.status(400).json({ error: 'All fields are required.' });
  }

  const tasks = loadTasks();
  const newTask = { id: tasks.length + 1, name, assignedTo, dueDate, status };
  tasks.push(newTask);
  saveTasks(tasks);
  res.status(201).json(newTask);
}

// Update task status
function updateTaskStatus(req, res) {
  const { id, status } = req.body;

  if (!id || !status) {
    return res.status(400).json({ error: 'Task ID and status are required.' });
  }

  const tasks = loadTasks();
  const task = tasks.find((task) => task.id === id);

  if (!task) {
    return res.status(404).json({ error: 'Task not found.' });
  }

  task.status = status;
  saveTasks(tasks);
  res.status(200).json(task);
}

module.exports = { getTasks, addTask, updateTaskStatus };
