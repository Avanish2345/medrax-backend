const express = require('express');
const cors = require('cors');
const fileUpload = require('express-fileupload');
const diagnosisRoutes = require('./routes/diagnosis');

const app = express();

app.use(cors());
app.use(express.json());
app.use(fileUpload({ limits: { fileSize: 50 * 1024 * 1024 } }));

app.use('/api/diagnosis', diagnosisRoutes);

module.exports = app;
