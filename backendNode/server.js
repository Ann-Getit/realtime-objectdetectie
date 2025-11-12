// server.js
const express = require('express');
const cors = require('cors'); // cors is een browserbeveiliging
const app = express();
const port = process.env.PORT || 3000;
//const path = require('path');


// Voor JSON body (base64 frame)
app.use(express.json({ limit: '10mb' }));


app.use(cors({ 
    origin: ['https://ann-getit.github.io'], //frontend pages url moet hier 
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type','Authorization']
}));






//app.use(express.static(path.join(__dirname, 'public'))); alleen voor lokaal gebruik




app.post('/detect', async (req, res) => {
    try {
        const imageBase64 = req.body.image;  // base64 string
        if (!imageBase64) return res.status(400).json({ error: 'No image provided' });
        
 
        // Stuur frame naar Flask server
        const flaskResponse = await fetch('https://backend-realtime-ai.onrender.com/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageBase64 })
        });

        const text = await flaskResponse.text(); // <-- lees de tekst terug van Flask

        if (!flaskResponse.ok) {
            return res.status(500).json({ error: 'Flask server error', details: text });
        }

          // ✅ geef de Flask-response terug aan de frontend
          res.type('application/json').send(text);

    } catch (err) {
        console.error('Server error:', err);
        res.status(500).json({ error: 'Server error', details: err.message });
    }
});

//app.get('/', (req, res) => {
  //res.sendFile(path.join(__dirname, 'public', 'index.html'));
//});

app.get('/', (req, res) => {
  res.send('Hello world!');
});




app.listen(port, '0.0.0.0', () => console.log(`✅ Node.js server listening on ${port}🚀`));

