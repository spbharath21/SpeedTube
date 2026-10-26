# SpeedTube frontend

React interface for loading a YouTube video, asking transcript-grounded questions,
and generating summaries.

## Development

Start the API on port 8000, then run:

```bash
npm ci
npm start
```

The app opens at <http://localhost:3000>. To use a backend at another address,
set `REACT_APP_API_BASE_URL` before running `npm start` or `npm run build`.

## Checks

```bash
npm test -- --watchAll=false
npm run build
```
