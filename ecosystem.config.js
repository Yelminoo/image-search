// PM2 process definition for running this FastAPI app under PM2 (which is
// primarily built for Node, but manages arbitrary processes fine via
// interpreter: "none" -- it just runs the script+args as-is and handles
// restart/logging/boot-persistence around it).
//
// Adjust the port below to whatever's actually free on the server -- see
// README.md's deployment section for how to check.
module.exports = {
  apps: [{
    name: "vector-search",
    cwd: __dirname,
    script: "venv/bin/uvicorn",
    args: "app.main:app --host 127.0.0.1 --port 8001",
    interpreter: "none",
    autorestart: true,
    watch: false,
    max_restarts: 10,
  }],
};
