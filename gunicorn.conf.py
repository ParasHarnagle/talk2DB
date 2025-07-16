import multiprocessing
cpu_count = multiprocessing.cpu_count()
workers = 2 * cpu_count + 1

# ------------------------
# Server Binding
# ------------------------
bind = "172.31.16.207:9000"  # Use internal IP for Docker compatibility

# ------------------------
# Worker Settings
# ------------------------
worker_class = "uvicorn.workers.UvicornWorker"  # Required for async FastAPI
timeout = 900                    # Worker timeout in seconds
graceful_timeout = 60            # Graceful shutdown timeout
keepalive = 5                    # TCP keep-alive

# ------------------------
# Logging
# ------------------------
accesslog = "-"                 # stdout
errorlog = "-"                  # stderr
loglevel = "info"
