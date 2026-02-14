FROM python:3.12-slim

WORKDIR /app

# Install ds_tools first (local dependency)
COPY ds_tools/ ds_tools/
RUN pip install --no-cache-dir -e ds_tools/

# Install fraud demo dependencies
RUN pip install --no-cache-dir lightgbm>=4.0 matplotlib>=3.7

# Copy fraud demo code
COPY ["fraud_detection/demo/", "fraud_detection/demo/"]

# Run the demo
CMD ["bash", "fraud_detection/demo/run_demo.sh"]
