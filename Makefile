.PHONY: build run stop clean test tensorboard jupyter shell logs

help:
	@echo "VizDoom RL Training Environment"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build the Docker image"
	@echo "  make run            - Start the container"
	@echo "  make stop           - Stop the container"
	@echo "  make shell          - Open a shell in the running container"
	@echo "  make test           - Run environment tests"
	@echo "  make tensorboard    - Start TensorBoard"
	@echo "  make jupyter        - Start Jupyter Lab"
	@echo "  make logs           - View container logs"
	@echo "  make clean          - Remove container and volumes"
	@echo "  make clean-all      - Remove everything including images"

# Build Docker image
build:
	@echo "Building Docker image..."
	docker-compose build

# Start container
run:
	@echo "Starting container..."
	docker-compose up -d
	@echo "Container started. Use 'make shell' to access it."

# Stop container
stop:
	@echo "Stopping container..."
	docker-compose down

# Open shell in container
shell:
	@echo "Opening shell in container..."
	docker-compose exec vizdoom-rl bash

# Run tests
test:
	@echo "Running environment tests..."
	docker-compose exec vizdoom-rl python scripts/test_environment.py

# # Train PPO agent
# train-ppo:
# 	@echo "Training PPO agent..."
# 	docker-compose exec vizdoom-rl python -c "from train_vizdoom import train_ppo_cnn; train_ppo_cnn()"

# # Train DQN agent
# train-dqn:
# 	@echo "Training DQN agent..."
# 	docker-compose exec vizdoom-rl python -c "from train_vizdoom import train_dqn; train_dqn()"

# Start TensorBoard
tensorboard:
	@echo "Starting TensorBoard..."
	@echo "Access at: http://localhost:6006"
	docker-compose exec -d vizdoom-rl tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006

# Start Jupyter Lab
jupyter:
	@echo "Starting Jupyter Lab..."
	@echo "Check logs for the access token"
	docker-compose exec -d vizdoom-rl jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
	@sleep 3
	@echo ""
	@echo "Jupyter Lab started. Access at: http://localhost:8888"
	@echo "Token:"
	@docker-compose exec vizdoom-rl jupyter server list 2>/dev/null | grep -oP 'token=\K[a-f0-9]+'

# View logs
logs:
	docker-compose logs -f

# Clean up (remove container and volumes)
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	@echo "Cleaned up containers and volumes."

# Remove everything including images
clean-all: clean
	@echo "Removing Docker images..."
	docker rmi vizdoom-rl:latest || true
	@echo "Full cleanup complete."