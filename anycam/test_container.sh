#!/bin/bash

# test_container.sh
# Test what's available in the Docker container and diagnose AnyCam issues

set -e

# Default container name
CONTAINER_NAME="anycam:latest"
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
if [[ $# -gt 0 ]]; then
    CONTAINER_NAME="$1"
fi

echo "========================================"
echo "  Docker Container Environment Test"
echo "========================================"
echo
print_info "Testing container: $CONTAINER_NAME"
print_info "Script directory: $SCRIPT_DIR"
echo

# Check if container exists
if ! docker image inspect "$CONTAINER_NAME" &> /dev/null; then
    print_error "Docker image '$CONTAINER_NAME' not found"
    print_info "Available images:"
    docker images | grep -E "(anycam|pytorch)" || echo "No anycam or pytorch images found"
    exit 1
fi

print_success "Container found: $CONTAINER_NAME"

# Test 1: Basic Python and PyTorch
print_info "Test 1: Basic Python and PyTorch environment"
docker run --rm "$CONTAINER_NAME" python -c "
import sys
import torch
print('Python version:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA devices:', torch.cuda.device_count())
    print('Current device:', torch.cuda.current_device())
print('✓ Basic environment OK')
"

# Test 2: Check required packages
print_info "Test 2: Checking required packages"
docker run --rm "$CONTAINER_NAME" python -c "
packages = ['numpy', 'PIL', 'tqdm', 'cv2', 'scipy', 'skimage', 'matplotlib', 'torch', 'torchvision']
missing = []
for pkg in packages:
    try:
        if pkg == 'PIL':
            import PIL
        elif pkg == 'cv2':
            import cv2
        elif pkg == 'skimage':
            import skimage
        else:
            __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - MISSING')
        missing.append(pkg)

if missing:
    print(f'Missing packages: {missing}')
    exit(1)
else:
    print('✓ All basic packages available')
"

# Test 3: Check AnyCam specific dependencies
print_info "Test 3: Checking AnyCam specific dependencies"
docker run --rm "$CONTAINER_NAME" python -c "
anycam_deps = ['unimatch', 'unimatch.unimatch', 'timm', 'kornia', 'einops']
missing = []
for dep in anycam_deps:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError as e:
        print(f'✗ {dep} - MISSING: {e}')
        missing.append(dep)

if missing:
    print(f'Missing AnyCam dependencies: {missing}')
    print('These need to be installed for AnyCam to work')
else:
    print('✓ All AnyCam dependencies available')
"

# Test 4: Try torch.hub operations
print_info "Test 4: Testing torch.hub operations"
docker run --rm "$CONTAINER_NAME" python -c "
import torch

# Test torch.hub basic functionality
try:
    print('torch.hub cache dir:', torch.hub.get_dir())
    print('✓ torch.hub basic functions OK')
except Exception as e:
    print(f'✗ torch.hub error: {e}')
    exit(1)

# Try to list AnyCam models
try:
    print('Attempting to list AnyCam models...')
    models = torch.hub.list('Brummi/anycam', force_reload=True)
    print(f'Available models: {models}')
    print('✓ torch.hub can access AnyCam repository')
except Exception as e:
    print(f'✗ Cannot access AnyCam repository: {e}')
    print('This might be a network issue or repository problem')
"

# Test 5: Try to load AnyCam model
print_info "Test 5: Attempting to load AnyCam model"
docker run --rm "$CONTAINER_NAME" python -c "
import torch

try:
    print('Loading AnyCam model...')
    anycam = torch.hub.load('Brummi/anycam', 'AnyCam', version='1.0', training_variant='seq8', pretrained=True)
    print('✓ AnyCam model loaded successfully!')
    print(f'Model type: {type(anycam)}')
except Exception as e:
    print(f'✗ Failed to load AnyCam model: {e}')
    print(f'Error type: {type(e).__name__}')
    
    # Try to diagnose the specific error
    import traceback
    print('Full traceback:')
    traceback.print_exc()
    exit(1)
"

# Test 6: Run our test environment script if it exists
if [[ -f "$SCRIPT_DIR/test_container_environment.py" ]]; then
    print_info "Test 6: Running custom environment test script"
    docker run --rm \
        -v "$SCRIPT_DIR:/workspace/scripts:ro" \
        -w /workspace \
        "$CONTAINER_NAME" \
        python /workspace/scripts/test_container_environment.py
fi

echo
print_success "All tests completed!"
print_info "If any tests failed, you may need to:"
print_info "1. Build a proper container with: ./setup_anycam_container.sh"
print_info "2. Or install missing dependencies in your current container"
print_info "3. Or use a different base image"
