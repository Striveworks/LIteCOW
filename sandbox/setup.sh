#/bin/bash

set -e

REPO_URL="git@nacho.striveworks.us:chariot/icow-light.git"

clone_repo() {
  echo ""
  echo "Cloning repo..."
  git clone "$REPO_URL" > /dev/null 2>&1
  cd icow-light
  echo "Repo cloned: ✅ "
}


check_docker_installed() {
  echo ""
  echo "Checking if docker is installed..."
  if [ -x "$(command -v docker)" ]; then
    echo "Docker installed: ✅ "
  else
      echo "Install docker and try again"
      echo "https://docs.docker.com/get-docker/"
  fi
}

run_sandbox() {
  echo ""
  echo "Setting up sandbox container..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    docker build -t sandbox -f sandbox/Dockerfile . > /dev/null 2>&1
    docker run -d --privileged --network host -v /var/run/docker.sock:/var/run/docker.sock sandbox > /dev/null 2>&1
  else
    sudo docker build -t sandbox -f sandbox/Dockerfile . > /dev/null 2>&1
    sudo docker run -d --privileged --network host -v /var/run/docker.sock:/var/run/docker.sock sandbox > /dev/null 2>&1
  fi
  echo "Sandbox setup: ✅ "
}

install_client() {
  echo ""
  echo "Installing client..."
  echo ""
  pushd sandbox > /dev/null 2>&1
  pip install -r requirements.txt
  popd > /dev/null 2>&1
  echo "Client installed: ✅ "
}

intro_message() {
  echo ""
  echo "🎉 Setup complete 🎉"
  echo ""
  echo "Run your first litecow command"
  echo ""
  echo "litecow import-model --source https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx tinyyolov3"


}

pushd /tmp > /dev/null 2>&1
clone_repo
check_docker_installed
run_sandbox
install_client
intro_message
popd > /dev/null 2>&1
