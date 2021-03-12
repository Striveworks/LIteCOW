.PHONY: sandbox
sandbox:
	sudo docker build -t sandbox -f sandbox/Dockerfile .
	sudo docker run -d --privileged --network host -v /var/run/docker.sock:/var/run/docker.sock sandbox

.PHONY: cleanup
cleanup:
	sudo docker run -it --privileged --network host -v /var/run/docker.sock:/var/run/docker.sock sandbox kind delete cluster --name knative

.PHONY: litecow_server
litecow_server:
	sudo docker build -t litecow_server -f docker/server/Dockerfile src

.PHONY: litecow_server_gpu
litecow_server_gpu:
	sudo docker build -t litecow_server:gpu -f docker/gpu-server/Dockerfile src

dev-docs :
	docker run --rm -it --network=host -v ${PWD}:/icow-light --workdir /icow-light --entrypoint bash --name litecow-docs polinux/mkdocs scripts/mkdocs_startup.sh
