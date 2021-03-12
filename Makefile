.PHONY: sandbox
sandbox:
	docker build -t sandbox -f sandbox/Dockerfile .
	docker run --name sandbox -d --privileged --network host -v /var/run/docker.sock:/var/run/docker.sock sandbox
	@echo ""
	@echo ""
	@echo "Run the following to check the status: "
	@echo 'docker ps -aqf "name=sandbox" | xargs docker logs -f'
	@echo ""
	@echo "Exec into the sandbox: "
	@echo "docker exec -it sandbox sh"

.PHONY: cleanup
cleanup:
	docker run -it --privileged --network host -v /var/run/docker.sock:/var/run/docker.sock sandbox kind delete cluster --name knative
	docker rm -f sandbox

.PHONY: litecow_server
litecow_server:
	sudo docker build -t litecow_server -f docker/server/Dockerfile src

.PHONY: litecow_server_gpu
litecow_server_gpu:
	sudo docker build -t litecow_server:gpu -f docker/gpu-server/Dockerfile src

dev-docs :
	docker run --rm -it --network=host -v ${PWD}:/icow-light --workdir /icow-light --entrypoint bash --name litecow-docs polinux/mkdocs scripts/mkdocs_startup.sh
