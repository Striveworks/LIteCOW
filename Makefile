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
	docker build -t litecow_server -f docker/server/Dockerfile src

.PHONY: litecow_server_gpu
litecow_server_gpu:
	docker build -t litecow_server:gpu -f docker/gpu-server/Dockerfile src

.PHONY: docs
docs:
	docker build -t docs -f docker/docs/Dockerfile .

.PHONY: publish-docs
publish-docs:
	docker build -t docs -f docker/docs/Dockerfile .
	git checkout gh-pages
	docker run --rm -d -p 8000:80 --name docs docs
	docker cp docs:/usr/local/apache2/htdocs/. .
	docker stop docs
	git commit -am "Update docs"
	git push
	git checkout main
