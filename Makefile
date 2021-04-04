.DEFAULT_GOAL: build

build:
	docker build -t gcr.io/rowan-senior-project/sauce-train:$(V) .
	# docker push gcr.io/rowan-senior-project/sauce-train:$(V)

build_dev:
	docker build -f Dockerfile.devmachine -t mrthinger/pytorch-dev:$(V) .
	docker push docker.io/mrthinger/pytorch-dev:$(V)