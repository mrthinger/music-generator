.DEFAULT_GOAL: build

build:
	docker build -t gcr.io/rowan-senior-project/sauce-train:$(V) .
	# docker push gcr.io/rowan-senior-project/sauce-train:$(V)

build_dev:
	docker build -f Dockerfile.devmachine -t gcr.io/rowan-senior-project/pytorch-dev:$(V) .
	docker push gcr.io/rowan-senior-project/pytorch-dev:$(V)