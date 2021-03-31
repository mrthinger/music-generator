.DEFAULT_GOAL: build

build:
	docker build -t gcr.io/rowan-senior-project/sauce-train:$(V) .
	# docker push gcr.io/rowan-senior-project/sauce-train:$(V)