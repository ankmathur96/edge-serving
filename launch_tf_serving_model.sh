docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/home/ankitmathur/edge-serving/partial_inception_v1,target=/models/partial_inception_v1 -e MODEL_NAME=partial_inception_v1 -t tensorflow/serving:latest-gpu &
