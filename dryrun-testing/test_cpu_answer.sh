SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run --runtime=nvidia -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
docker exec -i $NAME ./test_cpu_answer.py /host/Users/a7.txt /host/Users/a7_questions.txt
echo '****************'
docker stop $NAME >/dev/null