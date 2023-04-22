SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run --runtime=nvidia -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '****************'
docker exec -i $NAME ./ask /host/Users/a7.txt 5
echo '****************'
docker exec -i $NAME ./answer /host/Users/a7.txt /host/Users/a7_questions.txt
echo '****************'
docker stop $NAME >/dev/null