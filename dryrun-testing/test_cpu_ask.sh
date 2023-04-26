SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run --runtime=nvidia -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '****************'
docker exec -i $NAME ./test_cpu_ask.py /host/Users/a7.txt 5 1
#docker exec -i $NAME ./test_cpu_ask.py /host/Users/a7.txt 5 0
