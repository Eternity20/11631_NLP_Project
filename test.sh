SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04)
#echo '*******QG*********'
#sudo docker exec -i $NAME ./ask /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt 3 2>/dev/null
echo '*******QA*********'
sudo docker exec -i $NAME ./answer /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt /host/Users/test_questions.txt 2>/dev/null
echo '*****Finish********'
sudo docker stop $NAME >/dev/null