SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo nvidia-docker run -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash --env CUDA_VISIBLE_DEVICES=0)
#echo '*******QG*********'
#sudo docker exec -i $NAME ./ask /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt 3 2>/dev/null
echo '*******QA*********'
sudo nvidia-docker exec -i $NAME ./answer /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt /host/Users/test_questions.txt 2>/dev/null
echo '*****Finish********'
sudo nvidia-docker stop $NAME >/dev/null
