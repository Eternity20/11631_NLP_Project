SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run --runtime=nvidia -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '*******QG*********'
sudo docker exec -i $NAME ./ask /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt 3 2>/dev/null
# sudo docker exec -i $NAME ./ask /host/Users/data/set1/a7.txt 3 2>/dev/null
echo '*******QA*********'
sudo docker exec -i $NAME ./answer /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt /host/Users/test_questions.txt 2>/dev/null
echo '*****Finish********'
sudo docker stop $NAME >/dev/null