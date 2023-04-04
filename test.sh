SHARE=$(pwd)
IMAGE=$1
NAME=$(sudo docker run -d -v ${SHARE}:/host/Users -it ${IMAGE} /bin/bash)
echo '****************'
sudo docker exec -i $NAME ./ask /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt 3 2>/dev/null
echo '****************'
sudo docker exec -i $NAME ./answer /host/Users/data/wikipedia_text/pokemon_characters/pikachu_pokemon.txt /host/Users/test_questions.txt
echo '****************'
sudo docker stop $NAME >/dev/null
