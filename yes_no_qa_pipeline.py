import sys

import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AdamW
#import seaborn as sns
#import matplotlib.pyplot as plt


# load data
def encode_data(tokenizer, questions, passages, max_length):
    """Encode the question/passage pairs into features than can be fed to the model."""
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)

    return np.array(input_ids), np.array(attention_masks)


def train_loop(train_dataloader, model, optimizer, device, grad_acc_steps):
    epoch_train_loss = 0  # Cumulative loss
    model.train()
    model.zero_grad()

    for step, batch in tqdm(enumerate(train_dataloader), desc='Train Steps', total=len(train_dataloader)):
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

        loss = outputs[0]
        loss = loss / grad_acc_steps
        epoch_train_loss += loss.item()

        loss.backward()

        if (step + 1) % grad_acc_steps == 0:  # Gradient accumulation is over
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients
            optimizer.step()
            model.zero_grad()

    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    print(f'Training Loss: {epoch_train_loss:.4f}')
    return model, epoch_train_loss
    #train_loss_values.append(epoch_train_loss)

def train(train_dataloader, dev_dataloader, model, optimizer, device, grad_acc_steps, epochs):
    best_val_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Epoch"):
      # Training
      model, train_loss = train_loop(train_dataloader, model, optimizer, device, grad_acc_steps)
      model.eval()
      val_loss = eval_loop(model, device, dev_dataloader)
      print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), 'best_model.pth')
          model.save_pretrained("best_yn_qa_roberta_base", from_pt=True)
      model.train()
    return model


def eval_loop(model, device, dev_dataloader):
  epoch_dev_accuracy = 0
  for step, batch in tqdm(enumerate(dev_dataloader), desc='Eval Steps', total=len(dev_dataloader)):
    input_ids = batch[0].to(device)
    attention_masks = batch[1].to(device)
    labels = batch[2]

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()

    predictions = np.argmax(logits, axis=1).flatten()
    labels = labels.numpy().flatten()

    epoch_dev_accuracy += np.sum(predictions == labels) / len(labels)

  epoch_dev_accuracy = epoch_dev_accuracy / len(dev_dataloader)
  return epoch_dev_accuracy
  #dev_acc_values.append(epoch_dev_accuracy)


#def visualize():
#    sns.set()

#    plt.plot(train_loss_values, label="train_loss")

#    plt.xlabel("Epoch")
#    plt.ylabel("Loss")
#    plt.title("Training Loss")
#    plt.legend()
#    plt.xticks(np.arange(0, 5))
#    plt.show()

#    plt.plot(dev_acc_values, label="dev_acc")

#    plt.xlabel("Epoch")
#    plt.ylabel("Accuracy")
#    plt.title("Evaluation Accuracy")
#    plt.legend()
#    plt.xticks(np.arange(0, 5))
#    plt.show()

#    print(dev_acc_values)


def predict(question, passage):
  sequence = tokenizer.encode_plus(question, passage,  truncation=True,max_length=512,return_tensors="pt")['input_ids'].to(device)

  logits = model(sequence)[0]
  probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
  proba_yes = round(probabilities[1], 2)
  proba_no = round(probabilities[0], 2)

  print(f"Question: {question}, Yes: {proba_yes}, No: {proba_no}")

if __name__ == '__main__':
    # Use a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set seeds for reproducibility
    random.seed(26)
    np.random.seed(26)
    torch.manual_seed(26)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    model.to(device) # Send the model to the GPU if we have one
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    # Loading data
    #train_data_df = pd.read_json("/content/train.jsonl", lines=True, orient='records')
    #dev_data_df = pd.read_json("/content/dev.jsonl", lines=True, orient="records")
    train_data_df = pd.read_json("train.jsonl", lines=True, orient='records')
    dev_data_df = pd.read_json("dev.jsonl", lines=True, orient="records")

    passages_train = train_data_df.passage.values
    questions_train = train_data_df.question.values
    answers_train = train_data_df.answer.values.astype(int)

    passages_dev = dev_data_df.passage.values
    questions_dev = dev_data_df.question.values
    answers_dev = dev_data_df.answer.values.astype(int)

    # Encoding data
    max_seq_length = 512
    input_ids_train, attention_masks_train = encode_data(tokenizer, questions_train, passages_train, max_seq_length)
    input_ids_dev, attention_masks_dev = encode_data(tokenizer, questions_dev, passages_dev, max_seq_length)

    train_features = (input_ids_train, attention_masks_train, answers_train)
    dev_features = (input_ids_dev, attention_masks_dev, answers_dev)

    batch_size = 8

    train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
    dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]

    train_dataset = TensorDataset(*train_features_tensors)
    dev_dataset = TensorDataset(*dev_features_tensors)

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=3)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size, num_workers=3)

    #training
    epochs = 10
    grad_acc_steps = 8
    train_loss_values = []
    dev_acc_values = []
    model = train(train_dataloader, dev_dataloader, model, optimizer, device, grad_acc_steps, epochs)

    #predict
    passage_superbowl = """Pikachu is a fictional species in the Pokémon media franchise. Designed by Atsuko Nishida and Ken Sugimori, Pikachu first appeared in the 1996 Japanese video games Pokémon Red and Green created by Game Freak and Nintendo, which were released outside of Japan in 1998 as Pokémon Red and Blue. Pikachu is a yellow, mouse-like creature with electrical abilities. It is a major character in the Pokémon franchise, serving as its mascot and as a major mascot for Nintendo.
    Pikachu is widely considered to be the most popular and well-known Pokémon species, largely due to its appearance in the Pokémon anime television series as the companion of protagonist Ash Ketchum. In most vocalized appearances Pikachu is voiced by Ikue Ōtani, though it has been portrayed by other actors, notably Ryan Reynolds in the live-action animated film Pokémon Detective Pikachu. Pikachu has been well received by critics, with particular praise given for its cuteness, and has come to be regarded as an icon of Japanese pop culture.


    == Concept and design ==

    Developed by Game Freak and published by Nintendo, the Pokémon series began in Japan in 1996, and features several species of creatures called "Pokémon" that players, called "trainers", are encouraged to capture, train, and use to battle other players' Pokémon or interact with the game's world. Pikachu was one of several different Pokémon designs conceived by Game Freak's character development team. Artist Atsuko Nishida is credited as the main person behind Pikachu's design, which was later finalized by artist Ken Sugimori. According to series producer Satoshi Tajiri, the name is derived from a combination of two Japanese onomatopoeia: ピカピカ (pikapika), a sparkling sound, and チュウチュウ (chūchū), a sound a mouse makes. Despite its name's origins, however, Nishida based Pikachu's original design, especially its cheeks, on squirrels. Developer Junichi Masuda noted Pikachu's name as one of the most difficult to create, due to an effort to make it appealing to both Japanese and American audiences.Standing 40 centimetres (1 ft 4 in) tall, Pikachu were the first "Electric-type" Pokémon created, their design intended to revolve around the concept of electricity. They are creatures that have short, yellow fur with brown markings covering their backs and parts of their lightning bolt-shaped tails. They have black-tipped, pointed ears and red circular pouches on their cheeks, which can spark with electricity.  They attack primarily by projecting electricity from their bodies at their targets. Within the context of the franchise, Pikachu can transform, or "evolve," into a Raichu when exposed to a "Thunder Stone." In Pokémon Gold and Silver, "Pichu" was introduced as an evolutionary predecessor to Pikachu. In Pokémon Diamond and Pearl, gender differences were introduced; since those games, female Pikachu have an indent at the end of their tails, giving the tail a heart-shaped appearance.
    Initially, both Pikachu and fellow Pokémon Clefairy were chosen to be lead characters for the franchise merchandising, with the latter as the primary mascot to make the early comic book series more "engaging". The idea of Pikachu as the mascot of the animated series was suggested by the production company OLM, Inc., which found that Pikachu was popular amongst schoolchildren and could appeal to both boys and girls, as well as their mothers. Pikachu resembled a familiar, intimate pet, and yellow is a primary color and easier for children to recognize from a distance. Additionally, the only other competing yellow mascot at the time was Winnie-the-Pooh.  Pikachu was originally planned to have a second evolution called Gorochu, which was intended to be the evolved form of Raichu.Pikachu's design has evolved from its once-pudgy body to having a slimmer waist, straighter spine, and more defined face and neck; Sugimori has stated these design changes originated in the anime, making Pikachu easier to animate, and were adopted to the games for consistency. "Fat Pikachu"
    was revisited in Pokémon Sword and Shield, where Pikachu received a Gigantamax Form resembling its original design.


    == Appearances ==


    === In video games ===
    Pikachu has appeared in all Pokémon video games, except Black and White, without having to trade. The game Pokémon Yellow features a Pikachu as the only available Starter Pokémon. Based on the Pikachu from the Pokémon anime, it refuses to stay in its Poké Ball, and instead follows the main character around on screen. The trainer can speak to it and it displays different reactions depending on how it is treated. Pikachu also received the ability to learn new attacks such as the Electric-type attack, Thunderbolt, which no other Pokémon could learn naturally.An event from April 1 to May 5, 2010, allowed players of Pokémon HeartGold and SoulSilver to access a route on the Pokéwalker, which solely contained Pikachu which knew attacks that they were not normally compatible with, Surf and Fly. Both of these attacks can be used outside battles as travel aids. Seven "Cap" forms of Pikachu, wearing caps belonging to Ash Ketchum across different seasons, were released across Pokémon Sun and Moon as well as their Ultra versions. These games also released two Z-Crystals exclusive to Pikachu: Pikanium Z, which upgrades Volt Tackle into Catastropika, and Pikashunium Z, which upgrades Thunderbolt into 10,000,000 Volt Thunderbolt when held by a Cap form of Pikachu.Pokémon Let's Go, which is based heavily on Yellow, has Pikachu as a starter in one of its two versions, with the latter version using Eevee instead. This starter Pikachu has access to several secret techniques and exclusive moves. Finally, in Pokémon Sword and Shield, Pikachu gained access to a special Gigantamax form which grants it the ability to deal massive damage and paralyze opponents at the same time.Aside from the main series, Pikachu stars in Hey You, Pikachu! for the Nintendo 64; the player interacts with Pikachu through a microphone, issuing commands to play various mini-games and act out situations. The game Pokémon Channel follows a similar premise of interacting with the Pikachu, though without the microphone. Pikachu appear in almost all levels of Pokémon Snap and its sequel, New Pokémon Snap, games where the player takes pictures of Pokémon for a score. A Pikachu is one of the sixteen starters and ten partners in the Pokémon Mystery Dungeon series. PokéPark Wii: Pikachu's Adventure and its sequel, PokéPark 2: Wonders Beyond, features a Pikachu as the main protagonist. Pikachu has appeared in all five Super Smash Bros. fighting games as a playable character, including in Pokkén Tournament, along with "Pikachu Libre", based on "Cosplay Pikachu" from Omega Ruby and Alpha Sapphire. Detective Pikachu features a talking Pikachu who becomes a detective and helps to solve mysteries, while Pikachu also appears in a multiplayer online battle arena game Pokémon Unite. It has also appeared in Pokémon Rumble World, Pokémon Go, and also puzzle games such as Pokémon Shuffle, Pokémon Battle Trozei, Pokémon Picross, Pokémon Café Mix, and including 2022 Pokémon Legends: Arceus.


    === In anime ===

    The Pokémon anime series and films feature the adventures of Ash Ketchum and his Pikachu, traveling through the various regions of the Pokémon universe. They are accompanied by a group of alternating friends.
    In the first episode, Ash Ketchum, a young boy from Pallet Town, turns 10 years old and acquires his first Pokémon, a Pikachu, from Professor Oak. At first, Pikachu largely ignores Ash's requests, shocking him frequently and refusing to be confined to the conventional method of Pokémon transportation, a Poké Ball. However, Ash puts himself in danger to defend Pikachu from a flock of wild Spearow, then rushes Pikachu to a Pokémon Center. Through Ash's actions, Pikachu warms up to Ash, although Pikachu still refuses to go into the Poké Ball. Soon after, Pikachu shows great power that sets him apart from other Pokémon, and other Pikachu, which causes Team Rocket to constantly attempt to capture him in order to win favor from their boss, Giovanni.Other wild and trained Pikachu appear throughout the series, often interacting with Ash and his Pikachu. The most notable among these is Ritchie's Pikachu, Sparky (レオン, Reon, Leon). Like most other Pokémon, Pikachu communicates only by saying syllables of his own name. He is voiced by Ikue Ōtani in all versions of the anime. In Pokémon Live!, the musical stage show adapted from the anime, Pikachu was played by Jennifer Risser.


    === In film ===
    In the 2019 film Pokémon Detective Pikachu, a detective Pikachu is voiced by Ryan Reynolds and Ōtani. Pikachu is featured in a 2021 Katy Perry music video, "Electric".


    === In other Pokémon media ===
    Pikachu is a prominent Pokémon in many of the Pokémon manga series. In Pokémon Adventures, main characters Red and Yellow both train Pikachu, which create an egg that Gold hatches into a Pichu. Other series, including Pokémon Pocket Monsters,  Magical Pokémon Journey and Getto Da Ze also feature Pikachu. Other manga series, such as Electric Tale of Pikachu, and Ash & Pikachu, feature Ash Ketchum's Pikachu from the anime series.


    == Reception ==


    === Promotion ===

    As the mascot of the franchise, Pikachu has made multiple appearances in various promotional events and merchandise. In 1998, then Topeka, Kansas Mayor Joan Wagnon renamed the town "ToPikachu" for a day, and the renaming was repeated in 2018 by Mayor Michelle De La Isla with the release of the Pokémon Let's Go games. A "got milk?" advertisement featured Pikachu on April 25, 2000.A Pikachu balloon has been featured in the Macy's Thanksgiving Day Parade since 2001. The original balloon was flown for the last time publicly at the Pokémon: Tenth Anniversary "Party of the Decade" on August 8, 2006, in Bryant Park in New York City, and a new Pikachu balloon that chases a Poké Ball and has light-up cheeks debuted at the 2006 parade. In the 2014 parade, a new Pikachu balloon was wearing a green scarf and holding a smaller Pikachu snowman. As of 2021, the latest balloon is that of a Pikachu and Eevee together in a sled.Pikachu and ten other Pokémon were chosen as Japan's mascots in the 2014 FIFA World Cup. In August 2017, The Pokémon Company had partnered with Snap Inc. to bring Pikachu to the social media app, Snapchat. In December 2020, a 15-minute long ASMR video of Pikachu by The Pokémon Company was released. ANA Boeing 747-400 (JA8962) planes have been covered with images of Pokémon including Pikachu since 1998.  In 2021, the first Pokémon Jet (Boeing 747-400D) featuring entirely Pikachu debuted. Pikachu has been made into several different toy and plush forms, as well as other items, including a robot Tomy Pikachu, figures, fishing lures, gaming setups, necklaces, hats, inflatable furniture, and wire loop games. In 2022, My Nintendo Japan released a Pikachu and Eevee cable holder.Collectible cards featuring Pikachu have appeared since the initial Pokémon Trading Card Game released in October 1996, including limited edition promotional cards. One of these collectible cards was "Pikachu Illustrator", limited to about 20-40 printed in 1998, and was auctioned off for about $55,000 in 2016, and then $375,000 in 2021.  For the franchise's 25th anniversary, The Pokémon Company announced special trading cards in 2021, each featuring 25 Pikachu drawn by 25 artists. The character has also been used in promotional merchandising at fast-food chains such as McDonald's, Wendy's and Burger King.Pikachu has been mentioned in a variety of media, including TV series Top Gear and Heroes. Pikachu has appeared several times on The Simpsons from 2002 to 2010.


    === Protests ===
    The Chilean independent politician Giovanna Grandón famously went to many protests during the 2019–2021 Chilean protests dressed in an inflatable Pikachu suit. In July 2021 during the Group of Seven climate summit, a group of protestors dressed as Pikachus demonstrated on Gyllyngvase Beach, Falmouth, while in November 2021, a group of activists dressed up as Pikachu to protest Japan's refusal to reduce coal consumption at COP26.


    === Biology ===
    In 2008, a ligand believed to provide better visual acuity was discovered by the Osaka Bioscience Institute Foundation and named "Pikachurin", in reference to the nimbleness of Pikachu. The name was inspired due to Pikachu's "lightning-fast moves and shocking electric effects".


    === Critical response ===
    Pikachu has been well received by reviewers; it was ranked as the "second best person of the year" by Time in 1999, who called it "the most beloved animated character since Hello Kitty". The magazine noted Pikachu as the "public face of a phenomenon that has spread from Nintendo's fastest selling video game to a trading-card empire", citing the franchise's profits for the year as "the reason for the ranking", behind singer Ricky Martin but ahead of author J.K. Rowling.The character was ranked eighth in a 2000 Animax poll of favorite anime characters. In 2002, Ash's Pikachu received 15th place in TV Guide's 50 greatest cartoon characters of all time. GameSpot featured it in their article "All Time Greatest Game Hero". In 2003, Forbes ranked Pikachu as the "eighth top-earning fictional character of the year" with an income of $825 million. In 2004, the character dropped two spots to tenth on the list, taking in $825 million for a second straight year. In a 2008 Oricon poll, Pikachu was voted as the fourth most popular video game character in Japan, tying with Solid Snake. The character has been regarded as the Japanese answer to Mickey Mouse and as being part of a movement of "cute capitalism". Pikachu was listed 8th in IGN's "Top 25 Anime Characters of All Time." Manga artist Hiro Mashima referred to Pikachu as "the greatest mascot character of all time!" when talking about adding these types of characters to series. Nintendo Power listed Pikachu as their ninth favourite hero, noting that while it was one of the first Pokémon, it still remained "popular to this day". Authors Tracey West and Katherine Noll called Pikachu the best Electric-type Pokémon and the best Pokémon overall. They added that if a person were to go around and ask Pokémon players who their favourite Pokémon was, they would "almost always" choose Pikachu. They also called Pikachu "brave and loyal". In 2011, readers of Guinness World Records Gamer's Edition voted Pikachu as the 20th-top video game character of all time.Zack Zwiezen of Kotaku praised the simplicity of Pikachu's design, describing it as "possibly one of the most iconic characters on the planet". Kevin Slackie of Paste listed Pikachu as second of the best Pokémon. Dale Bishir of IGN described Pikachu as the most important Pokémon that impacted the franchise's history, and further stated that "Its irresistible cuteness, merchandising power, army of clones in every generation... if your mom calls every Pokémon 'Pikachu', then you know in your heart that it is the most important Pokémon of all time." In 2019, Mitsuhiro Arita said that Pikachu and Charizard were "fan favourites" in Pokémon's design on the trading cards. Lauren Rouse of Kotaku listed Pikachu as the best animal companions that are the real MVPs of video games, stating that "Pikachu symbolises one of the best animal-human friendships in pop culture history and it makes a damn good Pokémon to have in your roster." Steven Bogos of The Escapist listed Pikachu as third of their favorite Pokémon, describing it as the "one of the cutest little monsters of all". Time Magazine named Pikachu as one of the twelve most influential video game characters of all time, lauding its appearance as the "most recognizable and beloved sidekick in pop culture." Hobby Consolas also included Pikachu on their "30 best heroes of the last 30 years." In 2021, Chris Morgan for Yardbarker described Pikachu as one of "the most memorable characters from old school Nintendo games", while Rachel Weber of GamesRadar ranked him as second iconic video game character of all time, stating that "If Pokemon has a spokesperson, it's the adorable and electrifying yellow fuzzball."Conversely, Pikachu was ranked first in AskMen's top 10 of the most irritating 1990s cartoon characters. Similarly, in a poll conducted by IGN, it was voted as the 48th best Pokémon, with the staff commenting "despite being the most recognized Pokémon in the world... Pikachu ranks surprisingly low on our top 100". Kotaku writer Patricia Hernandez criticized Pikachu's over-representation in Pokémon-related media, saying: "it's hard not to be barraged by Pikachu's constant presence if you're a Pokémon fan, and it sucks."


    == See also ==

    Pikachu (sculpture), New Orleans


    == Notes ==


    == References ==


    === Citations ===


    === Bibliography ===
    Loe, Casey, ed. Pokémon Special Pikachu Edition Official Perfect Guide. Sunnydale, California: Empire 21 Publishing, 1999.
    Barbo, Maria. The Official Pokémon Handbook. Scholastic Publishing, 1999. ISBN 0-439-15404-9.
    Mylonas, Eric. Pokémon Pokédex Collector's Edition: Prima's Official Pokémon Guide. Prima Games, September 21, 2004. ISBN 0-7615-4761-4
    Nintendo Power. Official Nintendo Pokémon FireRed Version & Pokémon LeafGreen Version Player's Guide. Nintendo of America Inc., August 2004. ISBN 1-930206-50-X
    Nintendo Power. Official Nintendo Pokémon Emerald Player's Guide. Nintendo of America Inc., April 2005. ISBN 1-930206-58-5
    Tobin, Joseph Jay, ed. (2004). Pikachu's Global Adventure: The Rise and Fall of Pokémon. Duke University Press. ISBN 978-0-8223-3287-9.


    == Further reading ==
    Ashcraft, Brian (July 28, 2021). "Japanese Fans Thought Olympians Were 'Pikachu' And 'Raichu'". Kotaku.


    == External links ==

    Pikachu on Pokemon.com
    Pikachu on Bulbapedia
    Pikachu on Serebii
    """

    passage_illuin = """Illuin designs and builds solutions tailored to your strategic needs using Artificial Intelligence
                      and the new means of human interaction this technology enables."""

    superbowl_questions = [
    "Does Pikachu appear in various promotional events and merchandise?",
    "Does Pikachu receive the ability to learn new attacks which no other Pokémon could learn naturally?",
    "Could the trainer speak to Pikachu or not?",
    "Does Pikachu have appeared in all Pokémon video games?",
    "Is a detective Pikachu voiced by Ryan Reynolds and Ōtani?"
    ]

    question_illuin = "Is Illuin the answer to your strategic needs?"

    for s_question in superbowl_questions:
      predict(s_question, passage_superbowl)
    predict(question_illuin, passage_illuin)
sys.exit(0)