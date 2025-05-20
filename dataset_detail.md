## Details of Datasets

### Dataset Statistics

The statistics of the datasets used in experiments can be found in the following table. For CIFAR-100, we use the original training and test dataset. For EuroSAT, Oxford-IIIT Pet-b, and Caltech-101, we use the training, validation, and test dataset splitting of CoOp.

**Table: Dataset statistics and hand-crafted prompts**

| Dataset         | Classes | Train  | Val   | Test   | Hand-crafted prompt                       |
| --------------- | ------- | ------ | ----- | ------ | ----------------------------------------- |
| CIFAR-100       | 100     | 50,000 | N/A   | 10,000 | "a photo of a \[CLASS]."                  |
| EuroSAT         | 10      | 13,500 | 5,400 | 8,100  | "a centered satellite photo of \[CLASS]." |
| Oxford-IIIT Pet | 37      | 2,944  | 736   | 3,669  | "a photo of a \[CLASS], a type of pet."   |
| Caltech-101     | 100     | 4,128  | 1,649 | 2,465  | "a photo of a \[CLASS]."                  |

---

### Dataset Splitting

Since the datasets used in the experiment are multi-class, we transform them into binary classification datasets by dividing the classes into two groups: positive and negative. Given the large number of classes, manual splitting is challenging, so we utilize GPT-4o to group the classes in each dataset.

**Grouping Prompt**:

*I want to use the \[DATASET\_NAME] dataset for binary classification and need to divide the \[CLASS\_NUMBER] classes into two groups. The classes in each group should be semantically similar, while the classes between the two groups should be semantically different so that each new class is discriminative. The ratio of the amount of the data from these two new classes should be 4:6 or 5:5. Can you provide the detailed group information of this? Please provide a detailed step-by-step explanation, and you do not need to provide the python code.*

We provide the final dataset splitting below:

---

**CIFAR-100**
**Group1 (40 classes):**
`[trout, orchids, bowls, telephone, porcupine, oranges, cockroach, bear, oak, palm, aquarium fish, apples, clock, squirrel, train, roses, cans, butterfly, spider, dinosaur, pears, table, skyscraper, fox, boy, otter, poppies, bee, castle, tank, shark, sunflowers, road, elephant, bicycle, leopard, worm, mouse, maple, willow]`

**Group2 (60 classes):**
`[beaver, dolphin, seal, whale, flatfish, ray, tulips, bottles, cups, plates, mushrooms, sweet peppers, computer keyboard, lamp, television, bed, chair, couch, wardrobe, beetle, caterpillar, lion, tiger, wolf, bridge, house, cloud, forest, mountain, plain, sea, camel, cattle, chimpanzee, kangaroo, possum, raccoon, skunk, crab, lobster, snail, baby, girl, man, woman, crocodile, lizard, snake, turtle, hamster, rabbit, shrew, pine, bus, motorcycle, pickup truck, lawn-mower, rocket, streetcar, tractor]`

---

**EuroSAT**
**Group1 (3 classes):**
`[Highway or Road, Industrial Buildings, Residential Buildings]`
**Group2 (7 classes):**
`[Annual Crop Land, Forest, Herbaceous Vegetation Land, Pasture Land, Permanent Crop Land, River, Sea or Lake]`

---

**Oxford-IIIT Pet-b**
**Group1 (19 classes):**
`[abyssinian, bengal, birman, bombay, british_shorthair, chihuahua, egyptian_mau, havanese, japanese_chin, maine_coon, miniature_pinscher, persian, pomeranian, pug, ragdoll, russian_blue, siamese, sphynx, yorkshire_terrier]`

**Group2 (18 classes):**
`[american_bulldog, american_pit_bull_terrier, basset_hound, beagle, boxer, english_cocker_spaniel, english_setter, german_shorthaired, great_pyrenees, keeshond, leonberger, newfoundland, saint_bernard, samoyed, scottish_terrier, shiba_inu, staffordshire_bull_terrier, wheaten_terrier]`

---

**Caltech-101**
**Group1 (49 classes):**
`[ant, bass, beaver, bonsai, brontosaurus, butterfly, cougar_body, cougar_face, crab, crayfish, crocodile, crocodile_head, dalmatian, dolphin, dragonfly, elephant, emu, flamingo, flamingo_head, garfield, gerenuk, hedgehog, hawksbill, ibis, joshua_tree, kangaroo, leopard, llama, lobster, lotus, mayfly, nautilus, octopus, okapi, panda, pigeon, platypus, rhino, rooster, scorpion, sea_horse, starfish, stegosaurus, strawberry, sunflower, tick, trilobite, water_lilly, wild_cat]`

**Group2 (51 classes):**
`[accordion, airplane, anchor, barrel, binocular, brain, buddha, camera, cannon, car_side, ceiling_fan, cellphone, chair, chandelier, cup, dollar_bill, electric_guitar, euphonium, ewer, face, ferry, gramophone, grand_piano, headphone, helicopter, inline_skate, ketch, lamp, laptop, mandolin, menorah, metronome, minaret, motorbike, pagoda, pizza, pyramid, revolver, saxophone, schooner, scissors, snoopy, soccer_ball, stapler, stop_sign, umbrella, watch, wheelchair, windsor_chair, wrench, yin_yang]`

---

**PU Learning Set-Up**
*Table: Positive and negative label groups of datasets and the statistics of those PU training sets.*

| Dataset           | Positive Class Group | Negative Class Group | π    |
| ----------------- | -------------------- | -------------------- | ---- |
| CIFAR-100-a       | Group-1              | Group-2              | 0.4  |
| CIFAR-100-b       | Group-2              | Group-1              | 0.6  |
| EuroSAT-a         | Group-1              | Group-2              | 0.3  |
| EuroSAT-b         | Group-2              | Group-1              | 0.7  |
| Oxford-IIIT Pet-a | Group-1              | Group-2              | 0.51 |
| Oxford-IIIT Pet-b | Group-2              | Group-1              | 0.49 |

---

## Zero-shot Prompt

We use the following prompt with GPT-4o to generate text descriptions for zero-shot CLIP binary classification:

**Prompt**:

*I am doing a binary classification for \[DATASET\_NAME]. I divided the \[DATASET\_NUMBER] classes into two groups to consider them positive and negative. The positive classes are: \[POSITIVE\_GROUP\_NAME\_LIST]. The negative classes are \[NEGATIVE\_GROUP\_NAME\_LIST]. Can you help me summarize each of the two classes so that I can use it for CLIP Zero-Shot Classification? The result should be two sentences, one for each class. Please provide rich and sufficient descriptions for each class, starting with "a photo of {}" and not exceeding the maximum token limit of CLIP. Please think step-by-step.*

We replace the placeholders accordingly. The resulting class descriptions are:

---

**CIFAR-100**
**Positive:**
*a photo of various living organisms and plants, including aquatic mammals, fish, flowers, fruits, vegetables, insects, large carnivores, large omnivores and herbivores, medium-sized mammals, non-insect invertebrates, reptiles, small mammals, and trees.*

**Negative:**
*a photo of various man-made objects, scenes, and people, including vehicles, household electrical devices, household furniture, large man-made outdoor things, large natural outdoor scenes, people, small man-made outdoor things, and small man-made indoor things.*

---

**EuroSAT**
**Positive:**
*a centered satellite photo of man-made environments, featuring expansive highways, industrial complexes with large factories, and dense residential areas filled with houses and buildings, representing urban and developed landscapes.*

**Negative:**
*a centered satellite photo of natural environments, including lush forests, sprawling pastures, agricultural lands like annual and permanent crops, vibrant herbaceous vegetation, flowing rivers, and serene lakes or seas, representing the harmony of diverse ecosystems.*

---

**Oxford-IIIT Pet**
**Positive:**
*a photo of a small pet, including elegant cat breeds like Abyssinian, Bengal, Persian, Siamese, Sphynx, and toy dog breeds like Chihuahua, Pug, Pomeranian, Yorkshire Terrier, Havanese, known for their petite size and distinctive features, a type of pet.*

**Negative:**
*a photo of a medium to large dog breed, such as energetic and robust breeds like American Bulldog, Beagle, Boxer, German Shorthaired Pointer, Saint Bernard, Samoyed, Shiba Inu, known for their strength and active nature, a type of pet.*

---

**Caltech-101**
**Positive:**
*a photo of animals, plants, and other living things, including insects, mammals, birds, aquatic creatures, and plant species like bonsai and lotus.*

**Negative:**
*a photo of man-made objects and vehicles, including items like airplanes, musical instruments, tools, cameras, laptops, and cars.*

---
