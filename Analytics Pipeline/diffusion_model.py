import os
import random
import torch
from deepface import DeepFace
from diffusers import StableDiffusionPipeline
from PIL import Image

def define_stable_diffusion_model():
    """
    Define and return a stable diffusion model pipeline.

    Returns:
        pipe (StableDiffusionPipeline): A pre-trained stable diffusion model pipeline.
    """
    
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


def generate_images_situational_prompt(df, pipe, directory, image_path, file_path):
    """
    Generate images using a situational prompt and save them to the specified directory.

    Args:
        df (DataFrame): DataFrame containing information about the images to be generated.
        pipe (StableDiffusionPipeline): Pre-trained stable diffusion model pipeline.
        directory (str): Directory where generated images and CSV file will be saved.
        image_path (str): Path to the directory for saving generated images.
        file_path (str): Path to the CSV file for saving image information.
    """
    
    # Get the current working directory
    cwd = os.getcwd()

    # Construct the absolute file path to the CSV file
    file_path = os.path.join(cwd, '..', directory, file_path)
    image_path = os.path.join(cwd, '..', directory, image_path)

    
    index = 0
    total_new_samples = 0
    total_images = 0
    for index, row in df.iterrows():

        # Accessing values of each column in the current row
        print(index, total_new_samples)

        if(index > 0):

            race = row['race']
            gender = row['gender']
            emotion = row['emotion']
            other_attributes = row ['other_attributes']
            count = row['count']

            #print(total_new_samples, race, gender, emotion, count)
            total_images = 0
            other_attributes = other_attributes.replace("Attractive,", "").replace(", Young", "").replace("No_Beard,", "").replace("Smiling,", "").replace("_", " ").replace("No Beard,", "").lower()

            # Situation examples for each emotion
            emotion_situations = {
                "happy": [
                    f"A {race} {gender} with a bright smile, {other_attributes}, celebrating their achievements with friends and family.",
                    f"An {race} {gender} beaming with happiness having {other_attributes} while enjoying a delicious meal with loved ones.",
                    f"A {race} {gender} having {other_attributes} displaying a cheerful expression as they listen to their favorite music.",
                    f"{race.capitalize()} {gender} with a joyful grin, having {other_attributes} sharing laughter with friends at a social gathering.",
                    f"A {race} {gender} radiating happiness having {other_attributes}while playing with a pet outdoors.",
                    f"An {race} {gender} having {other_attributes} showing a gleeful smile as they embrace a loved one.",
                    f"{race.capitalize()} {gender} having {other_attributes} laughing wholeheartedly during a fun-filled moment with friends.",
                    f"A {race} {gender} having {other_attributes} expressing sheer delight while participating in a joyful dance.",
                    f"An {race} {gender} with a beaming face, having {other_attributes}, enjoying a beautiful sunset by the beach.",
                    f"{race.capitalize()} {gender} having {other_attributes} celebrating a personal achievement with a genuine and happy expression.",
                    f"A {race} {gender} having {other_attributes}, brightening up the room with their infectious laughter and positive vibe.",
                    f"An {race} {gender} having {other_attributes}, grinning from ear to ear while exploring an amusement park.",
                    f"{race.capitalize()} {gender} having {other_attributes}, sharing an elated expression while engaging in their favorite hobby.",
                    f"A {race} {gender} having {other_attributes}, displaying a radiant smile during a heartwarming family reunion.",
                    f"An {race} {gender} having {other_attributes}, spreading joy with their enthusiastic and cheerful demeanor."
                ],
                "sad": [
                    f"A {race} {gender} having {other_attributes}, looking somber and reflective, lost in thought about past memories.",
                    f"An {race} {gender} having {other_attributes}, with a downcast gaze, silently dealing with their feelings of sadness.",
                    f"{race.capitalize()} {gender} having {other_attributes},showing a mournful expression, seeking solace by spending time alone.",
                    f"A {race} {gender} having {other_attributes}, wiping away tears, struggling to cope with the weight of their emotions.",
                    f"An {race} {gender} having {other_attributes}, with a pensive expression, lost in introspection and deep contemplation.",
                    f"{race.capitalize()} {gender} having {other_attributes}, gazing sadly out of a window on a rainy day.",
                    f"A {race} {gender} having {other_attributes}, conveying a sense of sorrow while reading a heartfelt letter.",
                    f"An {race} {gender} having {other_attributes}, with a wistful look, reminiscing about fond memories from the past.",
                    f"{race.capitalize()} {gender} having {other_attributes}, experiencing a moment of vulnerability and sadness while listening to a sad song.",
                    f"A {race} {gender}, having {other_attributes}, reflecting a sense of longing and melancholy in their expression.",
                    f"An {race} {gender} having {other_attributes}, with a teary-eyed gaze, navigating the complexities of their emotions.",
                    f"{race.capitalize()} {gender}, having {other_attributes}, wearing a somber expression, quietly contemplating life's challenges.",
                    f"A {race} {gender} having {other_attributes}, showing a forlorn expression as they process difficult news.",
                    f"An {race} {gender} having {other_attributes}, with a heavy heart, experiencing the weight of their emotions.",
                    f"{race.capitalize()} {gender} having {other_attributes}, expressing their inner sadness through their introspective eyes."
                ],
                "neutral": [
                    f"An {race} {gender} having {other_attributes}with a composed demeanor, attentively observing their surroundings.",
                    f"{race.capitalize()} {gender} having {other_attributes}displaying an indifferent expression, focused on their current task.",
                    f"A {race} {gender} having {other_attributes}maintaining a calm and neutral face while engrossed in deep thought.",
                    f"{race.capitalize()} {gender} having {other_attributes} with an expressionless gaze, reflecting a tranquil state of mind.",
                    f"An {race} {gender} having {other_attributes} exuding a sense of serenity, unaffected by external distractions.",
                    f"A {race} {gender} having {other_attributes} with a poised and collected expression during a professional presentation.",
                    f"{race.capitalize()} {gender} having {other_attributes} displaying a sense of detachment while immersed in a good book.",
                    f"An {race} {gender} having {other_attributes} embodying a sense of calm and equilibrium, even in the midst of chaos.",
                    f"A {race} {gender} having {other_attributes} portraying a state of inner peace while practicing mindfulness.",
                    f"{race.capitalize()} {gender} having {other_attributes} with an impassive look, concealing their emotions with a steady demeanor.",
                    f"An {race} {gender} having {other_attributes} projecting an air of cool composure as they navigate a challenging situation.",
                    f"A {race} {gender} having {other_attributes} maintaining a neutral expression during a moment of deep introspection.",
                    f"{race.capitalize()} {gender} having {other_attributes} exuding a sense of tranquility while enjoying a solitary walk in nature.",
                    f"An {race} {gender} having {other_attributes} reflecting a calm and collected demeanor while engaged in meditation.",
                    f"A {race} {gender} having {other_attributes} maintaining a steady and composed face in the midst of a busy environment."
                ],
                "surprise": [
                    f"A {race} {gender} having {other_attributes} wearing an astonished look, reacting in surprise to an unexpected event.",
                    f"An {race} {gender} having {other_attributes} with wide eyes and an open mouth, caught off guard by a surprising revelation.",
                    f"{race.capitalize()} {gender} having {other_attributes} displaying amazement and astonishment as they witness an extraordinary sight.",
                    f"A {race} {gender} having {other_attributes} showing a stunned expression, unable to hide their surprise at a sudden turn of events.",
                    f"An {race} {gender} having {other_attributes} with a jaw-dropping reaction, taken aback by an unexpected announcement.",
                    f"{race.capitalize()} {gender} having {other_attributes} reacting with shock and surprise, their eyes widening in disbelief.",
                    f"A {race} {gender} having {other_attributes} displaying a gobsmacked expression, reacting to an astonishing sight.",
                    f"An {race} {gender} having {other_attributes} showing a mix of wonder and disbelief, captivated by an awe-inspiring moment.",
                    f"{race.capitalize()} {gender} having {other_attributes} exclaiming in surprise and amazement, their expression reflecting genuine astonishment.",
                    f"A {race} {gender} having {other_attributes} showing genuine surprise and excitement while unwrapping a thoughtful gift.",
                    f"An {race} {gender} having {other_attributes} with an amazed and captivated expression, experiencing wonder at a breathtaking scene.",
                    f"{race.capitalize()} {gender} having {other_attributes} displaying a pleasantly surprised expression, reacting to a joyful surprise.",
                    f"A {race} {gender} having {other_attributes} caught off guard with a look of astonishment, reacting to an unexpected visit.",
                    f"An {race} {gender} having {other_attributes} showing a surprised and delighted expression while witnessing a magical moment.",
                    f"{race.capitalize()} {gender} having {other_attributes} exuding wonder and astonishment as they experience an unexpected adventure."
                ],
                "fear": [
                    f"A {race} {gender} having {other_attributes} with a fearful expression, eyes wide and heart racing after encountering something frightening.",
                    f"An {race} {gender} having {other_attributes} showing apprehension and nervousness, facing their fears with a cautious look.",
                    f"{race.capitalize()} {gender} with a frightened gaze, reacting with alarm to a sudden noise.",
                    f"A {race} {gender} having {other_attributes} looking panicked and scared, reacting to a challenging situation with trepidation.",
                    f"An {race} {gender} having {other_attributes} displaying an alarmed expression, responding to a startling event with urgency.",
                    f"{race.capitalize()} {gender} showing a sense of unease and anxiety in their expression.",
                    f"A {race} {gender} having {other_attributes}, with a worried and tense look, grappling with their inner fears.",
                    f"An {race} {gender} having {other_attributes}, displaying a nervous expression, their eyes darting around as they assess their surroundings.",
                    f"{race.capitalize()} {gender} having {other_attributes}, reacting with a startled expression, their heart racing from a sudden scare.",
                    f"A {race} {gender} having {other_attributes}, conveying a sense of uneasiness and vulnerability through their worried gaze.",
                    f"An {race} {gender} having {other_attributes}, with a cautious expression, their body language reflecting a state of caution.",
                    f"{race.capitalize()} {gender} having {other_attributes}, showing signs of distress and fear as they encounter an unexpected challenge.",
                    f"A {race} {gender} having {other_attributes}, looking anxious and on edge, their expression mirroring their internal turmoil.",
                    f"An {race} {gender} having {other_attributes}, displaying a startled expression, their body language tense as they face their fears.",
                    f"{race.capitalize()} {gender} having {other_attributes}, reacting with a mixture of fear and surprise to an unexpected event."
                ],
                "disgust": [
                    f"A {race} {gender} with {other_attributes}, wrinkling their nose in disgust, reacting to an unpleasant smell with a grimace.",
                    f"An {race} {gender} having {other_attributes}, with a displeased expression, recoiling from something distasteful.",
                    f"{race.capitalize()} {gender} having {other_attributes}, showing aversion with a disgusted look, encountering something they find repulsive.",
                    f"A {race} {gender} having {other_attributes}, displaying a revulsed expression, reacting to something unappetizing with clear distaste.",
                    f"An {race} {gender} having {other_attributes}, cringing and displaying a repelled expression, reacting to an off-putting sight.",
                    f"{race.capitalize()} {gender} with {other_attributes}, conveying a sense of displeasure and distaste through their facial expression.",
                    f"A {race} {gender} with {other_attributes}, showing a disgusted look, their expression revealing their strong negative reaction.",
                    f"An {race} {gender} with {other_attributes}, displaying a grimace of disgust, responding to an unpalatable taste.",
                    f"{race.capitalize()} {gender} having {other_attributes}, exuding a strong sense of aversion and distaste in their expression.",
                    f"A {race} {gender} having {other_attributes}, reacting with a clear look of disgust and disapproval to an unsavory situation.",
                    f"An {race} {gender} having {other_attributes}, displaying a repulsed expression, their body language reflecting their dislike.",
                    f"{race.capitalize()} {gender} having {other_attributes}, showing signs of displeasure and aversion, their expression speaking volumes.",
                    f"A {race} {gender} having {other_attributes}, visibly reacting with a cringe of disgust to an unexpected encounter.",
                    f"An {race} {gender} having {other_attributes}, with a disgusted expression, their body language suggesting a strong negative response.",
                    f"{race.capitalize()} {gender} conveying their revulsion through their facial expression upon encountering something unpleasant."
                ],
                "angry": [
                    f"A {race} {gender} having {other_attributes}, with a determined and intense expression, channeling their anger into focused action.",
                    f"An {race} {gender}  displaying an irate expression, their eyes narrowed and brows furrowed.",
                    f"{race.capitalize()} {gender} having {other_attributes}, showing signs of frustration and anger, their expression reflecting their emotions.",
                    f"A {race} {gender} having {other_attributes}, with a fiery gaze, their facial muscles tense as they navigate their anger.",
                    f"An {race} {gender} having {other_attributes}, projecting a stern and assertive look, clearly conveying their anger.",
                    f"{race.capitalize()} {gender} having {other_attributes}, displaying a heated expression, their features twisted with irritation.",
                    f"A {race} {gender} having {other_attributes}, visibly agitated and displaying signs of anger through their intense stare.",
                    f"An {race} {gender} having {other_attributes}, with a fierce expression, their lips pressed tightly together as they control their anger.",
                    f"{race.capitalize()} {gender} exuding a sense of outrage and frustration through their body language.",
                    f"A {race} {gender} displaying a confrontational expression, ready to express their anger and opinions.",
                    f"An {race} {gender} showing a determined and resolute expression, channeling their anger into determination.",
                    f"{race.capitalize()} {gender} with a steely glare, visibly angered by a challenging situation.",
                    f"A {race} {gender} conveying their indignation through their facial expression, unafraid to speak their mind.",
                    f"An {race} {gender} having {other_attributes}, with a forceful expression, their eyes ablaze with the intensity of their anger.",
                    f"{race.capitalize()} {gender} showing signs of righteous anger, standing up for their beliefs with conviction."
                ]
            }


            while (total_images != count * 3):
                # Select a random situation example based on the given emotion
                selected_example = random.choice(emotion_situations[emotion])
                prompt = selected_example
                print(prompt)

                image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
                image.save(f"sample.png")
                #print(total_images, "True")
                image_name = f"{total_new_samples}_stable_diffusion_{race}_{gender}_{emotion}_{total_images}.jpg"
                fields=[image_name, race, gender, emotion]

                image.save(image_path + "/" + image_name)
                with open(file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    total_new_samples = total_new_samples + 1

                total_images = total_images + 1
                
                
                
def generate_images_static_prompt(df, pipe, directory, image_path, file_path):
    """
    Generate images using a static prompt and save them to the specified directory.

    Args:
        df (DataFrame): DataFrame containing information about the images to be generated.
        pipe (StableDiffusionPipeline): Pre-trained stable diffusion model pipeline.
        directory (str): Directory where generated images and CSV file will be saved.
        image_path (str): Path to the directory for saving generated images.
        file_path (str): Path to the CSV file for saving image information.
    """
    

    # Get the current working directory
    cwd = os.getcwd()

    # Construct the absolute file path to the CSV file
    file_path = os.path.join(cwd, '..', directory, file_path)
    image_path = os.path.join(cwd, '..', directory, image_path)
    
    index = 0
    total_new_samples = 0
    total_images = 0
    
    # Defining values which are characteristic of the emotion
    happy_emotion = ['smile', 'grin', 'full smile', 'smiling', 'beaming', 'grinning', 'radiant', 'joyful', 'gleaming', 'cheerful', 'bright-eyed', 'lively', 'laughing']
    sad_emotion = ['frown', 'teary-eyed', 'dismal', 'downcast', 'gloomy', 'mournful', 'sorrowful', 'dejected', 'heartbroken', 'weepy', 'downtrodden', 'melancholy', 'grief-stricken']
    neutral_emotion = ['expressionless', 'blank', 'indifferent', 'apathetic', 'stoic', 'emotionless', 'deadpan', 'impassive', 'unemotional', 'calm', 'composed', 'detached', 'serene']
    surprise_emotion = ['astonished', 'amazed', 'startled', 'shocked', 'dumbfounded', 'stunned', 'wide-eyed', 'gobsmacked', 'taken aback', 'flabbergasted', 'baffled', 'bewildered', 'astounded']
    fear_emotion = ['terrified', 'frightened', 'scared', 'startled', 'panicked', 'horrified', 'petrified', 'aghast', 'shaken', 'anxious', 'nervous', 'fearful', 'jittery']
    disgust_emotion = ['repulsed', 'revolted', 'nauseated', 'sickened', 'displeased', 'appalled', 'offended', 'abhorrent', 'distasteful', 'repelled', 'disliking', 'aversion', 'horrified']
    angry_emotion = ['angry', 'irate', 'furious', 'enraged', 'infuriated', 'livid', 'indignant', 'outraged', 'annoyed', 'exasperated', 'irritated', 'agitated', 'incensed']


    for index, row in df.iterrows():

        # Accessing values of each column in the current row
        print(index, total_new_samples)

        if(index > 0):

            race = row['race']
            gender = row['gender']
            emotion = row['emotion']
            other_attributes = row ['other_attributes']
            count = row['count']

            total_images = 0
            print(race, gender, emotion)

            if emotion == 'happy':
                attribute = random.choice(happy_emotion)
            elif emotion == 'sad':
                attribute = random.choice(sad_emotion)
            elif emotion == 'neutral':
                attribute = random.choice(neutral_emotion)
            elif emotion == 'surprise':
                attribute = random.choice(surprise_emotion)
            elif emotion == 'fear':
                attribute = random.choice(fear_emotion)
            elif emotion == 'disgust':
                attribute = random.choice(disgust_emotion)
            """
            if gender == 'Man':
                attribute1 = random.choice(facial_features_man)
            elif gender == 'Woman':
                attribute1 = random.choice(facial_features_woman)

            if race == 'asian':
                attribute2 = random.choice(asian_features)
            elif race == 'white':
                attribute2 = random.choice(white_features)
            elif race == 'latino hispanic':
                attribute2 = random.choice(latino_hispanic_features)
            elif race == 'black':
                attribute2 = random.choice(black_features)
            elif race == 'middle eastern':
                attribute2 = random.choice(middle_eastern_features)"""

            other_attributes = other_attributes.replace("Attractive,", "").replace(", Young", "").replace("No_Beard,", "").replace("Smiling,", "").replace("_", " ").replace("No Beard,", "").lower()
            prompt = f"A {race} {gender} having with a {emotion} {attribute} face having {other_attributes}"
            print(prompt)

            while (total_images != count * 3):
                # Select a random situation example based on the given emotion
                selected_example = random.choice(emotion_situations[emotion])
                prompt = selected_example
                print(prompt)

                image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
                image.save(f"sample.png")
                #print(total_images, "True")
                image_name = f"{total_new_samples}_stable_diffusion_{race}_{gender}_{emotion}_{total_images}.jpg"
                fields=[image_name, race, gender, emotion]

                image.save(image_path + "/" + image_name)
                with open(file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    total_new_samples = total_new_samples + 1

                total_images = total_images + 1



def remove_samples(directory, image_path, file_path):
    """
    Evaluate image samples and remove entries from CSV mapping if image is black.

    Args:
        directory (str): Directory where data is stored.
        image_path (str): Path to the image folder.
        file_path (str): Path to the CSV file containing image mappings.
    """
    # Get the current working directory
    cwd = os.getcwd()
    
    # Construct the absolute file path to the image folder
    image_path = os.path.join(cwd, '..', directory, image_path)
    
    # Load the CSV file with image mappings
    csv_file_path = os.path.join(cwd, '..', image_path, file_path)
    image_mappings = pd.read_csv(csv_file_path)
    image_mappings.columns = ["image_name", "race", "gender", "emotion"]
    
    # Iterate over the files in the directory
    for image_name in os.listdir(image_path):
        # Construct the absolute file path to the image
        image_file_path = os.path.join(image_path, image_name)
        
        # Open the image using PIL
        image = Image.open(image_file_path)
        
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        # Check if the image contains only black pixels
        if np.all(image_array == 0):
            # Delete the image file
            os.remove(image_file_path)
            
            # Remove the image mapping from the CSV file
            image_mappings = image_mappings[image_mappings['image_name'] != image_name]
            image_mappings.to_csv(csv_file_path, index=False)
            
            # Provide feedback to the user
            print(f"Removed {image_name} and its mapping from the CSV.")
    
    # Notify user when the process is completed
    print("Process completed.")


def evaluate_and_interact(directory, image_path, file_path):
    """
    Evaluate image samples and interactively remove unwanted images.

    Args:
        directory (str): Directory where data is stored.
        image_path (str): Path to the image folder.
        file_path (str): Path to the CSV file containing image mappings.
    """
    # Get the current working directory
    cwd = os.getcwd()
    
    # Construct the absolute file path to the image folder
    image_path = os.path.join(cwd, '..', directory, image_path)
    
    # Load the CSV file with image mappings
    csv_file_path = os.path.join(cwd, '..', directory, file_path)
    image_mappings = pd.read_csv(csv_file_path)
    
    # Iterate over the files in the directory
    for image_name in os.listdir(image_path):
        # Construct the absolute file path to the image
        image_file_path = os.path.join(image_path, image_name)
        
        # Open the image using PIL
        image = Image.open(image_file_path)
        
        # Display the image
        image.show()
        
        # Ask the user if they want to keep the image
        user_input = input("Do you want to keep this image? (y/n): ")
        
        if user_input.lower() == 'n':
            # Delete the image file
            os.remove(image_file_path)
            
            # Remove the image mapping from the CSV file
            image_mappings = image_mappings[image_mappings['image_name'] != image_name]
            image_mappings.to_csv(csv_file_path, index=False)

    print("Process completed.")

# Call the function with appropriate arguments
evaluate_and_interact('Data', 'synthetic_data_augmentation_celebA_vggnet/', 'images_generated_stable_diffusion_vggnet.csv')

