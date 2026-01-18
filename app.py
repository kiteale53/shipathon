import streamlit as st
import re
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import shutil

def load_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v1')

model = load_model()
upload_dir = "/Users/piyushkumar/Documents/Shipathon"
image_path = "/Users/piyushkumar/Documents/hogwarts_logo.jpeg"
sorting_path = "/Users/piyushkumar/Documents/sorting_scene.jpeg"

#page_layout
st.set_page_config(
    page_title="Hogwarts Sorting Hat",
    page_icon="🧙‍♂️",
    layout="wide"
)
#title
st.title("🪄 Hogwarts Sorting Hat")

#explain
st.markdown("""
Ever wondered where **you and your friends** would be sorted at Hogwarts?

📜 Upload your **WhatsApp group chat**  
🧠 We analyze everyone’s messages  
🏰 The Sorting Hat decides your house!
""")

#image
col1, col2 = st.columns([1, 2])
with col1:
    st.image(image_path, width=250)
with col2:
    st.image(
        sorting_path,
        caption="Please wait your turn to be sorted...",
        use_container_width=True
    )

#Instructions
with st.expander("📌 How to use"):
    st.markdown("""
    1. Export your **WhatsApp group chat** as a `.txt` file  
    2. Upload it below  
    3. Click **Run Analysis**  
    4. Watch the Sorting Hat work ✨
    """)

#fiel_uploading
chat_file = st.file_uploader("📂 Upload WhatsApp chat (.txt only)",type=["txt"])
progress = st.progress(0)
progress.progress(20)
progress.progress(50)
progress.progress(90)
progress.progress(100)
if chat_file:
    st.success(f"Uploaded: **{chat_file.name}**")

chat_path = os.path.join(upload_dir, 'chat_file.txt')



def analysis(chats):
    global model
    members = "users"

    Path(members).mkdir(exist_ok=True)

    header_re = re.compile(r'^\[(.*?)\]\s([^:]+):\s(.*)$')

    messages_by_user = {}

    active_user = None
    active_message = []

    seperated_text = chats.split("\n")
    for line in seperated_text:
        line = line.rstrip("\n")

        match = header_re.match(line)
        if match:
            if active_user is not None:
                messages_by_user.setdefault(active_user, []).append("\n".join(active_message))

            active_user = match.group(2)
            first_line = match.group(3)
            active_message = [first_line]
        else:
            if active_user is not None:
                active_message.append(line)

    if active_user is not None:
        messages_by_user.setdefault(active_user, []).append(
            "\n".join(active_message)
        )

    for user, messages in messages_by_user.items():
        name = re.sub(r'[\\/*?:"<>|]', "_", user)
        output_file = Path(members) / f"{name}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(msg + "\n\n")

    work_dir = Path('/Users/piyushkumar/Documents/Shipathon/users')
    curr_dir = Path('/Users/piyushkumar/Documents/Shipathon')
    house_dir = Path('/Users/piyushkumar/Documents/Shipathon/houses')
    houses  = [house for house in os.listdir(house_dir) if (house_dir/house).is_file()] 
    members = [member for member in os.listdir(work_dir) if (work_dir/member).is_file()]
    house_dist = {}
    for single in members:
        dic_scores = {}
        for house in houses:
            file = open(work_dir/single, 'r')
            text = file.read()
            house_file = open(house_dir/house, 'r')
            house_content = house_file.read()
            house_file.close()
            file.close()
            embedding = model.encode(text)
            comparison = model.encode(house_content)
            cosine_score = util.cos_sim(comparison, embedding)
            dic_scores[cosine_score] = Path(house).stem
            answer_score = max(dic_scores.keys())
    
        only_name = Path(single).stem
        house_dist[only_name] = dic_scores[answer_score]
    return house_dist

if chat_file is not None:

    with open(chat_path, 'w') as f:
        f.write(chat_file.getbuffer().tobytes().decode('utf-8'))
    with open(chat_path, 'r') as text_obj:
        text_string = text_obj.read()
    if st.button("run analysis"):
        with st.spinner("PROCESSING:...."):
            result = analysis(text_string)

        st.subheader('Results:')
        st.subheader("🏆 Sorting Results")
        cols = st.columns(4)
        house_emojis = {"Gryffindor": "🦁","Slytherin": "🐍","Ravenclaw": "🦅","Hufflepuff": "🦡"}
        count = 0
        for user, house in result.items():
            with cols[count % 4]:
                st.markdown(f"""
                ### {house_emojis.get(house, '🏰')} {user}
                **House:** {house}, {house_emojis[house.capitalize()]}
                """)
            count += 1
