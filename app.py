import os
import json
import sqlite3
import smtplib
from email.message import EmailMessage

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO


DB_PATH = "missing_pet.db"
UPLOAD_DIR = "uploaded_pet_images"
MODEL_PATH = "efficientnetb0_pet_identity_model.keras"
SIMILARITY_THRESHOLD = 0.50

os.makedirs(UPLOAD_DIR, exist_ok=True)


st.set_page_config(
    page_title="Missing Pet Identification System",
    page_icon="🐾",
    layout="wide"
)


@st.cache_resource
def load_models():
    identity_model = tf.keras.models.load_model(MODEL_PATH)
    embedding_model = tf.keras.Model(
        inputs=identity_model.input,
        outputs=identity_model.layers[-2].output
    )
    yolo_model = YOLO("yolov8s.pt")
    return embedding_model, yolo_model


embedding_model, yolo_model = load_models()


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS owners (
            owner_id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_name TEXT,
            phone TEXT,
            email TEXT,
            address TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pets (
            pet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER,
            pet_name TEXT,
            species TEXT,
            breed TEXT,
            color TEXT,
            age TEXT,
            status TEXT DEFAULT 'Registered',
            missing_date TEXT,
            last_seen_location TEXT,
            description TEXT,
            FOREIGN KEY(owner_id) REFERENCES owners(owner_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pet_images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER,
            image_path TEXT,
            FOREIGN KEY(pet_id) REFERENCES pets(pet_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pet_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet_id INTEGER,
            embedding_vector TEXT,
            FOREIGN KEY(pet_id) REFERENCES pets(pet_id)
        )
    """)

    conn.commit()
    conn.close()


init_db()


def crop_pet_with_yolo(image):
    results = yolo_model(image)

    best_box = None
    best_conf = 0

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # COCO class IDs: cat = 15, dog = 16
            if cls_id in [15, 16] and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

    if best_box is None:
        return image

    x1, y1, x2, y2 = best_box.astype(int)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped


def extract_embedding(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    embedding = embedding_model.predict(img_array, verbose=0)[0]
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def save_image(uploaded_file, pet_id):
    image = Image.open(uploaded_file).convert("RGB")
    file_path = os.path.join(UPLOAD_DIR, f"pet_{pet_id}_{uploaded_file.name}")
    image.save(file_path)
    return file_path, image


def register_pet(owner_name, phone, email, address, pet_name, species, breed, color, age, uploaded_images):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO owners(owner_name, phone, email, address)
        VALUES (?, ?, ?, ?)
    """, (owner_name, phone, email, address))

    owner_id = cur.lastrowid

    cur.execute("""
        INSERT INTO pets(owner_id, pet_name, species, breed, color, age, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (owner_id, pet_name, species, breed, color, age, "Registered"))

    pet_id = cur.lastrowid

    for uploaded_file in uploaded_images:
        image_path, image = save_image(uploaded_file, pet_id)

        cur.execute("""
            INSERT INTO pet_images(pet_id, image_path)
            VALUES (?, ?)
        """, (pet_id, image_path))

        cropped_image = crop_pet_with_yolo(image)
        embedding = extract_embedding(cropped_image)

        cur.execute("""
            INSERT INTO pet_embeddings(pet_id, embedding_vector)
            VALUES (?, ?)
        """, (pet_id, json.dumps(embedding.tolist())))

    conn.commit()
    conn.close()


def get_registered_pets():
    conn = get_connection()
    query = """
        SELECT 
            pets.pet_id,
            pets.pet_name,
            pets.species,
            pets.breed,
            pets.color,
            pets.age,
            pets.status,
            owners.owner_name,
            owners.phone,
            owners.email
        FROM pets
        JOIN owners ON pets.owner_id = owners.owner_id
        ORDER BY pets.pet_id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_missing_pets():
    conn = get_connection()
    query = """
        SELECT 
            pets.pet_id,
            pets.pet_name,
            pets.species,
            pets.breed,
            pets.color,
            pets.status,
            pets.last_seen_location,
            owners.owner_name,
            owners.phone,
            owners.email
        FROM pets
        JOIN owners ON pets.owner_id = owners.owner_id
        WHERE pets.status = 'Missing'
        ORDER BY pets.pet_id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def mark_pet_missing(pet_id, missing_date, last_seen_location, description):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE pets
        SET status = 'Missing',
            missing_date = ?,
            last_seen_location = ?,
            description = ?
        WHERE pet_id = ?
    """, (missing_date, last_seen_location, description, pet_id))

    rows_updated = cur.rowcount
    conn.commit()
    conn.close()

    return rows_updated




def search_missing_pet(found_image):
    cropped_image = crop_pet_with_yolo(found_image)
    found_embedding = extract_embedding(cropped_image)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            pets.pet_id,
            pets.pet_name,
            pets.species,
            pets.breed,
            pets.color,
            pets.last_seen_location,
            owners.owner_name,
            owners.phone,
            owners.email,
            pet_embeddings.embedding_vector
        FROM pets
        JOIN owners ON pets.owner_id = owners.owner_id
        JOIN pet_embeddings ON pets.pet_id = pet_embeddings.pet_id
        WHERE pets.status = 'Missing'
    """)

    rows = cur.fetchall()
    conn.close()

    best_matches = {}

    for row in rows:
        pet_id = row[0]
        stored_embedding = np.array(json.loads(row[9]))
        similarity = float(np.dot(found_embedding, stored_embedding))

        if pet_id not in best_matches or similarity > best_matches[pet_id]["similarity"]:
            best_matches[pet_id] = {
                "pet_id": row[0],
                "pet_name": row[1],
                "species": row[2],
                "breed": row[3],
                "color": row[4],
                "last_seen_location": row[5],
                "owner_name": row[6],
                "phone": row[7],
                "email": row[8],
                "similarity": similarity
            }

    matches = list(best_matches.values())
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)

    return matches[:3]


def send_owner_notification(owner_email, owner_name, pet_name, similarity_score,
                            found_location, finder_name, finder_phone):
    try:
        sender_email = st.secrets["email"]["sender_email"]
        app_password = st.secrets["email"]["app_password"]

        subject = "Possible Match Found for Your Missing Pet"

        body = f"""
Hello {owner_name},

A possible match has been found for your missing pet.

Pet Name: {pet_name}
Matching Confidence: {similarity_score:.2f}%

Found Location:
{found_location}

Finder Details:
Name: {finder_name}
Phone: {finder_phone}

Please contact the finder and verify the pet carefully.

Regards,
Missing Pet Identification System
"""

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = owner_email
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return True

    except Exception as e:
        st.error(f"Notification failed: {e}")
        return False


st.title("Missing Pet Identification System")

menu = st.sidebar.radio(
    "Navigation",
    ["Register Pet", "Mark Pet as Missing", "Search Found Pet", "Registered Pets"]
)


if menu == "Register Pet":
    st.header("Register Pet")

    owner_name = st.text_input("Owner Name")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email")
    address = st.text_area("Address")

    pet_name = st.text_input("Pet Name")
    species = st.selectbox("Species", ["Dog", "Cat"])
    breed = st.text_input("Breed")
    color = st.text_input("Color")
    age = st.text_input("Age")

    uploaded_images = st.file_uploader(
        "Upload Pet Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Register Pet"):
        if owner_name and phone and email and pet_name and uploaded_images:
            register_pet(
                owner_name, phone, email, address,
                pet_name, species, breed, color, age,
                uploaded_images
            )
            st.success("Pet registered successfully.")
        else:
            st.warning("Please fill required details and upload at least one image.")


elif menu == "Mark Pet as Missing":
    st.header("Mark Pet as Missing")

    pets_df = get_registered_pets()

    if pets_df.empty:
        st.info("No registered pets found.")
    else:
        st.dataframe(pets_df)

        pet_options = {
            f'{row["pet_id"]} - {row["pet_name"]} - {row["owner_name"]}': row["pet_id"]
            for _, row in pets_df.iterrows()
        }

        selected_pet = st.selectbox("Select Pet", list(pet_options.keys()))
        pet_id = pet_options[selected_pet]

        missing_date = st.date_input("Missing Date")
        last_seen_location = st.text_input("Last Seen Location")
        description = st.text_area("Description")

        if st.button("Mark as Missing"):
            rows_updated = mark_pet_missing(
                pet_id,
                str(missing_date),
                last_seen_location,
                description
            )

            if rows_updated > 0:
                st.success("Pet status changed to Missing.")
            else:
                st.error("Could not update pet status.")

        st.subheader("Currently Missing Pets")
        missing_df = get_missing_pets()

        if missing_df.empty:
            st.info("No pets are currently marked as missing.")
        else:
            st.dataframe(missing_df)


elif menu == "Search Found Pet":
    st.header("Search Found Pet")

    found_location = st.text_input("Found Location")
    finder_name = st.text_input("Finder Name")
    finder_phone = st.text_input("Finder Phone Number")

    uploaded_found_image = st.file_uploader(
        "Upload Found Pet Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_found_image:
        found_image = Image.open(uploaded_found_image).convert("RGB")
        st.image(found_image, caption="Uploaded Found Pet Image", width=350)

        if st.button("Search Missing Pets"):
            if not found_location or not finder_name or not finder_phone:
                st.warning("Please enter found location, finder name, and finder phone.")
            else:
                matches = search_missing_pet(found_image)

                st.session_state["found_location"] = found_location
                st.session_state["finder_name"] = finder_name
                st.session_state["finder_phone"] = finder_phone
                st.session_state["matches"] = matches

    if "matches" in st.session_state:
        matches = st.session_state["matches"]

        if len(matches) == 0:
            st.warning("No missing pets found in the database.")
        else:
            best_match = matches[0]
            best_score = best_match["similarity"]

            if best_score >= SIMILARITY_THRESHOLD:
                st.success("Possible matching missing pet found.")

                st.subheader("Best Match")
                st.write("Pet Name:", best_match["pet_name"])
                st.write("Species:", best_match["species"])
                st.write("Breed:", best_match["breed"])
                st.write("Color:", best_match["color"])
                st.write("Similarity:", round(best_score * 100, 2), "%")

                st.subheader("Owner Contact")
                st.write("Owner Name:", best_match["owner_name"])
                st.write("Phone:", best_match["phone"])
                st.write("Email:", best_match["email"])
                st.write("Last Seen Location:", best_match["last_seen_location"])

                if st.button("Send Email Notification to Owner"):
                    sent = send_owner_notification(
                        owner_email=best_match["email"],
                        owner_name=best_match["owner_name"],
                        pet_name=best_match["pet_name"],
                        similarity_score=best_score * 100,
                        found_location=st.session_state["found_location"],
                        finder_name=st.session_state["finder_name"],
                        finder_phone=st.session_state["finder_phone"]
                    )

                    if sent:
                        st.success("Email notification sent to owner successfully.")
            else:
                st.warning("No strong match found above the threshold.")

            st.subheader("Top Matches")
            for match in matches:
                st.write(
                    match["pet_name"],
                    "- Similarity:",
                    round(match["similarity"] * 100, 2),
                    "%"
                )


elif menu == "Registered Pets":
    st.header("Registered Pets")

    pets_df = get_registered_pets()

    if pets_df.empty:
        st.info("No pets registered yet.")
    else:
        st.dataframe(pets_df)
