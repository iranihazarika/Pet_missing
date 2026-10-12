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

# Final score = weighted fusion of deep embedding + colour histogram + texture.
# Weights should sum to 1.0. Tune them on your own validation images.
W_DEEP = 0.60
W_COLOR = 0.20
W_TEXTURE = 0.20

# Re-tune this threshold on your own data.
SIMILARITY_THRESHOLD = 0.60

# Colour histogram: HSV with 8 hue x 4 saturation x 4 value bins = 128 dims
HSV_BINS = (8, 4, 4)

# Texture: uniform LBP (10 codes) over a 2x2 spatial grid = 40 dims
LBP_CODES = 10
LBP_GRID = 2

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

    # Migration: add colour/texture columns to existing databases
    cur.execute("PRAGMA table_info(pet_embeddings)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "color_hist" not in existing_cols:
        cur.execute("ALTER TABLE pet_embeddings ADD COLUMN color_hist TEXT")
    if "texture_feature" not in existing_cols:
        cur.execute("ALTER TABLE pet_embeddings ADD COLUMN texture_feature TEXT")

    conn.commit()
    conn.close()


init_db()


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def crop_pet_with_yolo(image):
    results = yolo_model(image, verbose=False)

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
    return image.crop((x1, y1, x2, y2))


def extract_embedding(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    embedding = embedding_model.predict(img_array, verbose=0)[0]
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def extract_color_histogram(image):
    """Normalised 3-D HSV histogram (8x4x4 = 128 dims, sums to 1)."""
    image = image.convert("RGB").resize((128, 128))
    hsv = np.array(image.convert("HSV")).reshape(-1, 3).astype(np.float64)

    hist, _ = np.histogramdd(
        hsv,
        bins=HSV_BINS,
        range=((0, 256), (0, 256), (0, 256))
    )
    hist = hist.flatten()
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def extract_texture_feature(image):
    """
    Rotation-invariant uniform LBP histogram (10 codes), computed on a
    LBP_GRID x LBP_GRID grid. Output: 4 x 10 = 40 dims.
    """
    gray = np.array(image.convert("L").resize((128, 128)), dtype=np.float32)

    center = gray[1:-1, 1:-1]
    h, w = center.shape

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = np.stack([
        (gray[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] >= center).astype(np.uint8)
        for dy, dx in offsets
    ])

    transitions = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    ones = bits.sum(axis=0)
    lbp = np.where(transitions <= 2, ones, 9).astype(np.int32)

    cell_h, cell_w = h // LBP_GRID, w // LBP_GRID
    features = []
    for i in range(LBP_GRID):
        for j in range(LBP_GRID):
            cell = lbp[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            hist = np.bincount(cell.ravel(), minlength=LBP_CODES).astype(np.float64)
            hist = hist / max(hist.sum(), 1.0)
            features.append(hist)

    return np.concatenate(features)


def extract_all_features(image):
    """Crop with YOLO once and return all three feature types."""
    cropped = crop_pet_with_yolo(image)
    return {
        "embedding": extract_embedding(cropped),
        "color": extract_color_histogram(cropped),
        "texture": extract_texture_feature(cropped),
    }


def color_similarity(h1, h2):
    """Bhattacharyya coefficient between two normalised histograms (0..1)."""
    return float(np.sum(np.sqrt(h1 * h2)))


def texture_similarity(t1, t2):
    """Mean histogram intersection over the LBP grid cells (0..1)."""
    t1 = t1.reshape(-1, LBP_CODES)
    t2 = t2.reshape(-1, LBP_CODES)
    return float(np.minimum(t1, t2).sum(axis=1).mean())


def backfill_missing_features():
    """One-time upgrade for pets registered before colour/texture existed."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT pet_id FROM pet_embeddings
        WHERE color_hist IS NULL OR texture_feature IS NULL
    """)
    pet_ids = [r[0] for r in cur.fetchall()]

    for pet_id in pet_ids:
        cur.execute(
            "SELECT embedding_id FROM pet_embeddings WHERE pet_id = ? ORDER BY embedding_id",
            (pet_id,)
        )
        emb_ids = [r[0] for r in cur.fetchall()]

        cur.execute(
            "SELECT image_path FROM pet_images WHERE pet_id = ? ORDER BY image_id",
            (pet_id,)
        )
        paths = [r[0] for r in cur.fetchall()]

        for emb_id, path in zip(emb_ids, paths):
            try:
                image = Image.open(path).convert("RGB")
                cropped = crop_pet_with_yolo(image)
                cur.execute("""
                    UPDATE pet_embeddings
                    SET color_hist = ?, texture_feature = ?
                    WHERE embedding_id = ?
                """, (
                    json.dumps(extract_color_histogram(cropped).tolist()),
                    json.dumps(extract_texture_feature(cropped).tolist()),
                    emb_id
                ))
            except Exception:
                continue

    conn.commit()
    conn.close()


backfill_missing_features()


# ----------------------------------------------------------------------------
# Registration / DB helpers
# ----------------------------------------------------------------------------
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

        feats = extract_all_features(image)

        cur.execute("""
            INSERT INTO pet_embeddings(pet_id, embedding_vector, color_hist, texture_feature)
            VALUES (?, ?, ?, ?)
        """, (
            pet_id,
            json.dumps(feats["embedding"].tolist()),
            json.dumps(feats["color"].tolist()),
            json.dumps(feats["texture"].tolist())
        ))

    conn.commit()
    conn.close()


def get_registered_pets():
    conn = get_connection()
    query = """
        SELECT 
            pets.pet_id, pets.pet_name, pets.species, pets.breed, pets.color,
            pets.age, pets.status, owners.owner_name, owners.phone, owners.email
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
            pets.pet_id, pets.pet_name, pets.species, pets.breed, pets.color,
            pets.status, pets.last_seen_location,
            owners.owner_name, owners.phone, owners.email
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


# ----------------------------------------------------------------------------
# Multi-image matching
# ----------------------------------------------------------------------------
def similarity_matrices(found_feats, stored_feats):
    """
    Pairwise similarity of every found image (rows) against every stored
    image of one registered pet (columns).

    Returns (deep, color, texture, fused), each of shape (n_found, n_stored).
    Colour/texture entries are NaN where the stored record has no such
    feature; the fused score then renormalises weights over what is available.
    """
    n, m = len(found_feats), len(stored_feats)
    deep = np.zeros((n, m))
    color = np.full((n, m), np.nan)
    texture = np.full((n, m), np.nan)

    for i, f in enumerate(found_feats):
        for j, s in enumerate(stored_feats):
            deep[i, j] = float(np.dot(f["embedding"], s["embedding"]))
            if s["color"] is not None:
                color[i, j] = color_similarity(f["color"], s["color"])
            if s["texture"] is not None:
                texture[i, j] = texture_similarity(f["texture"], s["texture"])

    mc = ~np.isnan(color)
    mt = ~np.isnan(texture)
    num = (W_DEEP * deep
           + W_COLOR * np.where(mc, color, 0.0)
           + W_TEXTURE * np.where(mt, texture, 0.0))
    den = W_DEEP + W_COLOR * mc + W_TEXTURE * mt
    fused = num / den

    return deep, color, texture, fused


def aggregate_matrix(mat):
    """
    Symmetric best-match aggregation over an (n_found x n_stored) matrix:
      - for each found image, take its best-matching stored image, then average
      - for each stored image, take its best-matching found image, then average
      - return the mean of the two directions.
    This rewards a pet whose photos are collectively well covered by the found
    photos (and vice versa) while ignoring unrelated angles/poses.
    Columns that are entirely NaN (missing feature) are skipped.
    Returns None if nothing is available.
    """
    valid_cols = ~np.isnan(mat).all(axis=0)
    mat = mat[:, valid_cols]
    if mat.size == 0:
        return None
    found_to_stored = mat.max(axis=1).mean()
    stored_to_found = mat.max(axis=0).mean()
    return float((found_to_stored + stored_to_found) / 2.0)


def search_missing_pet(found_images):
    """found_images: list of PIL images of the found pet."""
    found_feats = [extract_all_features(img) for img in found_images]

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            pets.pet_id, pets.pet_name, pets.species, pets.breed, pets.color,
            pets.last_seen_location,
            owners.owner_name, owners.phone, owners.email,
            pet_embeddings.embedding_vector,
            pet_embeddings.color_hist,
            pet_embeddings.texture_feature
        FROM pets
        JOIN owners ON pets.owner_id = owners.owner_id
        JOIN pet_embeddings ON pets.pet_id = pet_embeddings.pet_id
        WHERE pets.status = 'Missing'
        ORDER BY pets.pet_id, pet_embeddings.embedding_id
    """)
    rows = cur.fetchall()

    # Group all stored images/embeddings per missing pet
    pets = {}
    for row in rows:
        pid = row[0]
        if pid not in pets:
            pets[pid] = {
                "info": {
                    "pet_id": row[0], "pet_name": row[1], "species": row[2],
                    "breed": row[3], "color": row[4], "last_seen_location": row[5],
                    "owner_name": row[6], "phone": row[7], "email": row[8],
                },
                "feats": [],
            }
        pets[pid]["feats"].append({
            "embedding": np.array(json.loads(row[9])),
            "color": np.array(json.loads(row[10])) if row[10] else None,
            "texture": np.array(json.loads(row[11])) if row[11] else None,
        })

    # Stored image paths (same order as embeddings) so the best pair can be shown
    for pid, data in pets.items():
        cur.execute(
            "SELECT image_path FROM pet_images WHERE pet_id = ? ORDER BY image_id",
            (pid,)
        )
        data["paths"] = [r[0] for r in cur.fetchall()]

    conn.close()

    matches = []
    for pid, data in pets.items():
        deep, color, texture, fused = similarity_matrices(found_feats, data["feats"])

        # Best single pair (for display)
        bi, bj = np.unravel_index(np.argmax(fused), fused.shape)
        best_path = data["paths"][bj] if bj < len(data["paths"]) else None

        match = dict(data["info"])
        match.update({
            "similarity": aggregate_matrix(fused),
            "deep_sim": aggregate_matrix(deep),
            "color_sim": aggregate_matrix(color),
            "texture_sim": aggregate_matrix(texture),
            "best_pair_score": float(fused[bi, bj]),
            "best_found_idx": int(bi),
            "best_stored_path": best_path,
            "n_found_images": len(found_feats),
            "n_stored_images": len(data["feats"]),
            "pair_matrix": fused.tolist(),
        })
        matches.append(match)

    matches.sort(key=lambda x: x["similarity"], reverse=True)
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


def fmt_pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
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
        "Upload Pet Images (multiple angles recommended)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Register Pet"):
        if owner_name and phone and email and pet_name and uploaded_images:
            with st.spinner("Registering pet and extracting features..."):
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
                pet_id, str(missing_date), last_seen_location, description
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

    uploaded_found_images = st.file_uploader(
        "Upload Found Pet Images (one or more)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    found_images = []
    if uploaded_found_images:
        found_images = [Image.open(f).convert("RGB") for f in uploaded_found_images]

        st.caption(f"{len(found_images)} image(s) uploaded")
        cols = st.columns(min(len(found_images), 4))
        for idx, img in enumerate(found_images):
            cols[idx % len(cols)].image(img, caption=f"Found image {idx + 1}", width=200)

        if st.button("Search Missing Pets"):
            if not found_location or not finder_name or not finder_phone:
                st.warning("Please enter found location, finder name, and finder phone.")
            else:
                with st.spinner("Comparing all found images against registered pets..."):
                    matches = search_missing_pet(found_images)

                st.session_state["found_location"] = found_location
                st.session_state["finder_name"] = finder_name
                st.session_state["finder_phone"] = finder_phone
                st.session_state["matches"] = matches
                st.session_state["found_images_preview"] = found_images

    if "matches" in st.session_state:
        matches = st.session_state["matches"]
        preview_images = st.session_state.get("found_images_preview", [])

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
                st.write("Overall Similarity (all images):", round(best_score * 100, 2), "%")
                st.write(
                    f'Compared {best_match["n_found_images"]} found image(s) with '
                    f'{best_match["n_stored_images"]} registered image(s). '
                    f'Best single-pair score: {best_match["best_pair_score"] * 100:.1f}%'
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Deep Features", fmt_pct(best_match["deep_sim"]))
                c2.metric("Colour Histogram", fmt_pct(best_match["color_sim"]))
                c3.metric("Texture (LBP)", fmt_pct(best_match["texture_sim"]))

                # Show the most similar pair side by side
                st.markdown("**Most similar image pair**")
                p1, p2 = st.columns(2)
                if best_match["best_found_idx"] < len(preview_images):
                    p1.image(
                        preview_images[best_match["best_found_idx"]],
                        caption="Found pet", width=300
                    )
                if best_match["best_stored_path"] and os.path.exists(best_match["best_stored_path"]):
                    p2.image(
                        best_match["best_stored_path"],
                        caption="Registered pet", width=300
                    )

                with st.expander("Pairwise similarity matrix (found x registered)"):
                    mat = np.array(best_match["pair_matrix"]) * 100
                    st.dataframe(pd.DataFrame(
                        mat,
                        index=[f"Found {i + 1}" for i in range(mat.shape[0])],
                        columns=[f"Registered {j + 1}" for j in range(mat.shape[1])]
                    ).round(1))

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
                    f'{match["pet_name"]} - Overall: {match["similarity"] * 100:.2f}% '
                    f'(deep {fmt_pct(match["deep_sim"])}, '
                    f'colour {fmt_pct(match["color_sim"])}, '
                    f'texture {fmt_pct(match["texture_sim"])}; '
                    f'best pair {match["best_pair_score"] * 100:.1f}%, '
                    f'{match["n_found_images"]} found vs {match["n_stored_images"]} registered images)'
                )


elif menu == "Registered Pets":
    st.header("Registered Pets")

    pets_df = get_registered_pets()

    if pets_df.empty:
        st.info("No pets registered yet.")
    else:
        st.dataframe(pets_df)
