import streamlit as st
st.set_page_config(page_title="Music AI Website", layout="wide")
import os
import bcrypt
import re  # Thêm thư viện kiểm tra email hợp lệ
from openai import OpenAI
import numpy as np
import base64
import pytube
import os
import subprocess 
import librosa
import tempfile 
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import tensorflow as tf
from statistics import mode
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation)
from streamlit_option_menu import option_menu
import time
from dotenv import load_dotenv
from supabase import create_client, Client
import requests  # Dùng để gửi yêu cầu API
import asyncio 
import streamlit.components.v1 as components    
from auth import register_user
import streamlit as st
from streamlit_cookies_manager import CookieManager
import base64
import logging
from chatbot import display_chatbot 
import streamlit as st
from streamlit_toggle_switch import st_toggle_switch
import time
import requests
import hmac
import hashlib
import uuid
import pandas as pd
from datetime import datetime, timedelta


# Load API key từ file .env
load_dotenv()
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
api_token = os.getenv("SUNO_API_TOKEN")

# Kết nối Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(os.path.exists("D:/test/Music-Genre-Recognition-main/.streamlit/secrets.toml"))

# Cấu hình logging - Lưu các lỗi vào file 'app.log'
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(message)s')

# Hàm ghi lỗi vào log
def log_error(message):
    """Ghi lỗi vào file log và hiển thị thông báo lỗi cho người dùng."""
    logging.error(message)  # Ghi lỗi vào file log
    st.error(f"🚨 Lỗi xảy ra: {message}")  # Hiển thị lỗi cho người dùng

def generate_lyrics(prompt):
    """Gửi prompt đến OpenAI API để tạo lời bài hát"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Hoặc "gpt-3.5-turbo" nếu tài khoản không có quyền truy cập GPT-4
            messages=[
                {"role": "system", "content": "Bạn là một nhạc sĩ sáng tác lời bài hát chuyên nghiệp."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=900
        )

        # ✅ Lấy nội dung phản hồi đúng cách
        return response.choices[0].message.content  

    except Exception as e:
        return f"⚠️ Lỗi khi tạo lời bài hát: {str(e)}"

# CSS nâng cao cho giao diện
st.markdown(
    """
    <style>
        /* Thiết lập nền và font chữ chung */
        body, .stApp {
            # background: linear-gradient(135deg, #0E0808 0%, #1A1A1A 100%) !important;
            background: url("https://t4.ftcdn.net/jpg/01/28/98/53/360_F_128985367_mNdX0b56w6OcgiyUDnZwukpK1bkpfWwI.jpg") no-repeat center center fixed;
            background-size: cover !important;
            font-family: 'Roboto', sans-serif;
            color: #FFFFFF;
        }

        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #ff7e5f, #feb47b);
            border-radius: 10px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: rgba(10, 10, 10, 0.8) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 5px 0 15px rgba(0, 0, 0, 0.2);
        }
        [data-testid="stSidebar"] .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* Header styles */
        h1, h2, h3 {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        h2 {
            font-size: 1.8rem;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        h3 {
            font-size: 1.4rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 126, 95, 0.4);
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 7px 15px rgba(255, 126, 95, 0.6);
            background: linear-gradient(90deg, #feb47b, #ff7e5f);
        }
        .stButton > button:active {
            transform: translateY(1px);
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background-color: rgba(30, 30, 30, 0.6) !important;
            border: 1px solid rgba(255, 126, 95, 0.3) !important;
            border-radius: 8px !important;
            color: white !important;
            padding: 12px !important;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #ff7e5f !important;
            box-shadow: 0 0 0 2px rgba(255, 126, 95, 0.2) !important;
        }
        
        /* File uploader */
        .stFileUploader > div > button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            color: white;
        }
        .stFileUploader > div {
            border: 2px dashed rgba(255, 126, 95, 0.5);
            border-radius: 10px;
            padding: 20px;
        }
        
        /* Audio player */
        audio {
            width: 100%;
            border-radius: 30px;
            background-color: rgba(40, 40, 40, 0.8);
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.4);
        }
        audio::-webkit-media-controls-panel {
            background: linear-gradient(45deg, #333333, #1A1A1A);
        }
        audio::-webkit-media-controls-play-button {
            background-color: #ff7e5f;
            border-radius: 50%;
        }
        audio::-webkit-media-controls-timeline,
        audio::-webkit-media-controls-volume-slider {
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            border-radius: 15px;
            height: 4px;
        }
        
        /* Music card styling */
        .music-card {
            background: rgba(30, 30, 30, 0.7);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #ff7e5f;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .music-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }
        
        /* Toggle switch */
        .css-16h7emx {
            color: rgba(250, 250, 250, 0.8) !important;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio > div[role="radiogroup"] > label,
        .stCheckbox > label {
            color: white !important;
        }
        
        /* Loading spinner */
        .stSpinner > div {
            border-top-color: #ff7e5f !important;
        }
        
        /* Section dividers */
        hr {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 126, 95, 0.5), transparent);
            margin: 2rem 0;
        }
        
        /* Status messages */
        .stAlert {
            background-color: rgba(30, 30, 30, 0.7) !important;
            border-left: 4px solid;
            border-radius: 8px;
        }
        .element-container:has(.stAlert) {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Menu styling for option menu */
        .nav-link {
            margin: 5px 0 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        .nav-link:hover {
            background-color: rgba(255, 126, 95, 0.2) !important;
        }
        .nav-link-selected {
            background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
            box-shadow: 0 4px 10px rgba(255, 126, 95, 0.4) !important;
        }
        
        /* Custom animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .pulse-effect {
            animation: pulse 2s infinite;
        }
        
        /* Custom containers for sections */
        .custom-container {
            background: rgba(30, 30, 30, 0.7);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255, 126, 95, 0.2);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            background: linear-gradient(90deg, #ff7e5f, #feb47b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        [data-testid="stMetricLabel"] {
            color: rgba(255, 255, 255, 0.8) !important;
        }
        
        /* Info box */
        .info-box {
            background: rgba(255, 126, 95, 0.1);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #ff7e5f;
            margin: 15px 0;
        }
        
        /* Glassmorphism elements */
        .glass-effect {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Hàm mã hóa email
def encode_email(email):
    return base64.b64encode(email.encode()).decode()

# Hàm giải mã email
def decode_email(encoded):
    try:
        return base64.b64decode(encoded.encode()).decode()
    except Exception:
        return None

# SIDEBAR NAVIGATION
with st.sidebar:
    st.image("a-minimalist-logo-design-on-a-black-back.jpeg", use_container_width=True)

    # Xử lý đăng nhập với cookie
    cookies = CookieManager()

    # Kiểm tra cookies có sẵn và đã mã hóa email
    if cookies.ready() and cookies.get("user_email") and "user" not in st.session_state:
        decoded_email = decode_email(cookies.get("user_email"))
        if decoded_email:
            # 👉 Gọi Supabase để lấy thông tin đầy đủ từ email
            profile_data = supabase.table("user_profiles").select("*").eq("email", decoded_email).execute()
            if profile_data.data:
                profile = profile_data.data[0]
                st.session_state["user"] = {
                    "id": profile["id"],
                    "email": profile["email"],
                    "full_name": profile.get("full_name", ""),
                    "role": profile.get("role", "client"),
                    "created_at": profile.get("created_at", "")
                }

    # KHOẢNG TAI KHOẢN (AUTH)
    if "user" not in st.session_state:
        st.markdown("""
            <div class="custom-container" style="padding: 15px; margin-bottom: 20px;">
                <h3 style="margin-top: 0; font-size: 18px; text-align: center;">
                    🔐 Tài khoản
                </h3>
        """, unsafe_allow_html=True)
        
        auth_menu = st.radio("", ["Đăng nhập", "Đăng ký", "Quên mật khẩu"], horizontal=True, label_visibility="collapsed")
        
        if auth_menu == "Đăng ký":
            st.markdown('<p style="font-weight: 600; font-size: 16px; margin-bottom: 10px;">✍️ Đăng ký tài khoản</p>', unsafe_allow_html=True)
            
            email = st.text_input("Email", type="default", placeholder="your.email@example.com")
            password = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
            full_name = st.text_input("Họ tên", placeholder="Nhập họ và tên của bạn")
            
            if st.button("🚀 Đăng ký"):
                from auth import register_user
                success, msg = register_user(email, password, full_name)
                if success:
                    st.success(msg)
                    st.info("📧 Vui lòng kiểm tra hộp thư để xác minh tài khoản trước khi đăng nhập.")
                else:
                    st.error(msg)

        elif auth_menu == "Đăng nhập":
            st.markdown('<p style="font-weight: 600; font-size: 16px; margin-bottom: 10px;">🔑 Đăng nhập</p>', unsafe_allow_html=True)
            
            email = st.text_input("Email đăng nhập", placeholder="your.email@example.com")
            password = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
            
            if st.button("🔓 Đăng nhập"):
                from auth import login_user
                success, msg = login_user(email, password)
                if success:
                    cookies["user_email"] = encode_email(email)
                    cookies["user_id"] = st.session_state["user"]["id"]
                    cookies.save()
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        elif auth_menu == "Quên mật khẩu":
            st.markdown('<p style="font-weight: 600; font-size: 16px; margin-bottom: 10px;">📧 Đặt lại mật khẩu</p>', unsafe_allow_html=True)
            
            email = st.text_input("Nhập email đã đăng ký", placeholder="your.email@example.com")
            
            if st.button("Gửi email đặt lại mật khẩu"):
                from auth import supabase
                try:
                    res = supabase.auth.reset_password_for_email(email)
                    st.success("📬 Đã gửi email đặt lại mật khẩu. Vui lòng kiểm tra hộp thư đến.")
                except Exception as e:
                    st.error(f"❌ Lỗi khi gửi email: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # HIỂN THỊ THÔNG TIN NGƯỜI DÙNG ĐÃ ĐĂNG NHẬP
        full_name = st.session_state["user"].get("full_name", "bạn")
        
        # Lấy thông tin credits
        user_id = st.session_state["user"]["id"]
        credit_data = supabase.table("user_credits").select("credits").eq("id", user_id).execute()
        credits = credit_data.data[0]["credits"] if credit_data.data else 0
        
        st.markdown(f"""
            <div class="custom-container" style="padding: 15px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        background: linear-gradient(45deg, #ff7e5f, #feb47b);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 10px;
                        font-weight: bold;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                        ">{full_name[0].upper()}</div>
                    <div>
                        <div style="font-weight: bold;">👋 {full_name}</div>
                        <div style="font-size: 0.9rem; opacity: 0.7;">{st.session_state["user"]["email"]}</div>
                    </div>
                </div>
                
            <div style="
                background: linear-gradient(45deg, rgba(255,126,95,0.2), rgba(254,180,123,0.2));
                padding: 10px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                margin-bottom: 15px;">
                <span style="font-size: 24px; margin-right: 10px;">💎</span>
                <div>
                    <div style="font-size: 0.9rem; opacity: 0.8;">Tín dụng hiện có</div>
                    <div style="font-weight: bold;">{credits:,} credits</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # if st.button("🚪 Đăng xuất", key="logout_button"):
        #     del cookies["user_email"]
        #     del st.session_state['user']
        #     cookies.save()
        #     st.success("✅ Đã đăng xuất.")
        #     st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    menu = option_menu(
        menu_title=None,
        options=["Home", "Create Lyrics", "Feel The Beat", "Classify", "Library", "Payment"],
        icons=["house", "music-note-list", "soundwave", "graph-up", "book", "credit-card"],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {"background-color": "rgba(30,30,30,0.7)", "padding": "10px", "border-radius": "15px"},
            "icon": {"color": "#ff7e5f", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "color": "#ffffff", "text-align": "left", "margin": "5px", "border-radius": "8px", "padding": "10px"},
            "nav-link-selected": {"background": "linear-gradient(90deg, #ff7e5f, #feb47b)"},
        }
    )
    if "user" in st.session_state:
        if st.button("🚪 Đăng xuất", key="logout_button"):
            del cookies["user_email"]
            del st.session_state['user']
            cookies.save()
            st.success("✅ Đã đăng xuất.")
            st.rerun()

        
        # Hiển thị chatbot
        display_chatbot()


# 🚫 Chặn menu nếu chưa đăng nhập
protected_menus = ["Create Lyrics", "Feel The Beat", "Classify", "Explore", "Library","Quản lý thanh toán"]

if menu in protected_menus and "user" not in st.session_state:
    st.markdown("""
        <div class="custom-container" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 60px; margin-bottom: 20px;">🔒</div>
            <h2 style="margin-bottom: 20px;">Vui lòng đăng nhập</h2>
            <p style="margin-bottom: 30px; color: rgba(255,255,255,0.7);">
                Bạn cần đăng nhập để truy cập chức năng này.
            </p>
            <div style="
                background: linear-gradient(45deg, rgba(255,126,95,0.2), rgba(254,180,123,0.2));
                padding: 15px;
                border-radius: 10px;
                max-width: 400px;
                margin: 0 auto;
                ">
                <p>👉 Sử dụng form đăng nhập ở menu bên trái để tiếp tục.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

def handle_empty_title(music_data):
    """Kiểm tra và điền tên bài hát nếu bị rỗng."""
    for song in music_data:
        if isinstance(song, dict):  # Kiểm tra xem song có phải là dictionary không
            # Kiểm tra nếu thiếu audioUrl hoặc imageUrl
            if not song.get('audioUrl'):
                song['audioUrl'] = "https://default-audio-url.com"  # Đặt URL mặc định nếu thiếu audioUrl
            if not song.get('imageUrl'):
                song['imageUrl'] = "https://default-image-url.com"  # Đặt URL mặc định nếu thiếu imageUrl

            # Kiểm tra nếu thiếu title
            if not song.get('title'):
                song['title'] = f"Track {song.get('id', 'Unknown')}"  # Đặt tên mặc định nếu không có title
                log_error(f"Bài hát với ID {song.get('id', 'Unknown')} thiếu title. Đặt tên mặc định.")
        else:
            log_error(f"Dữ liệu bài hát không hợp lệ: {song}")
    return music_data


# =========== TRANG HOME ===========
if menu == "Home":
    # Header Animation and Logo
    st.markdown("""
    <div style="text-align: center; animation: fadeIn 1.5s ease-out;">
        <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; 
                background: linear-gradient(45deg, #ff7e5f, #feb47b, #ff7e5f);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 0px;">
            ASTRONAUT MUSIC
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # HOT IN APRIL SECTION
    st.markdown("<h2 style='text-align: left;'>🔥 Bài Hát Hot Trong Tháng 4</h2>", unsafe_allow_html=True)

    public_songs = supabase.table("songs").select("*").eq("is_public", True).order("created_at", desc=True).execute()
    user_profiles = supabase.table("user_profiles").select("id, full_name").execute()
    user_map = {u["id"]: u["full_name"] for u in user_profiles.data}

    if public_songs.data:
        songs = public_songs.data

        slides_html = ""
        for idx, song in enumerate(songs):
            title = song.get("title", "Untitled")
            artist = user_map.get(song["user_id"], "Ẩn danh")
            image = song.get("image_url", "https://via.placeholder.com/300x180.png?text=No+Cover")
            audio = song.get("audio_url")
            duration = song.get("duration", 0)
            mins, secs = int(duration // 60), int(duration % 60)

            slide = f"""
            <div style='background:#1e1e1e; padding:10px; border-radius:12px; width:200px; color:white; font-family:sans-serif;'>
            
                <div style='position:relative;'>
                    <img src=\"{image}\" style=\"width: 100%; height: 200px; object-fit: cover; border-radius: 20px; background: #000;\" />
                    <div style='position:absolute; top:6px; left:6px; background:#00cc88; color:white; font-size:10px; padding:2px 6px; border-radius:4px;'>v3-5</div>
                    <div style='position:absolute; top:6px; right:6px; background:#333; color:white; font-size:10px; padding:2px 6px; border-radius:4px;'>{mins}:{secs:02}</div>
                    <div onclick='playTrack(\"{title}\", \"{artist}\", \"{audio}\", \"{image}\")' style='position:absolute; bottom:6px; right:6px; background:#ff7e5f; color:white; font-size:11px; padding:6px 10px; border-radius:6px; cursor:pointer;'>▶ Nghe ngay</div>
                </div>
                <div style='margin-top:8px; font-size:13px; font-weight:bold;'>{title}</div>
                <div style='font-size:11px; color:#bbb;'>👤 {artist}</div>
            </div>
            """
            slides_html += slide

        # Grid & Scrollable container
        full_html = f"""
        
        <div style='display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; max-height:750px; overflow-y:auto; padding:5px;'>
            {slides_html}
        </div>

        <div id='musicPlayerPopup' style='
            display:none;
            position:fixed;
            bottom:0;
            left:0;
            width:1000vw;
            background:#181818;
            border-top:1px solid #333;
            box-shadow:0 -2px 10px rgba(0,0,0,0.5);
            color:white;
            z-index:9999;
            padding: 10px 20px;
            transition: all 0.3s ease-in-out;
        '>
            <div style='
                display:flex;
                align-items:center;
                justify-content:space-between;
                gap:20px;
                max-width:100%;
                margin-left: 0;
                margin-right: auto;
                padding-left: 20px;
            '>

                <img id='popupImage' src='' style='width:60px; height:60px; object-fit:cover; border-radius:10px;'>
                <div style="flex-grow:1;">
                    <div id='popupTitle' style='font-size:15px; font-weight:bold;'></div>
                    <div id='popupArtist' style='font-size:13px; color:#ccc;'></div>
                </div>
                <audio id='popupAudio' controls autoplay style='
                    width: 80%;
                    height: 35px;
                    border-radius: 8px;
                    background-color: #222;
                '></audio>

                <button onclick="document.getElementById('musicPlayerPopup').style.display='none'" style="background:none; border:none; color:white; font-size:22px;">×</button>
            </div>
        </div>

        <script>
        function playTrack(title, artist, audioUrl, imageUrl) {{
            document.getElementById('musicPlayerPopup').style.display = 'block';
            document.getElementById('popupTitle').innerText = title;
            document.getElementById('popupArtist').innerText = artist;
            document.getElementById('popupAudio').src = audioUrl;
            document.getElementById('popupImage').src = imageUrl;
        }}
        </script>
        """

        components.html(full_html, height=800)
 

    else:
        st.info("🙈 Chưa có bài hát nào được chia sẻ.")




if menu == "Create Lyrics":
    import pyperclip
    st.markdown("<h1>🎶 AI Lyric Generator 🎵</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 5])
    with col1:
        # Người dùng nhập thể loại nhạc và chủ đề
        genre = st.text_area("🎼 Chọn thể loại nhạc: ",
                            placeholder="Pop, Rock, Hip-Hop, Jazz, Ballad, EDM,....")
        mood = st.text_area("🎭 Chọn cảm xúc: ",
                            placeholder="Vui vẻ, Buồn, Hào hứng, Thư giãn, Kịch ,....")
        theme = st.text_area("✍️ Mô tả bản nhạc bạn muốn tạo:",
                            placeholder="Tình yêu, Mùa thu, Tuổi trẻ, ...")
        if "lyrics_input" in st.session_state:
            lyrics = st.session_state.lyrics_input
        else:
            lyrics = ""
        if st.button("🎤 Sáng tác ngay!"):
            if theme.strip():
                with st.spinner("🎶 AI đang sáng tác lời bài hát cho bạn..."):
                    prompt = f"Hãy viết lời bài hát thể loại {genre} về chủ đề '{theme}', với cảm xúc {mood}."
                    lyrics = generate_lyrics(prompt)
            else:
                st.warning("⚠️ Vui lòng nhập chủ đề bài hát trước khi tạo!")
    with col2:
    # Hiển thị text_area và lưu giá trị trực tiếp vào lyrics    
        lyrics_input = st.text_area("🎼 Lời bài hát AI tạo:", lyrics, height=370)
    # Kiểm tra nếu nội dung text_area thay đổi và tự động sao chép vào clipboard
        st.session_state.lyrics_input = lyrics
    
        if st.button("Copy Lyrics"):
                # pyperclip.copy(lyrics_input)  # Sao chép lyrics vào clipboard
                lyrics = lyrics_input
                st.session_state.lyrics = lyrics
                st.success("Lyrics have been copied to clipboard and Feel The Beat")  # Hiển thị thông báo thành công

    if lyrics_input != lyrics:
        lyrics = lyrics_input
        st.session_state.lyrics_input = lyrics



# Nếu chọn "Classify", hiển thị nội dung này
if menu == "Classify":
    st.markdown("<h1 style='text-align: center; color: white;'>Music Genre Recognition</h1>", unsafe_allow_html=True)

    # Upload file mp3
    st.write("## Upload an MP3 file to classify:")
    mp3_file = st.file_uploader("Upload an audio file", type=["mp3"], label_visibility="collapsed")    
    
    if mp3_file is not None:
        st.write("**Play the song below:**")
        st.audio(mp3_file, "audio/mp3")

        # Hàm chuyển đổi MP3 sang WAV
        def convert_mp3_to_wav(music_file):  
            sound = AudioSegment.from_mp3(music_file)
            sound.export("music_file.wav", format="wav")

        # Hàm tạo Mel Spectrogram
        def create_melspectrogram(wav_file):  
            y, sr = librosa.load(wav_file)  
            mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))    
            plt.figure(figsize=(10, 5))
            plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel_spec, x_axis="time", y_axis='mel', sr=sr)
            plt.margins(0)
            plt.savefig('melspectrogram.png')

        # Xây dựng mô hình CNN
        def GenreModel(input_shape=(100,200,4), classes=10):
            classifier = Sequential()
            classifier.add(Conv2D(8, (3, 3), input_shape=input_shape, activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(16, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(32, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(64, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Conv2D(128, (3, 3), activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(Flatten())
            classifier.add(Dropout(0.5))
            classifier.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
            classifier.add(Dropout(0.25))
            classifier.add(Dense(10, activation='softmax'))
            classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return classifier

        # Dự đoán thể loại nhạc
        def predict(image_data, model):   
            image = img_to_array(image_data)   
            image = np.reshape(image, (1, 100, 200, 4))   
            prediction = model.predict(image / 255)   
            prediction = prediction.reshape((10,))     
            class_label = np.argmax(prediction)     
            return class_label, prediction

        # Nhãn của các thể loại nhạc
        class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        # Load mô hình
        model = GenreModel(input_shape=(100, 200, 4), classes=10)
        model.load_weights("music_genre_recog_model.h5")

        # Hiệu ứng loading
        with st.spinner("🔍 Analyzing music genre..."):
            time.sleep(2)

        # Chuyển đổi file và tạo spectrogram
        convert_mp3_to_wav(mp3_file)
        audio_full = AudioSegment.from_wav('music_file.wav')

        class_labels_total = []
        predictions_total = []
        for w in range(int(round(len(audio_full) / 3000, 0))):
            audio_3sec = audio_full[3 * (w) * 1000: 3 * (w + 1) * 1000]
            audio_3sec.export(out_f="audio_3sec.wav", format="wav")
            create_melspectrogram("audio_3sec.wav")
            image_data = load_img('melspectrogram.png', color_mode='rgba', target_size=(100, 200))   
            class_label, prediction = predict(image_data, model)
            class_labels_total.append(class_label)
            predictions_total.append(prediction)

        # Lấy thể loại có dự đoán cao nhất
        class_label_final = mode(class_labels_total)
        predictions_final = np.mean(predictions_total, axis=0)

        # Hiển thị kết quả
        st.success(f"✅ The genre of your song is: **{class_labels[class_label_final]}**")
        # Hiển thị biểu đồ với nền tối
        # Hiển thị biểu đồ với nền tối
        fig, ax = plt.subplots(figsize=(10, 5))

        # Thiết lập màu nền của biểu đồ
        fig.patch.set_facecolor('#0E0808')  # Màu nền của biểu đồ
        ax.set_facecolor('#0E0808')  # Màu nền của trục

        # Thiết lập màu cho các thanh trong biểu đồ
        ax.bar(class_labels, predictions_final, color=cm.viridis(np.linspace(0, 1, len(class_labels))))

        # Thiết lập các yếu tố hiển thị khác
        ax.set_xlabel("Music Genre", color='white', fontsize=16)  # Màu chữ cho trục X và cỡ chữ
        ax.set_ylabel("Prediction Probability", color='white', fontsize=16)  # Màu chữ cho trục Y và cỡ chữ
        ax.set_title("Genre Prediction Probability Distribution", color='white', fontsize=18)  # Màu chữ cho tiêu đề và cỡ chữ

        # Thiết lập các nhãn trục X với chữ không in đậm và kích thước chữ lớn hơn
        ax.set_xticklabels(class_labels, rotation=45, color='white', fontsize=14)

        # Xóa các đường kẻ ô (gridlines)
        ax.grid(False)

        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(fig)



# Hàm tạo nhạc từ API
async def generate_music(api_token, prompt, custom_mode, style, title, instrumental):
    api_url = "https://apibox.erweima.ai/api/v1/generate"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    
    if custom_mode == True:
        data = {
            "prompt": prompt,
            "style": style,
            "title": title,
            "customMode": custom_mode,
            "instrumental": instrumental,
            "model": "V3_5",
            "callBackUrl": "https://api.example.com/callback"
        }
    else:
        data = {
            "prompt": prompt,
            "customMode": custom_mode,
            "instrumental": instrumental,
            "model": "V3_5",
            "callBackUrl": "https://api.example.com/callback"
        }

    with st.spinner("🎼 Đang tạo nhạc..."):
        response = await asyncio.to_thread(requests.post, api_url, json=data, headers=headers)
    # Kiểm tra mã trạng thái của phản hồi từ API
    if response.status_code == 200:
        try:
            response_json = response.json()  # Cố gắng phân tích dữ liệu JSON từ phản hồi

            # Kiểm tra nếu 'data' có tồn tại trong phản hồi
            data = response_json.get("data")  # Sử dụng .get() để tránh lỗi nếu 'data' không tồn tại

            if data is not None:
                task_id = data.get("taskId")  # Lấy taskId từ 'data'
                if task_id:
                    return task_id
                else:
                    st.error("🚨 Không tìm thấy taskId trong phản hồi!")
            else:
                st.error("🚨 Không có dữ liệu 'data' trong phản hồi API!")
                st.write("📄 Nội dung API trả về:", response.text)
        except ValueError as e:
            st.error(f"🚨 Lỗi khi phân tích JSON từ API: {e}")
            st.write("📄 Nội dung API trả về:", response.text)
    else:
        st.error(f"🚨 API trả về lỗi: {response.status_code}")
        st.write("📄 Nội dung lỗi:", response.text)
    return None

# Hàm kiểm tra và hiển thị nhạc
async def check_music_status(api_token, task_id):
    check_url = f"https://apibox.erweima.ai/api/v1/generate/record-info?taskId={task_id}"
    headers = {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}
    # Truy vấn user_id từ bảng user_profiles bằng email
    if "user" in st.session_state and "email" in st.session_state["user"]:
        user_email = st.session_state["user"]["email"]  # Lấy email từ session

        # Truy vấn user_id từ bảng user_profiles
        user_profile = supabase.table("user_profiles").select("id").eq("email", user_email).execute()

        if user_profile.data:
            user_id = user_profile.data[0]["id"]  # Lấy user_id từ profile
    else:
        st.error("❌ Không tìm thấy thông tin người dùng.")
        return None
        
    for _ in range(60):  # Lặp tối đa 60 lần (5 phút)
        check_response = await asyncio.to_thread(requests.get, check_url, headers=headers)

        if check_response.status_code == 200:
            try:
                music_info = check_response.json()
                data = music_info.get("data", {})
                status = data.get("status", "PENDING")
                # st.write("🛠️ Trạng thái từ API:", status)
                # st.write("📄 Full dữ liệu API trả về:", data)
                if status == "SUCCESS":
                    suno_data = data.get("response", {}).get("sunoData", [])
                    if suno_data:

                        # Lưu bài hát vào cơ sở dữ liệu (bảng songs)
                        for song in suno_data:
                            song_data = {
                                #"user_id": st.session_state["user"]["id"],  # Liên kết với user_id
                                "user_id": user_id,  # Liên kết với user_id từ bảng user_profiles
                                "title": song.get("title"),
                                "audio_url": song.get("audioUrl"),
                                "image_url": song.get("imageUrl"),
                                "prompt": song.get("prompt"),
                                "model_name": song.get("modelName"),
                                "duration": song.get("duration")
                            }
                            # Lưu vào bảng songs trong Supabase
                            supabase.table("songs").insert(song_data).execute()

                        return [(item.get("audioUrl"), item.get("title"), item.get("imageUrl")) for item in suno_data]
            except ValueError as e:
                st.error(f"🚨 Lỗi khi phân tích JSON từ API: {e}")
                st.write("📄 Nội dung API trả về:", check_response.text)
                break
        else:
            st.error(f"🚨 Lỗi khi kiểm tra nhạc: {check_response.status_code}")
            break
        time.sleep(5)  # Chờ 5 giây trước khi kiểm tra lại
    return None

def render_music_player(title, audio_url, image_url):
    """
    Displays the music player interface with title, cover art and music player.
    """
    st.markdown(
        """
        <style>
            .audio-container {
                text-align: left;
                padding: 20px;
                position: relative;
            }
            audio {
                width: 100%;
                border: 4px solid #ff7e5f;
                border-radius: 30px;
                box-shadow: 0px 0px 15px #feb47b;
            }
            audio::-webkit-media-controls-timeline {
                background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
                border-radius: 30px;
                height: 6px;
                box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                transition: all 0.3s ease-in-out;
                padding: 1px;
            }
            audio::-webkit-media-controls-play-button {
                background-color: #ff7e5f !important;
                box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                border-radius: 50%;
            }
            audio::-webkit-media-controls-volume-slider {
                background: linear-gradient(90deg, #ff7e5f, #feb47b) !important;
                border-radius: 30px;
                height: 6px;
                box-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
                transition: all 0.3s ease-in-out;
                margin-top: 11px;
                padding-top:1px;
                padding-bottom:1px;
            }
            .song-title {
                font-size: 20px;
                font-weight: bold;
                color: white;
                text-align: left;
                margin-top: 10px;
                text-shadow: 0px 0px 10px rgba(255, 126, 95, 0.8);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(image_url, width=150)
    with col2:
        st.markdown(f'<div class="song-title">{title}</div>', unsafe_allow_html=True)
        st.audio(audio_url, format="audio/mp3")


# Hàm hiển thị trò chơi chờ nhạc
def render_game_html():
    game_html = """
    <iframe src="https://chromedino.com/color/" frameborder="0" scrolling="no" width="100%" height="100%" loading="lazy"></iframe>
    <div style="
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0E1117; /* Màu nền */
        color: #FFA500; /* Màu chữ cam */
        font-size: 24px; /* Kích thước chữ */
        font-weight: bold; /* Đậm chữ */
        z-index: 102;
        display: flex; /* Căn giữa */
        align-items: center; /* Căn giữa theo chiều dọc */
        justify-content: center; /* Căn giữa theo chiều ngang */
        white-space: pre-line; /* Giữ nguyên xuống dòng */
        flex-direction: column; /* Xếp nội dung theo chiều dọc */
        text-align: center; /* Căn giữa chữ */
    ">
        <div>
        🔥 Survive until the music is over 🔥
        </div>
        <p style="font-size: 16px; font-weight: normal;">
            You can play Running Dinosaur while waiting for the music (up to 5 minutes).  
            Press Space to start the game online and jump your Dino, use down arrow (↓) to duck.
        </p>
    </div>
    
    <style type="text/css">
    iframe { 
        margin-top: 20px;
        position: absolute; 
        width: 100%; 
        height: 100%; 
        z-index: 100; 
    }
    </style>
    """
    st.components.v1.html(game_html, height=320)


# Phần chính của ứng dụng
async def Feel_The_Beat():
    st.title("🎵 Feel The Beat - Tạo Nhạc AI")

    custom_mode = st.toggle("Custom Mode", value=True)
    if "lyrics" in st.session_state:
        lyrics = st.session_state.lyrics
        prompt = st.text_area("💡 Enter a description of the track you want to create:", 
                              value=lyrics, 
                              placeholder="A relaxing piano piece with a gentle melody...",height=300)
    else:
        prompt = st.text_area("💡 Enter a description of the track you want to create:", 
                              placeholder="A relaxing piano piece with a gentle melody...",height=300)

    # Kiểm tra nếu custom_mode tắt
    if custom_mode == False:
        style = "Classical"  # Gán giá trị mặc định nếu custom_mode tắt
        title = "My AI Music"  # Gán title mặc định nếu custom_mode tắt
        instrumental = False  # Gán giá trị mặc định cho instrumental nếu custom_mode tắt
    else:
        # Danh sách gợi ý phong cách nhạc
        music_styles = ["Classical", "Jazz", "Lo-fi", "Ambient", "Rock"]

        # Nếu chưa có session_state cho style_list, đặt giá trị mặc định
        if "style_list" not in st.session_state:
            st.session_state["style_list"] = []

        # Hộp nhập phong cách nhạc (hiển thị danh sách dưới dạng chuỗi)
        style = st.text_input("🎼 Enter music style:", ", ".join(st.session_state["style_list"]))

        # Đảm bảo style được sử dụng khi gửi yêu cầu
        style = style if style else "Classical"  # Nếu người dùng không nhập, sử dụng mặc định "Classical"

        # Hiển thị các nút theo hàng ngang
        cols = st.columns(len(music_styles))

        for i, music in enumerate(music_styles):
            with cols[i]:
                if st.button(music, use_container_width=True):
                    if music in st.session_state["style_list"]:
                        # Nếu đã có trong danh sách thì xóa đi (bỏ chọn)
                        st.session_state["style_list"].remove(music)
                    else:
                        # Nếu chưa có thì thêm vào danh sách
                        st.session_state["style_list"].append(music)
                    
                    # Cập nhật text box với danh sách mới
                    st.rerun()  # Cập nhật giao diện ngay lập tức

        title = st.text_input("🎶 Name the song:", "My AI Music")
        instrumental = st.checkbox("🎻 Instrumental", value=False)
    # Xóa music_data khi người dùng bấm nút
    # Xóa music_data khi người dùng bấm nút

    if st.button("🎧 Feel The Beat"):
        # ✅ Kiểm tra user đã đăng nhập
        if "user" not in st.session_state:
            st.warning("🔐 Bạn cần đăng nhập để sử dụng tính năng này.")
            st.stop()

        user_id = st.session_state["user"]["id"]

        # ✅ Kiểm tra số dư
        credit_data = supabase.table("user_credits").select("credits").eq("id", user_id).execute()
        current_credits = credit_data.data[0]["credits"] if credit_data.data else 0

        if current_credits < 25:
            st.error("❌ Bạn không đủ 25 tín dụng để sử dụng chức năng này. Vui lòng nạp thêm.")
            st.stop()

        # ✅ Xóa nhạc cũ nếu có
        if "music_data" in st.session_state:
            del st.session_state["music_data"]

        if not api_token or not prompt:
            st.warning("⚠️Please enter music description!")
        else:
            task_id = await generate_music(api_token, prompt, custom_mode, style, title, instrumental)
            if task_id:
                render_game_html()

                music_data = await check_music_status(api_token, task_id)

                if music_data:
                    # ✅ Trừ tín dụng nếu nhạc tạo thành công
                    new_credits = current_credits - 25
                    supabase.table("user_credits").update({"credits": new_credits}).eq("id", user_id).execute()

                    st.session_state["music_data"] = music_data
                    for audio_url, title, image_url in music_data:
                        # st.success(f"🎵 Your music is ready: [{title}]")
                        # render_music_player(title, audio_url, image_url)
                        st.session_state["music_data"] = music_data
                else:
                    st.warning("⏳ Music not ready after 5 minutes, please try again later!")
            else:
                st.error("🚨 Error in music generation!")

    # Kiểm tra nếu có nhạc đã tạo trong session_state
    if "music_data" in st.session_state:
        music_data = st.session_state["music_data"]
        for audio_url, title, image_url in music_data:
            st.success(f"🎵 Your music is ready: [{title}]")
            render_music_player(title, audio_url, image_url)
if menu == "Feel The Beat":
    asyncio.run(Feel_The_Beat())



if menu == "Library":
    if "user" in st.session_state and "email" in st.session_state["user"]:
        user_email = st.session_state["user"]["email"]
        user_profile = supabase.table("user_profiles").select("id").eq("email", user_email).execute()

        if user_profile.data:
            user_id = user_profile.data[0]["id"]
            songs = supabase.table("songs").select("*").eq("user_id", user_id).execute()

            if songs.data:
                st.subheader("🎶 Your Music Library")

                # ✅ Sắp xếp bài public lên đầu
                sorted_songs = sorted(songs.data, key=lambda x: not x.get("is_public", False))

                for song in sorted_songs:
                    # Tạo 2 cột: 1 bên ảnh + switch, 1 bên audio + info
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        render_music_player(song['title'], song['audio_url'], song['image_url'])
                        st.write(f"📝 Prompt: {song['prompt']}")
                        col3, col4 = st.columns([1, 7])
                        with col3:
                            is_public = song.get("is_public", False)
                            new_status = st_toggle_switch(
                                label="Public",  # Label chữ Public
                                key=f"toggle_{song['id']}",
                                default_value=is_public,
                                label_after=False,
                                active_color="#FF69B4",
                                inactive_color="#444",
                                track_color="#fce4ec" if is_public else "#999",
                            )
                            if new_status != is_public:
                                supabase.table("songs").update({"is_public": new_status}).eq("id", song["id"]).execute()
                        with col4:
                         # Thêm nút xóa bài hát dưới phần switch public
                            delete_button = st.button(f"🗑️ Xóa", key=f"delete_{song['id']}")

                            if delete_button:
                                # Hiển thị hộp chọn xác nhận trước khi xóa
                                confirm_delete = st.selectbox(
                                    "Bạn có chắc chắn muốn xóa bài hát này?",
                                    ["Chắc chắn", "Không"]
                                )

                                if confirm_delete == "Chắc chắn":
                                    # Xóa bài hát khỏi Supabase (cả cơ sở dữ liệu SQL)
                                    supabase.table("songs").delete().eq("id", song["id"]).execute()

                                    # Thông báo thành công
                                    st.success(f"Bài hát '{song['title']}' đã được xóa thành công.")
                                    
                                    # Làm mới lại danh sách bài hát sau khi xóa
                                    songs = supabase.table("songs").select("*").eq("user_id", user_id).execute()
                                    st.rerun()  # Tải lại trang để làm mới danh sách

                    with col2:

                        #render_music_player(song['title'], song['audio_url'], song['image_url'])
                        
                        # Giả sử song['prompt'] là một chuỗi dà

                        st.write(f"⏱ Duration: {song['duration']} seconds")
                        st.write(f"🎧 Model: {song['model_name']}")
                        st.write(f"🗓 Created at: {song['created_at']}")
                    st.markdown("---")
            else:
                st.info("🎵 Bạn chưa có bài hát nào.")
        else:
            st.error("❌ Không tìm thấy thông tin người dùng.")
    else:
        st.warning("🔒 Vui lòng đăng nhập để xem thư viện của bạn.")




# MoMo config
MOMO_CONFIG = {
    "MomoApiUrl": "https://test-payment.momo.vn/v2/gateway/api/create",
    "PartnerCode": "MOMO",
    "AccessKey": "F8BBA842ECF85",
    "SecretKey": "K951B6PE1waDMi640xX08PD3vg6EkVlz",
    "ReturnUrl": "http://localhost:8501",
    "IpnUrl": "https://webhook.site/b052aaf4-3be0-43c5-8bad-996d2d0c0e54",
    "RequestType": "captureWallet",
    "ExtraData": "Astronaut_Music_payment"
}


@st.cache_data(ttl=86400)
def get_usd_to_vnd():
    try:
        url = "https://v6.exchangerate-api.com/v6/5bfc9ccf0ed4b1708159250f/latest/USD"
        res = requests.get(url)
        if res.status_code == 200:
            rate = res.json()["conversion_rates"]["VND"]
            st.write(f"💱 USD → VND Exchange Rate (ExchangeRate-API): {rate:,.0f}")
            return int(rate)
    except:
        st.error("❌  Error fetching exchange rate.")
    return 25000

def generate_signature(data, secret_key):
    raw_signature = (
        f"accessKey={data['accessKey']}&amount={data['amount']}&extraData={data['extraData']}&"
        f"ipnUrl={data['ipnUrl']}&orderId={data['orderId']}&orderInfo={data['orderInfo']}&"
        f"partnerCode={data['partnerCode']}&redirectUrl={data['redirectUrl']}&"
        f"requestId={data['requestId']}&requestType={data['requestType']}"
    )
    return hmac.new(secret_key.encode(), raw_signature.encode(), hashlib.sha256).hexdigest()

if menu == "Payment":
    st.title("💰 Payment")
    if "user" not in st.session_state:
        st.warning("🔐 Please log in.")
        st.stop()
    user_id = st.session_state["user"]["id"]
    # Lấy số dư hiện tại
    credit_data = supabase.table("user_credits").select("credits").eq("id", user_id).execute()
    credits = credit_data.data[0]["credits"] if credit_data.data else 0
    st.metric("Current Credits", f"{credits:,} credits")
    # Bảng giá
    st.subheader("📦 Credit Packages")
    usd_to_vnd = get_usd_to_vnd()
    st.markdown("🎶 **Cost per music creation: 25 credits**")
    packages = [
        {"price": 5, "credits": 1000, "discount": None},
        {"price": 50, "credits": 10000, "discount": None},
        {"price": 500, "credits": 105000, "discount": "Save 5%"},
        {"price": 1250, "credits": 275000, "discount": "Save 10%"},
    ]
    cols = st.columns(len(packages), gap="large")

    for i, (col, pack) in enumerate(zip(cols, packages)):
        with col:
            if pack["discount"]:
                package_html = f"""
                <div class="package highlight">
                    <div class="ribbon">{pack["discount"]}</div>
                    <h3>${pack['price']}</h3>
                    <p>{pack['credits']:,} Credits</p>
                </div>
                """
            else:
                package_html = f"""
                <div class="package">
                    <h3>${pack['price']}</h3>
                    <p>{pack['credits']:,} Credits</p>
                </div>
                """
            st.markdown(package_html, unsafe_allow_html=True)

            with st.form(f"form_{i}"):
                if st.form_submit_button("🛒Buy Credits"):
                    order_id = str(uuid.uuid4())
                    request_id = str(uuid.uuid4())
                    price_vnd = int(pack["price"] * usd_to_vnd)
                    order_info = f"Mua {pack['credits']} credits cho user {user_id}"                  
                    payload = {
                        "partnerCode": MOMO_CONFIG["PartnerCode"],
                        "accessKey": MOMO_CONFIG["AccessKey"],
                        "requestId": request_id,
                        "amount": str(price_vnd),
                        "orderId": order_id,
                        "orderInfo": order_info,
                        "redirectUrl": MOMO_CONFIG["ReturnUrl"],
                        "ipnUrl": MOMO_CONFIG["IpnUrl"],
                        "extraData": MOMO_CONFIG["ExtraData"],
                        "requestType": MOMO_CONFIG["RequestType"]
                    }
                    payload["signature"] = generate_signature(payload, MOMO_CONFIG["SecretKey"])

                    res = requests.post(MOMO_CONFIG["MomoApiUrl"], json=payload)
                    if res.status_code == 200 and res.json().get("payUrl"):
                        pay_url = res.json()["payUrl"]
                        supabase.table("pending_payments").insert({
                            "user_id": user_id,
                            "order_id": order_id,
                            "credits": pack["credits"],
                            "amount": price_vnd
                        }).execute()

                        st.success("✅ Order created. Click the button below to pay.")
                        st.markdown(f"""
                            <a href="{pay_url}" target="_blank">
                                <button style="background-color:#f72585; color:white; padding:10px 20px;
                                               border:none; border-radius:5px; cursor:pointer;">
                                    🚀 Open MoMo to pay
                                </button>
                            </a>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to create order. Please try again.")

    st.markdown("<hr>", unsafe_allow_html=True)
    # CSS đẹp
    st.markdown("""
        <style>
        
        .package {
            position: relative;
            background: linear-gradient(to right, #9b59b6, #ff4e50, #ff7eb3) !important;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            color: #ffffff;
            min-height: 120px;
            transition: 0.3s;
        }
        .package.highlight {
            background: linear-gradient(to right, #2196F3, #9C27B0, #FF4081, #FFEB3B, #4CAF50) !important;
            color: #ffffff;
        }
        
        .ribbon {
            width: 80px;
            background: linear-gradient(to right, #1A237E, #4A148C) !important;;
            color: #FFB300;
            font-weight: bold;
            text-align: center;
            font-size: 0.7rem;
            position: absolute;
            right: -25px;
            top: 10px;
            transform: rotate(45deg);
            padding: 3px 0;
        }

        </style>
    """, unsafe_allow_html=True)

    

    # ✅ Xử lý khi quay lại từ MoMo qua ReturnUrl
    params = st.query_params
    order_id_param = params.get("orderId")
    result_code = params.get("resultCode")
    trans_id = params.get("transId")
    amount = int(params.get("amount", "0"))

    if order_id_param:
        exists = supabase.table("payment_history").select("*").eq("order_id", order_id_param).execute()
        if exists.data:
            st.info("Transaction already processed.")
        else:
            pending = supabase.table("pending_payments").select("*").eq("order_id", order_id_param).execute().data
            if pending:
                pending = pending[0]
                if result_code == "0":
                    supabase.table("user_credits").update({"credits": credits + pending["credits"]}).eq("id", user_id).execute()
                    supabase.table("payment_history").insert({
                        "user_id": user_id,
                        "order_id": order_id_param,
                        "amount": amount,
                        "credits": pending["credits"],
                        "status": "completed",
                        "payment_method": "momo",
                        "transaction_id": trans_id,
                        "created_at": datetime.utcnow().isoformat()
                    }).execute()
                    supabase.table("pending_payments").delete().eq("order_id", order_id_param).execute()
                    st.success(f"✅ Added {pending['credits']:,} credits.")
                    st.rerun()
                else:
                    st.warning("❌ Payment failed or cancelled.")
    
    st.markdown("## 🧾 Transaction History (last 3 months)")

    # user_id = st.session_state['user']['id']
    three_months_ago = (datetime.now() - timedelta(days=90)).isoformat()

    # Lấy dữ liệu từ Supabase
    history = supabase.table("payment_history").select("*") \
        .eq("user_id", user_id).gte("created_at", three_months_ago) \
        .order("created_at", desc=True).execute()

    if history.data:
        df = pd.DataFrame(history.data)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%d-%m-%Y %H:%M')
        df_display = df[['order_id', 'amount', 'credits', 'status', 'payment_method', 'transaction_id', 'created_at']]

        st.dataframe(df_display, use_container_width=True, height=220)
    else:
        st.info("No transactions in the last 3 months.")
    # ✅ Trường hợp không có orderId → Kiểm tra đơn pending chưa xác nhận
    if not order_id_param:
        pending_query = supabase.table("pending_payments").select("*").eq("user_id", user_id).execute()
        pending_data = pending_query.data[0] if pending_query.data else None


