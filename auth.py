
import re
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# Load biến môi trường từ .env

load_dotenv()


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)



import re  # Đảm bảo bạn đã import re cho biểu thức chính quy

def register_user(email, password, full_name):
    try:
        # Kiểm tra định dạng email
        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_regex, email):
            return False, "❌ Invalid email format."

        # Đăng ký tài khoản
        res = supabase.auth.sign_up({
            "email": email,
            "password": password,
        })

        if not res.user:
            return False, "⚠️ Unable to register account, please try again."
        
        # Tạo hồ sơ người dùng trong bảng user_profiles
        supabase.table("user_profiles").insert({
            "id": res.user.id,
            "email": email,
            "full_name": full_name,
            "role": "client"
        }).execute()

        # Khởi tạo tín dụng cho người dùng mới (75 tín dụng ban đầu)
        supabase.table("user_credits").insert({
            "id": res.user.id,
            "credits": 75
        }).execute()
    
        return True, f"✅ Registration successful! Please verify your email: {email}"

    except Exception as e:
        return False, f"❌ Registration error: {str(e)}"

# =============================
# 2. ĐĂNG NHẬP
# =============================

def login_user(email, password):
    try:

        result = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        user = result.user

        if user.email_confirmed_at is None:
            return False, "📩 Please verify your email before logging in."
        
        # Lấy thông tin profile từ user_profiles
        profile_data = supabase.table("user_profiles").select("*").eq("id", user.id).execute()
        # Kiểm tra xem có profile hay chưa
        if profile_data.data:
            user_profile = profile_data.data[0]
            # Lưu thông tin user đầy đủ vào session
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "full_name": user_profile.get("full_name", ""),
                "role": user_profile.get("role", "client"),
                "created_at": user_profile.get("created_at", "")
            }
        else:
            # Nếu chưa có profile, tạo mới
            new_profile = {
                "id": user.id,
                "email": user.email,
                "full_name": user_profile.get("full_name", ""),
                "role": "client"
            }
            
            # Lưu thông tin vào session
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "full_name": new_profile["full_name"],
                "role": new_profile["role"]
            }

        # Kiểm tra xem người dùng đã có bản ghi credits chưa
        user_credits = supabase.table("user_credits").select("*").eq("id", user.id).execute()
        
        # Nếu chưa có bản ghi credits, tạo mới
        if not user_credits.data:
            supabase.table("user_credits").insert({
                "id": user.id,
                "credits": 75
            }).execute()

        return True, f"🎉 Login successful, welcome {st.session_state['user']['full_name']}!"

    except Exception as e:
        return False, f"❌ Login error: {e}"

