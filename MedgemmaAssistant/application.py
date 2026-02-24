import streamlit as st
import student_assistant
import patient_assistant
import doctor_assistant

def main():
    st.set_page_config(page_title="Medical Assistant", layout="wide")

    # 🔁 Handle "return to main" requests via query params
    params = st.query_params
    page_param = params.get("page")

    if page_param == "main" or page_param == ["main"]:
        # Clear the query param so it doesn't keep redirecting
        if "page" in st.query_params:
            del st.query_params["page"]

        # Reset role so main screen shows again
        if "role" in st.session_state:
            del st.session_state["role"]

        st.rerun()

    # If role not chosen yet → show role selection
    if "role" not in st.session_state:
        st.title("🩺 Welcome to the Medical Assistant")
        st.subheader("Please select your role")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎓 I am a Student", use_container_width=True):
                st.session_state.role = "student"
                st.rerun()

        with col2:
            if st.button("🧑‍⚕️ I am a Patient", use_container_width=True):
                st.session_state.role = "patient"
                st.rerun()

        with col3:
            if st.button("👨‍⚕️ I am a Doctor", use_container_width=True):
                st.session_state.role = "doctor"
                st.rerun()

        return

    # If role already chosen → load correct app
    if st.session_state.role == "student":
        student_assistant.main()
    elif st.session_state.role == "patient":
        patient_assistant.main()
    elif st.session_state.role == "doctor":
        doctor_assistant.main()

if __name__ == "__main__":
    main()
