# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Custom CSS with hover effects and animations
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f2f6;
#         padding: 20px;
#     }
#     .title {
#         color: #1e3a8a;
#         font-size: 40px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 20px;
#         transition: all 0.3s ease;
#     }
#     .title:hover {
#         color: #3b82f6;
#         transform: scale(1.05);
#     }
#     .subtitle {
#         color: #1e3a8a;
#         font-size: 24px;
#         margin: 20px 0;
#         transition: all 0.3s ease;
#     }
#     .subtitle:hover {
#         color: #3b82f6;
#     }
#     .metric-box {
#         background-color: white;
#         padding: 15px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 10px 0;
#         transition: all 0.3s ease;
#     }
#     .metric-box:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
#     }
#     .stButton>button {
#         background-color: #3b82f6;
#         color: white;
#         border-radius: 8px;
#         transition: all 0.3s ease;
#     }
#     .stButton>button:hover {
#         background-color: #1e3a8a;
#         transform: scale(1.05);
#     }
#     .sidebar .sidebar-content {
#         background-color: #ffffff;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Load models
# @st.cache_resource
# def load_models():
#     tier_model = joblib.load('tier_model.pkl')
#     name_model = joblib.load('name_model.pkl')
#     branch_model = joblib.load('branch_model.pkl')
#     salary_model = joblib.load('salary_model.pkl')
#     name_encoder = joblib.load('name_encoder.pkl')
#     branch_encoder = joblib.load('branch_encoder.pkl')
#     college_encoder = joblib.load('college_encoder.pkl')
#     preprocessor = joblib.load('preprocessor.pkl')
#     return (tier_model, name_model, branch_model, salary_model, 
#             name_encoder, branch_encoder, college_encoder, preprocessor)

# (tier_model, name_model, branch_model, salary_model, 
#  name_encoder, branch_encoder, college_encoder, preprocessor) = load_models()

# # College hierarchy
# college_hierarchy = [
#     'Tier 4 - Other',
#     'Tier 3 - Private/State', 
#     'Tier 2 - Mid Colleges',
#     'Tier 1 - Other IIT/Top NIT',
#     'Tier 1 - Top IIT'
# ]

# def main():
#     # Sidebar for navigation
#     with st.sidebar:
#         st.image("https://via.placeholder.com/150x50.png?text=Logo", use_column_width=True)
#         st.markdown("## Navigation")
#         page = st.radio("Go to", ["Predictor", "Dashboard", "About"], 
#                        label_visibility="collapsed")
        
#         st.markdown("## Quick Options")
#         sample_data = st.button("Load Sample Data")
#         st.button("Reset Form")

#     # Main content based on sidebar selection
#     if page == "Predictor":
#         st.markdown('<div class="title">College & Career Predictor</div>', unsafe_allow_html=True)
#         st.write("Predict your college placement and salary with AI precision")

#         # Input sections
#         col1, col2 = st.columns([2, 1])

#         with col1:
#             # Academic Section
#             st.markdown('<div class="subtitle">Academic Profile</div>', unsafe_allow_html=True)
#             with st.expander("Enter Academic Details", expanded=True):
#                 tenth = st.number_input("10th Percentage", 0.0, 100.0, 80.0, 0.1)
#                 twelfth = st.number_input("12th Percentage", 0.0, 100.0, 80.0, 0.1)
#                 jee = st.number_input("JEE Rank", 1, 1000000, 1000, 1)
#                 cgpa = st.slider("CGPA", 1.0, 10.0, 7.0, 0.1)

#             # Professional Section
#             st.markdown('<div class="subtitle">Professional Experience</div>', unsafe_allow_html=True)
#             with st.expander("Enter Experience Details"):
#                 workexp = st.number_input("Work Experience (years)", 0, 20, 2)
#                 fexp = st.slider("Field Experience (years)", 0, 20, 2)
#                 proj = st.slider("Number of Projects", 0, 20, 1)
#                 intern = st.slider("Number of Internships", 0, 10, 0)

#             # Skills Section
#             st.markdown('<div class="subtitle">Skills & Portfolio</div>', unsafe_allow_html=True)
#             col3, col4 = st.columns(2)
#             with col3:
#                 exp_lev = st.selectbox("Expertise Level", [1, 2, 3, 4, 5], 2)
#                 soft = st.selectbox("Soft Skills", [1, 2, 3, 4, 5], 2)
#                 codeqs = st.slider("Coding Questions", 0, 200, 50, 10)
#                 repos = st.slider("Repositories", 0, 50, 5)
            
#             with col4:
#                 apt = st.selectbox("Aptitude", [1, 2, 3, 4, 5], 2)
#                 dsa = st.selectbox("DSA Level", [1, 2, 3, 4, 5], 2)
#                 ghacts = st.slider("GitHub Activities", 0, 50, 10)
#                 li = st.slider("LinkedIn Posts", 0, 50, 3)

#         with col2:
#             # Quick Stats Sidebar
#             st.markdown('<div class="subtitle">Quick Stats</div>', unsafe_allow_html=True)
#             st.metric("Model Accuracy", "92%", "±3%")
#             st.metric("Predictions Made", "1,234", "+15 today")
            
#             # Additional Inputs
#             st.markdown('<div class="subtitle">Additional Details</div>', unsafe_allow_html=True)
#             gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], 0)
#             domain = st.selectbox("Domain", ['Full Stack', 'Machine Learning', 
#                                            'Android Development', 'Other'], 0)
#             ref = st.selectbox("Referral", ['Yes', 'No'], 1)
#             hack = st.slider("Hackathons", 0, 10, 0)
#             certs = st.slider("Certifications", 0, 20, 0)

#             # Predict Button
#             if st.button("Predict Now", key="predict", use_container_width=True):
#                 input_data = {
#                     '10th_percent': tenth,
#                     '12th_percent': twelfth,
#                     'jee_rank': jee,
#                     'experience': workexp,
#                     'experience_field': fexp,
#                     'num_projects': proj,
#                     'expertise_level': exp_lev,
#                     'num_internships': intern,
#                     'soft_skill_rating': soft,
#                     'aptitude_rating': apt,
#                     'dsa_level': dsa,
#                     'num_hackathons': hack,
#                     'competitive_coding_solved': codeqs,
#                     'num_repos': repos,
#                     'github_activities': ghacts,
#                     'linkedin_posts': li,
#                     'num_certifications': certs,
#                     'cgpa': cgpa,
#                     'gender': gender,
#                     'domain': domain,
#                     'referral': ref
#                 }
#                 inp_df = pd.DataFrame([input_data])

#                 with st.spinner("Analyzing your profile..."):
#                     t_code = tier_model.predict(inp_df)[0]
#                     n_code = name_model.predict(inp_df)[0]
#                     b_code = branch_model.predict(inp_df)[0]
#                     sal = salary_model.predict(inp_df)[0]

#                     tier_pred = college_hierarchy[int(t_code)]
#                     name_pred = name_encoder.inverse_transform([[n_code]])[0][0]
#                     branch_pred = branch_encoder.inverse_transform([[b_code]])[0][0]
#                     salary_pred = sal

#                 # Results Display
#                 st.markdown('<div class="subtitle">Your Results</div>', unsafe_allow_html=True)
#                 st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#                 st.metric("College Tier", tier_pred)
#                 st.metric("College Name", name_pred)
#                 st.metric("Branch", branch_pred)
#                 st.metric("Expected Salary", f"₹{salary_pred:,.2f}")
#                 st.markdown('</div>', unsafe_allow_html=True)
                
#                 # Download Button
#                 results = f"Tier: {tier_pred}\nCollege: {name_pred}\nBranch: {branch_pred}\nSalary: ₹{salary_pred:,.2f}"
#                 st.download_button("Download Results", results, "prediction_results.txt")

#     elif page == "Dashboard":
#         st.markdown('<div class="title">Prediction Dashboard</div>', unsafe_allow_html=True)
#         st.write("Coming Soon: Interactive charts and statistics")
#         # Add placeholder for charts
#         col1, col2 = st.columns(2)
#         with col1:
#             st.write("Accuracy Trends")
#             st.line_chart(np.random.randn(30, 1))
#         with col2:
#             st.write("Prediction Distribution")
#             st.bar_chart(np.random.randn(10))

#     elif page == "About":
#         st.markdown('<div class="title">About This Project</div>', unsafe_allow_html=True)
#         st.write("""
#         ### College & Career Predictor
#         An AI-powered solution for:
#         - College tier prediction
#         - Institution recommendation
#         - Branch suggestion
#         - Salary estimation
        
#         **Tech Stack:**
#         - Streamlit (Frontend)
#         - Scikit-learn (ML)
#         - Pandas & NumPy (Data)
        
#         **Created for:**
#         Project Expo 2025
        
#         **Team:**
#         [Your Name] - Lead Developer
#         """)
        
#         st.button("Contact Us", help="Reach out to the team")

#     # Handle sample data
#     if sample_data:
#         st.session_state.tenth = 92.5
#         st.session_state.twelfth = 88.0
#         st.session_state.jee = 500
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Custom CSS with hover effects and animations
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        color: #1e3a8a;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .title:hover {
        color: #3b82f6;
        transform: scale(1.05);
    }
    .subtitle {
        color: #1e3a8a;
        font-size: 24px;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    .subtitle:hover {
        color: #3b82f6;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    tier_model = joblib.load('tier_model.pkl')
    name_model = joblib.load('name_model.pkl')
    branch_model = joblib.load('branch_model.pkl')
    salary_model = joblib.load('salary_model.pkl')
    name_encoder = joblib.load('name_encoder.pkl')
    branch_encoder = joblib.load('branch_encoder.pkl')
    college_encoder = joblib.load('college_encoder.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return (tier_model, name_model, branch_model, salary_model, 
            name_encoder, branch_encoder, college_encoder, preprocessor)

(tier_model, name_model, branch_model, salary_model, 
 name_encoder, branch_encoder, college_encoder, preprocessor) = load_models()

# College hierarchy
college_hierarchy = [
    'Tier 4 - Other',
    'Tier 3 - Private/State', 
    'Tier 2 - Mid Colleges',
    'Tier 1 - Other IIT/Top NIT',
    'Tier 1 - Top IIT'
]

def main():
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50.png?text=Logo", use_column_width=True)
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Predictor", "Dashboard", "About"], label_visibility="collapsed")
        
        st.markdown("## Quick Options")
        sample_data = st.button("Load Sample Data")
        st.button("Reset Form")

    if page == "Predictor":
        st.markdown('<div class="title">College & Career Predictor</div>', unsafe_allow_html=True)
        st.write("Predict your college placement and salary with AI precision")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="subtitle">Academic Profile</div>', unsafe_allow_html=True)
            with st.expander("Enter Academic Details", expanded=True):
                tenth = st.number_input("10th Percentage", 0.0, 100.0, 80.0, 0.1)
                twelfth = st.number_input("12th Percentage", 0.0, 100.0, 80.0, 0.1)
                jee = st.number_input("JEE Rank", 1, 1000000, 1000, 1)
                cgpa = st.slider("CGPA", 1.0, 10.0, 7.0, 0.1)

            st.markdown('<div class="subtitle">Professional Experience</div>', unsafe_allow_html=True)
            with st.expander("Enter Experience Details"):
                workexp = st.number_input("Work Experience (years)", 0, 20, 2)
                fexp = st.slider("Field Experience (years)", 0, 20, 2)
                proj = st.slider("Number of Projects", 0, 20, 1)
                intern = st.slider("Number of Internships", 0, 10, 0)

            st.markdown('<div class="subtitle">Skills & Portfolio</div>', unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                exp_lev = st.selectbox("Expertise Level", [1, 2, 3, 4, 5], 2)
                soft = st.selectbox("Soft Skills", [1, 2, 3, 4, 5], 2)
                codeqs = st.slider("Coding Questions", 0, 200, 50, 10)
                repos = st.slider("Repositories", 0, 50, 5)
            with col4:
                apt = st.selectbox("Aptitude", [1, 2, 3, 4, 5], 2)
                dsa = st.selectbox("DSA Level", [1, 2, 3, 4, 5], 2)
                ghacts = st.slider("GitHub Activities", 0, 50, 10)
                li = st.slider("LinkedIn Posts", 0, 50, 3)

        with col2:
            st.markdown('<div class="subtitle">Quick Stats</div>', unsafe_allow_html=True)
            st.metric("Model Accuracy", "92%", "±3%")
            st.metric("Predictions Made", "1,234", "+15 today")

            st.markdown('<div class="subtitle">Additional Details</div>', unsafe_allow_html=True)
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], 0)
            domain = st.selectbox("Domain", ['Full Stack', 'Machine Learning', 'Android Development', 'Other'], 0)
            ref = st.selectbox("Referral", ['Yes', 'No'], 1)
            hack = st.slider("Hackathons", 0, 10, 0)
            certs = st.slider("Certifications", 0, 20, 0)

            if st.button("Predict Now", key="predict", use_container_width=True):
                if tenth < 35.0 or twelfth < 35.0:
                    st.error("⚠️ Both 10th and 12th percentages must be at least 35%. Please correct the input.")
                else:
                    input_data = {
                        '10th_percent': tenth,
                        '12th_percent': twelfth,
                        'jee_rank': jee,
                        'experience': workexp,
                        'experience_field': fexp,
                        'num_projects': proj,
                        'expertise_level': exp_lev,
                        'num_internships': intern,
                        'soft_skill_rating': soft,
                        'aptitude_rating': apt,
                        'dsa_level': dsa,
                        'num_hackathons': hack,
                        'competitive_coding_solved': codeqs,
                        'num_repos': repos,
                        'github_activities': ghacts,
                        'linkedin_posts': li,
                        'num_certifications': certs,
                        'cgpa': cgpa,
                        'gender': gender,
                        'domain': domain,
                        'referral': ref
                    }
                    inp_df = pd.DataFrame([input_data])

                    with st.spinner("Analyzing your profile..."):
                        t_code = tier_model.predict(inp_df)[0]
                        n_code = name_model.predict(inp_df)[0]
                        b_code = branch_model.predict(inp_df)[0]
                        sal = salary_model.predict(inp_df)[0]

                        tier_pred = college_hierarchy[int(t_code)]
                        name_pred = name_encoder.inverse_transform([[n_code]])[0][0]
                        branch_pred = branch_encoder.inverse_transform([[b_code]])[0][0]
                        salary_pred = sal

                    st.markdown('<div class="subtitle">Your Results</div>', unsafe_allow_html=True)
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("College Tier", tier_pred)
                    st.metric("College Name", name_pred)
                    st.metric("Branch", branch_pred)
                    st.metric("Expected Salary", f"₹{salary_pred:,.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    results = f"Tier: {tier_pred}\nCollege: {name_pred}\nBranch: {branch_pred}\nSalary: ₹{salary_pred:,.2f}"
                    st.download_button("Download Results", results, "prediction_results.txt")

    elif page == "Dashboard":
        st.markdown('<div class="title">Prediction Dashboard</div>', unsafe_allow_html=True)
        st.write("Coming Soon: Interactive charts and statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Accuracy Trends")
            st.line_chart(np.random.randn(30, 1))
        with col2:
            st.write("Prediction Distribution")
            st.bar_chart(np.random.randn(10))

    elif page == "About":
        st.markdown('<div class="title">About This Project</div>', unsafe_allow_html=True)
        st.write("""
        ### College & Career Predictor
        An AI-powered solution for:
        - College tier prediction
        - Institution recommendation
        - Branch suggestion
        - Salary estimation
        
        **Tech Stack:**
        - Streamlit (Frontend)
        - Scikit-learn (ML)
        - Pandas & NumPy (Data)
        
        **Created for:**
        Project Expo 2025
        
        **Team:**
        [Your Name] - Lead Developer
        """)
        st.button("Contact Us", help="Reach out to the team")

    if sample_data:
        st.session_state.tenth = 92.5
        st.session_state.twelfth = 88.0
        st.session_state.jee = 500
        st.experimental_rerun()

if __name__ == "__main__":
    main()
