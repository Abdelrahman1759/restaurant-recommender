import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import os

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv", encoding='latin-1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.drop(columns=[
        'restaurant_id', 'address', 'locality_verbose',
        'longitude', 'latitude', 'switch_to_order_menu', 'rating_color'
    ])
    df = df.dropna(subset=['cuisines'])
    df['average_cost_for_two'] = pd.to_numeric(df['average_cost_for_two'], errors='coerce')
    df['aggregate_rating'] = pd.to_numeric(df['aggregate_rating'], errors='coerce')
    df['cost_bucket'] = pd.qcut(df['average_cost_for_two'], q=3, labels=['Low', 'Medium', 'High'])
    df['primary_cuisine'] = df['cuisines'].astype(str).str.split(',').str[0].str.strip()
    df['rating_5scale'] = df['aggregate_rating'].replace(0, pd.NA).round(1)
    df['votes'] = df['votes'].fillna(0)
    return df

# Filter and rank function with two ranking strategies for A/B test
def filter_and_rank(df, cuisines=None, budget=None, location=None, top_n=10, group='A'):
    filtered_df = df.copy()
    cost_order = ['Low', 'Medium', 'High']
    filtered_df['cost_bucket'] = pd.Categorical(filtered_df['cost_bucket'], categories=cost_order, ordered=True)

    if cuisines:
        cuisines_lower = [c.lower() for c in cuisines]
        filtered_df = filtered_df[
            filtered_df['primary_cuisine'].str.lower().apply(lambda x: any(c in x for c in cuisines_lower))
        ]
    if budget:
        filtered_df = filtered_df[filtered_df['cost_bucket'] == budget]
    if location:
        filtered_df = filtered_df[
            filtered_df['city'].str.lower().str.contains(location.lower(), na=False)
        ]

    if filtered_df.empty:
        return pd.DataFrame(columns=[
            'restaurant_name', 'primary_cuisine', 'average_cost_for_two',
            'rating_5scale', 'votes', 'city', 'explanation'
        ])

    # Scaling votes and ratings separately
    scaler_votes = MinMaxScaler()
    scaler_rating = MinMaxScaler()

    votes_scaled = scaler_votes.fit_transform(filtered_df[['votes']].fillna(0))
    rating_scaled = scaler_rating.fit_transform(filtered_df[['rating_5scale']].fillna(0))

    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df['votes_scaled'] = votes_scaled
    filtered_df['rating_scaled'] = rating_scaled

    # A/B Test groups use different ranking weights or methods:
    if group == 'A':
        # Group A: Weighted sum with 70% rating, 30% votes
        filtered_df['score'] = 0.7 * filtered_df['rating_scaled'] + 0.3 * filtered_df['votes_scaled']
    else:
        # Group B: Weighted sum with 50% rating, 50% votes (example variation)
        filtered_df['score'] = 0.5 * filtered_df['rating_scaled'] + 0.5 * filtered_df['votes_scaled']

    result_df = filtered_df.sort_values(by='score', ascending=False).head(top_n).copy()

    explanation_parts = []
    if cuisines:
        explanation_parts.append(" or ".join(cuisines) + " cuisine")
    if budget:
        explanation_parts.append(f"₹{filtered_df['average_cost_for_two'].median():.0f} budget")
    if location:
        explanation_parts.append(f"in {location}")

    avg_rating = filtered_df['rating_5scale'].mean()
    if pd.isna(avg_rating):
        avg_rating = 0
    avg_rating = round(avg_rating, 1)

    explanation_str = f"Matched: {' | '.join(explanation_parts)} | Avg Rating: {avg_rating}"
    result_df['explanation'] = explanation_str

    return result_df[[
        'restaurant_name', 'primary_cuisine', 'average_cost_for_two',
        'rating_5scale', 'votes', 'city', 'explanation'
    ]]

# Assign user to A/B group once per session
def get_ab_group():
    if 'group' not in st.session_state:
        st.session_state.group = random.choice(['A', 'B'])
    return st.session_state.group

# Save feedback to CSV file
def save_feedback(group, satisfaction, relevance, usability):
    feedback_data = {
        'group': group,
        'satisfaction': satisfaction,
        'relevance': relevance,
        'usability': usability
    }
    feedback_file = 'feedback.csv'
    df_feedback = pd.DataFrame([feedback_data])
    if os.path.exists(feedback_file):
        df_feedback.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df_feedback.to_csv(feedback_file, index=False)

def main():
    st.title("Knowledge-Based Restaurant Recommender System")

    df = load_data()

    # Assign user to A/B group and show it (optional)
    group = get_ab_group()
    st.sidebar.write(f"**You are in Group {group} (A/B Test)**")

    st.sidebar.header("Select your preferences")

    cuisine_options = sorted(df['primary_cuisine'].dropna().unique())
    selected_cuisines = st.sidebar.multiselect("Select Cuisine(s)", cuisine_options)

    budget_options = ['Low', 'Medium', 'High']
    selected_budget = st.sidebar.selectbox("Select Budget", [''] + budget_options)

    location_input = st.sidebar.text_input("Enter Location (City)")

    top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    if st.sidebar.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            results = filter_and_rank(
                df,
                cuisines=selected_cuisines if selected_cuisines else None,
                budget=selected_budget if selected_budget else None,
                location=location_input if location_input else None,
                top_n=top_n,
                group=group
            )
        if results.empty:
            st.warning("No recommendations found with the selected filters.")
        else:
            for idx, row in results.iterrows():
                st.subheader(row['restaurant_name'])
                st.write(f"Cuisine: {row['primary_cuisine']}")
                st.write(f"Cost for Two: ₹{row['average_cost_for_two']}")
                st.write(f"Rating: {row['rating_5scale']}")
                st.write(f"Votes: {row['votes']}")
                st.write(f"City: {row['city']}")
                st.info(row['explanation'])
                st.markdown("---")

    # Persistent feedback section using session state keys
    st.header("Help us improve!")

    options_satisfaction = ["Very Unsatisfied", "Unsatisfied", "Neutral", "Satisfied", "Very Satisfied"]
    options_relevance = ["Not relevant", "Slightly relevant", "Moderately relevant", "Very relevant", "Extremely relevant"]
    options_usability = ["Very difficult", "Difficult", "Neutral", "Easy", "Very easy"]

    satisfaction = st.radio("How satisfied are you with these recommendations?", options=options_satisfaction, key='satisfaction')
    relevance = st.radio("How relevant are the recommendations to your preferences?", options=options_relevance, key='relevance')
    usability = st.radio("How easy was it to use this recommendation system?", options=options_usability, key='usability')

    if st.button("Submit Feedback"):
        save_feedback(group, satisfaction, relevance, usability)
        st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
