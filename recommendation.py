import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Course Dataset
courses = {
    "Course": [
        "Python for Beginners",
        "Machine Learning",
        "Artificial Intelligence",
        "Data Science",
        "Web Development",
        "Cyber Security",
        "Cloud Computing",
        "Java Programming",
        "C++ Programming",
        "UI UX Design"
    ],

    "Skills": [
        "python programming basics coding",
        "python machine learning ai data",
        "ai deep learning neural networks python",
        "python statistics visualization pandas",
        "html css javascript frontend",
        "network security ethical hacking",
        "aws azure cloud devops",
        "java programming oop",
        "c++ programming dsa",
        "figma ui ux design"
    ]
}

df = pd.DataFrame(courses)

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Skills"])

# User Input
user_interest = input("Enter your interests: ")

user_vector = vectorizer.transform([user_interest])

# Calculate similarity
similarity = cosine_similarity(user_vector, tfidf_matrix)

# Rank courses
scores = similarity.flatten()

recommendations = sorted(
    zip(df["Course"], scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nRecommended Courses:\n")

for course, score in recommendations[:5]:
    print(f"{course}  (Similarity Score: {score:.2f})")