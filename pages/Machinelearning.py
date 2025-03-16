import streamlit as st

st.set_page_config(page_title="Machine Learning Detail")
st.header("Hogwart House")
st.write(
    "At Hogwarts, the magical school from Harry Potter, new students must go through "
    "the Sorting Ceremony, which assigns them to one of the four houses: "
    "**Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.**"
)

st.markdown("---")

st.markdown("## Characteristics of Each House")

# Gryffindor
st.markdown('<h3 style="color:#AE0001;">Gryffindor</h3>', unsafe_allow_html=True)
st.write(
    "**Core Values:** Courage, Determination, Bravery  \n"
    "**Founder:** Godric Gryffindor  \n"
    "**Notable Students:** Harry Potter, Hermione Granger, Ron Weasley, Albus Dumbledore  \n"
    "**House Colors:** Red & Gold  \n"
    "**House Mascot:** Lion"
)

# Hufflepuff
st.markdown('<h3 style="color:#FFD700;">Hufflepuff</h3>', unsafe_allow_html=True)
st.write(
    "**Core Values:** Hard Work, Loyalty, Friendliness  \n"
    "**Founder:** Helga Hufflepuff  \n"
    "**Notable Students:** Cedric Diggory, Newt Scamander  \n"
    "**House Colors:** Yellow & Black  \n"
    "**House Mascot:** Badger"
)

# Ravenclaw
st.markdown('<h3 style="color:#000A90;">Ravenclaw</h3>', unsafe_allow_html=True)
st.write(
    "**Core Values:** Intelligence, Wisdom, Creativity  \n"
    "**Founder:** Rowena Ravenclaw  \n"
    "**Notable Students:** Luna Lovegood, Cho Chang  \n"
    "**House Colors:** Blue & Silver (or Gold in some versions)  \n"
    "**House Mascot:** Eagle"
)

# Slytherin
st.markdown('<h3 style="color:#1A472A;">Slytherin</h3>', unsafe_allow_html=True)
st.write(
    "**Core Values:** Ambition, Cunning, Resourcefulness  \n"
    "**Founder:** Salazar Slytherin  \n"
    "**Notable Students:** Severus Snape, Draco Malfoy, Tom Riddle (Lord Voldemort)  \n"
    "**House Colors:** Green & Silver  \n"
    "**House Mascot:** Snake"
)

st.markdown("---")

st.markdown("## The criteria for sorting students into Hogwarts houses.")
st.write("Students are sorted based on their personality traits. "
"Those who are brave and enjoy challenges are placed in Gryffindor. "
"Students who are honest and sincere belong to Hufflepuff. "
"Those with intelligence and creativity are sorted into Ravenclaw. "
"If a student is ambitious and inclined toward the Dark Arts, they will be placed in Slytherin.  ")

st.markdown("---")

st.markdown("## Data Overview")

import streamlit as st
import pandas as pd

st.title("Hogwarts's Student")

# สร้าง DataFrame
data = {
    "Blood Status": ["Half-blood", "Muggle-born", "Pure-blood", "Pure-blood","Pure-blood","Muggle-born"],
    "Bravery":              ["9", "6", "1", "9", "5" ,"7"],
    "Intelligence":         ["4", "8", "4", "1", "9" ,"6"],
    "Loyalty":              ["7", "4", "7", "3", "7" ,"2"],
    "Ambition":             ["5", "1", "7", "4", "3" ,"8"],
    "Dark Arts Knowledge":  ["0", "9", "1", "1", "3" ,"10"],
    "Dueling Skills":       ["8", "6", "4", "9", "6" ,"9"],
    "Creativity":           ["8", "2", "4", "10", "7" ,"7"],
    "House":                ["Gryffindor", "Ravenclaw", "Hufflepuff", "Gryffindor", "Ravenclaw" ,"Slytherin"]
}

df = pd.DataFrame(data)
st.markdown(
    """
    <style>
        .dataframe-table {
            width: 100%;
            min-width: 800px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# แสดงตารางแบบขยายได้
st.dataframe(df, width=1200)
st.markdown("[Harry Potter Sorting Dataset](https://www.kaggle.com/datasets/sahityapalacharla/harry-potter-sorting-dataset?fbclid=IwY2xjawJCebtleHRuA2FlbQIxMAABHdWF5TqFmnOPDHoYZSp1SiEL9h9CyZgzduGwXrSIPhXuFhl0awEBv_C1PQ_aem_2rcr56EWlQ1yQrNf8tM5Pw)")

st.title("Feature Explanation")

with st.expander("**Blood Status**"):
    st.write("Represents the magical heritage of a wizard. It can be one of the following:")
    st.markdown("- **Pure-blood**: Both parents are wizards.")
    st.markdown("- **Half-blood**: One parent is a wizard, and the other is a Muggle.")
    st.markdown("- **Muggle-born**: Both parents are non-magical (Muggles).")

with st.expander("**Bravery**"):
    st.write("Bravery is the willingness to face danger and stand up for what is right, even in the face of fear.")

with st.expander("**Intelligence**"):
    st.write("Intelligence represents one's ability to learn, analyze, and think critically.")

with st.expander("**Loyalty**"):
    st.write("Loyalty reflects devotion and faithfulness to friends, family, and one's house.")

with st.expander("**Ambition**"):
    st.write("Ambition is the drive to achieve one's goals, often with strong determination.")

with st.expander("**Dark Arts Knowledge**"):
    st.write("Measures one's understanding of dark magic, whether for study, defense, or power.")

with st.expander("**Dueling Skills**"):
    st.write("Represents a wizard's ability to engage in magical combat and defend themselves effectively.")

with st.expander("**Creativity**"):
    st.write("Creativity reflects one's ability to think outside the box and find unique solutions.")

with st.expander("**House**"):
    st.write("Determines which of the four Hogwarts houses a student belongs to: Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.")

st.markdown("---")

st.header("Exploratory Data Analysis (EDA)")
st.write("Start by writing code to check for null values.")
code = '''df.isnull().any()'''
st.code(code,language='python')

isnulldata = {
    "Feature": [
        "Blood Status", "Bravery", "Intelligence", "Loyalty",
        "Dark Arts Knowledge", "Quidditch Skills", "Dueling Skills", 
        "Creativity", "House"
    ],
    "ISnull": [False, False, False, False, False, False, False, False, False]  # ใช้ Boolean จริง ๆ
}

# สร้าง DataFrame
isnulldf = pd.DataFrame(isnulldata)

# แสดงตารางใน Streamlit
st.subheader("Null Value Table")
st.table(isnulldf)
st.write("Then, convert the values of blood type and house to Numerical")
code2 = '''df['Blood Status'] = df['Blood Status'].map({'Muggle-born': 0, 'Half-blood': 1, 'Pure-blood': 2})
df['House'] = df['House'].map({'Gryffindor': 1, 'Ravenclaw': 2, 'Hufflepuff': 3, 'Slytherin': 4})'''
st.code(code2,language='python')
st.image("Picture/pictureforweb/Hogwartdata.png")
st.write("Drop certain features that are not relevant to the factors influencing Hogwarts house selection to improve the accuracy of the machine learning model.  "
"Then, split the data into training and testing sets using an 70:30 ratio.")
code3 = ''' X = df.drop(['Blood Status','Loyalty','Quidditch Skills','Dueling Skills','House'], axis=1)
y = df['House']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) '''
st.code(code3,language='python')

st.markdown("---")

st.title("Model Used")


# สร้าง Columns 2 ช่อง
col1, col2 = st.columns([1, 1])  

with col1:
    st.subheader("K-Nearest Neighbors (KNN)")
    st.write(
        "K-Nearest Neighbors (KNN) is a simple, non-parametric machine learning algorithm that classifies data points "
        "based on the majority class of their K nearest neighbors in feature space. "
        "It works by measuring distance (e.g., Euclidean) and assigning the most common label among the closest K points."
    )

with col2:
    st.subheader("Hyperparameters")
    st.markdown(":blue[**Number of Neighbors:**] Determines how many closest data points are considered when classifying a new point. ")
    
    st.markdown(":blue[**Distance Metric:**] The measure used to calculate the distance between data points (e.g., Euclidean, Manhattan).")

    # สร้าง Columns 2 ช่อง
col1, col2 = st.columns([1, 1]) 

with col1:
    st.subheader("Support Vector Machine (SVM)")
    st.write(
        " is a supervised learning algorithm that classifies data by finding the optimal hyperplane "
        "that maximizes the margin between different classes. It uses support vectors (critical data points)"
        "and can handle linear and non-linear classification using kernel tricks."
    )

with col2:
    st.subheader("Hyperparameters")
    st.markdown(":blue[**C:**] Regularization parameter that controls the trade-off between margin maximization and error minimization. ")
    
    st.markdown(":blue[**Kernel:**] Defines the function used to map input data into higher dimensions. Common choices include linear, polynomial, and RBF kernels.")

st.markdown("---")

st.header("Model Details")
code4 = ''' knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(data, House)'''
st.subheader("K-Nearest Neighbors (KNN)")
col1, col2 = st.columns([1.5, 1]) 
with col1:
    st.code(code4,language='python')
with col2:
    st.markdown("  Creates a KNN model with K = 5 (considers the 5 nearest neighbors).  "
    "Trains the model using data (features) and House (labels).")

code5 = ''' linear = svm.SVC(kernel='linear', C=0.1)
linear.fit(x_train, y_train)'''
st.subheader("Support Vector Machine (SVM)")
col1, col2 = st.columns([1.5, 1]) 
with col1:
    st.code(code5,language='python')
with col2:
    st.markdown(" A model that finds the best boundary between different classes using a linear kernel with a regularization parameter of 0.1.")

st.markdown("---")

st.header("Model Evaluation")
st.subheader("K-Nearest Neighbors (KNN)")
code6 = ''' from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(data, House)

# 4. Predict using the model
y_pred = knn.predict(x_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # 'macro' for multi-class
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# 6. Display results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)'''
st.code(code6,language='python')

result = ''' Accuracy: 0.9566666666666667
Precision: 0.9551295518207283
Recall: 0.9566391941391942
F1-Score: 0.9551917598071444
Confusion Matrix:
[[62  0  3  0]
 [ 1 83  0  0]
 [ 7  2 69  0]
 [ 0  0  0 73]]'''
st.code(result,language='python')
st.subheader("Support Vector Machine (SVM)")
code7 = ''' print("Train set accuracy = " + str(linear.score(x_train, y_train)))
print("Test set accuracy = " + str(linear.score(x_test, y_test)))'''
st.code(code7,language='python')

result2 = ''' Train set accuracy = 0.9657142857142857
Test set accuracy = 0.9533333333333334'''
st.code(result2,language='python')