import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

st.set_page_config(page_title="Fitbit Unsupervised Dashboard", layout="wide")

st.title("🏃 Fitbit Unsupervised Learning Dashboard")
st.write("Cluster Fitbit users based on activity patterns")

uploaded_file = st.file_uploader("Upload Fitbit CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Dataset")
    st.dataframe(df.head())
# Feature selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    selected_features = st.multiselect(
        "Select features for clustering",
        numeric_cols,
        default=numeric_cols[:6]
    )

    if len(selected_features) > 1:

        X = df[selected_features]
#Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
#PCA
        st.subheader("📉 PCA for 2D Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        st.dataframe(pca_df.head())
#Elbow method
        st.subheader("📊 Elbow Method")
        wcss = []

        for i in range(1, 7):
            model = KMeans(n_clusters=i, random_state=42)
            model.fit(X_pca)
            wcss.append(model.inertia_)

        fig_elbow, ax = plt.subplots()
        ax.plot(range(1, 7), wcss, marker="o")
        ax.set_xlabel("Clusters")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method")
        st.pyplot(fig_elbow)
#Kmeans
        st.subheader("🎯 KMeans Clustering")
        k = st.slider("Select number of clusters", 2, 6, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df["KMeans_Cluster"] = kmeans.fit_predict(X_pca)

        fig_kmeans = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df["KMeans_Cluster"].astype(str),
            title="KMeans User Clusters"
        )
        st.plotly_chart(fig_kmeans, use_container_width=True)

#DBSCAN
        st.subheader("🔍 DBSCAN Clustering")
        eps = st.slider("Select eps", 0.1, 5.0, 0.5)
        min_samples = st.slider("Select min_samples", 2, 10, 5)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df["DBSCAN_Cluster"] = dbscan.fit_predict(X_pca)

        fig_dbscan = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df["DBSCAN_Cluster"].astype(str),
            title="DBSCAN User Clusters"
        )
        st.plotly_chart(fig_dbscan, use_container_width=True)
#Cluster insights
        st.subheader("📌 Cluster Insights")
        cluster_summary = df.groupby("KMeans_Cluster")[selected_features].mean()
        st.dataframe(cluster_summary)
#Download clustered dataset
        st.subheader("⬇ Download Clustered Dataset")
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="fitbit_clustered.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please select at least 2 features.")

else:
    st.info("Upload your Fitbit CSV file to start clustering.")