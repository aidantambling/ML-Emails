from collections import Counter
import pandas as pd
import os
from email.parser import Parser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def email_analyse(inputfile, to_email_list, from_email_list, email_body):
    print("Debug")
    # with open(inputfile, "r") as f:
    #     data = f.read()
    #
    # email = Parser().parsestr(data)
    #
    #
    # # if email['to']:
    # #     email_to = email['to']
    # #     email_to = email_to.replace("\n", "")
    # #     email_to = email_to.replace("\t", "")
    # #     email_to = email_to.replace(" ", "")
    # #     email_to = email_to.split(",")
    # #     for email_to_1 in email_to:
    # #         to_email_list.append(email_to_1)
    # #
    # #
    # # from_email_list.append(email['from'])
    #
    # email_body.append(email.get_payload())


rootdir = "C:\\Users\Aidan\\PycharmProjects\\emailClassification\\maildir\\lay-k"
to_email_list = []
from_email_list = []
email_body = []
# for directory, subdirectory, filenames in os.walk(rootdir):
#     for filename in filenames:
#         email_analyse(os.path.join(directory, filename), to_email_list, from_email_list, email_body )

# print("\nTo email adresses: \n")
# print(Counter(to_email_list).most_common(10))
#
# print("\nFrom email adresses: \n")
# print(Counter(from_email_list).most_common(10))




tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(email_body)

num_clusters = 3  # Adjust this based on your data
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(tfidf_matrix)

cluster_labels = kmeans.labels_

clustered_emails = {}
for email_idx, cluster_label in enumerate(cluster_labels):
    print("Debug")
    # if cluster_label not in clustered_emails:
    #     clustered_emails[cluster_label] = [email_body[email_idx]]
    # else:
    #     clustered_emails[cluster_label].append(email_body[email_idx])

for cluster_label, email_list in clustered_emails.items():
    print("Debug")
    # print(f"Cluster {cluster_label}:")
    # if len(email_list) > 0:
    #     print(email_list[0])  # Print the first email in the cluster