import argparse
import numpy as np
import json
from lr_4_task_9 import pearson_score, euclidean_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser


def find_similar_users(dataset, user, num_users, metric='pearson'):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Compute Pearson score between user and all the users in the dataset
    if metric == 'pearson':
        scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if user != x])
    else:
        scores = np.array([[x, euclidean_score(dataset, user, x)] for x in dataset if user != x])

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Extract top 'k' indices
    top_k = scores_sorted[:num_users]

    return scores[top_k]


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    raitings_file = 'lab4/data/ratings.json'
    with open(raitings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nUsers similar to " + user + ":\n")
    similar_users = find_similar_users(data, user, 3)
    print("User\t\tSimilarity score\n")
    print("-" * 41)
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))